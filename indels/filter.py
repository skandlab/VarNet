import os
import re
import sys
import json
import pysam
import argparse
import numpy as np

# from guppy import hpy
from math import ceil
from time import time
from itertools import chain
from joblib import Parallel, delayed

# from os.path import dirname, abspath
# repo_base = dirname(dirname(abspath(__file__)))
# sys.path.append(repo_base)

from snvs.generate_training_data import is_usable_read

import indels.constants_filter as c

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_constants(json_name=None):
    """
    if a JSON string containing constants.py variables exists, load it into a global Struct named c
    if not, import constants.py as c directly
    """
    global c
    if json_name is not None:
        with open(json_name, 'r') as f:
            c = Struct(**json.load(f))
    else:
        # c = __import__("constants_filter")
        from . import constants_filter as c

def parse_args():
    parser = argparse.ArgumentParser(description="Finds and extracts all potential indels in a genome")
    parser.add_argument('--timestamp', default='')
    parser.add_argument('--data_path')
    parser.add_argument('--data_name')
    parser.add_argument('--num_nodes')
    parser.add_argument('--node_no')
    parser.add_argument('--num_processes', default=1)

    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
    print("=========================================================================")
    if args.node_no == 'final':
        postprocessing = True
        print("RUNNING FINAL generate_candidates.py TO CONCATENATE AND VALIDATE OUTPUT FILES")
    else:
        postprocessing = False      
        print(("RUNNING generate_candidates.py ON NODE {} OUT OF {}".format(args.node_no, args.num_nodes)))
    print("=========================================================================")

    print("\nARGUMENTS:")
    print(("timestamp", args.timestamp))
    print(("data_path", args.data_path))
    print(("data_name", args.data_name))
    print(("num_nodes", args.num_nodes))
    print(("node_no", args.node_no))
    print(("num_processes", args.num_processes))

    sys.path.append(c.path_to_smudl)

    ref_file = pysam.FastaFile(c.ref_path)
    num_threads = int(args.num_processes)
else:
    load_constants()  

# === PROGRAM HELPERS === #

def get_total_coverage(bamfile, chrX, position):
    """
    for column in bamfile.pileup(chrX, position, position + 1):
        if column.pos == position:
            coverage = column.n
    """
    coverage = 0
    for read in bamfile.fetch(chrX, position, position + 1):
        if is_usable_read(read):
            coverage += 1
    return coverage
    # THIS ALSO HAS MEMORY LEAK - > 50 GB FOR CHRMT

def compute_voss_fft(chrX, pos):
    """
    unfinished. probably useful as a feature but likely not in prefiltering
    """
    sequence = ref_file.fetch(chrX, max(pos - 10, 0), pos + 10).upper()

    voss_encoded_seq = (1+1j)*np.array([x == 'A' for x in sequence]) + \
                       (1-1j)*np.array([x == 'T' for x in sequence]) + \
                       (-1-1j)*np.array([x == 'C' for x in sequence]) + \
                       (-1+1j)*np.array([x == 'G' for x in sequence])

    np.fft(voss_encoded_seq)

def find_STR_from_sequence(sequence):
    observed_STR = (1, 1.0) # (STR_size, STR_reps)
    hypothesis = 1 # hypothesis size of STR
    STR_idx = 0
    for i, base in enumerate(sequence[1:] + 'x'): # 'x' is a terminating character, necessary if the STR continues past the fetched region
        if base == sequence[STR_idx % hypothesis]:
            STR_idx += 1
        else:
            if STR_idx / hypothesis > 0:
                observed_STR = (hypothesis, 1.0*STR_idx / hypothesis + 1)
            hypothesis = i + 2
            STR_idx = 0
    return observed_STR

def find_STR_reps(chrX, pos):
    """
    """
    observed_STR_right = find_STR_from_sequence(ref_file.fetch(chrX, pos, pos + 20).upper())

    observed_STR_left = find_STR_from_sequence(ref_file.fetch(chrX, max(pos - 20, 0), pos).upper()[::-1])

    if observed_STR_right[0] > observed_STR_left[0]:
        return observed_STR_right
    elif observed_STR_left[0] > observed_STR_right[0]:
        return observed_STR_left
    else:
        return (observed_STR_right[0], observed_STR_right[1] + observed_STR_left[1])

def get_seq_disagreement(data):
    """
    """
    mode_seq = max([x[1] for x in data], key=[x[1] for x in data].count)
    total = 0
    for x in data:
        if x[0] == 'I':
            total += x[1] != mode_seq
        else:
            total += len(x[1]) != len(mode_seq)
    return 100*total/len(data), max(len(x[1]) for x in data) # get maximum observed length

def soft_clip_proximity(soft_clip_end, soft_clip_start, curr_index, indel_length, default):
    if soft_clip_end and soft_clip_start:
        return min(soft_clip_end - curr_index - indel_length, curr_index - soft_clip_start)
    elif soft_clip_end:
        return soft_clip_end - curr_index - indel_length
    elif soft_clip_start:
        return curr_index - soft_clip_start
    else:
        return default

def record_indels_detail(bamfile, chrX, start_pos, end_pos):
    """
    Iterate through all reads between start_pos and end_pos in chrX of bamfile, and identify
    all positions where at least c.MIN_FREQ reads contain an insertion or deletion.
    Positions are defined as the reference position at which the indel begins.

    Arguments:
        bamfile:    an opened pysam AlignmentFile
        is_tumor:   True if bamfile is tumor sample, False if bamfile is normal sample
        chrX, start_pos, end_pos:   the chrX label, starting position and ending position
                                    to fetch reads from bamfile
    Returns:
        recorded_indels : 
            a dict where key is <int> ref pos, and val is a list of 6-tuples: [(<str> indel type: 'I' or 'D', <str> indel sequence, <int> min base quality of indel, <int> avg base quality of indel, <int> read mapping quality, <int> soft clip proximity), ...]
        positions_dict :
            a dictionary where key is <int> position, and val is list of length 4: [<int> number of reads with an insertion at this position, <int> number of reads with a deletion at this position, <int> nearest neighbouring indel position, <int> coverage at this position]
    """
    # load reads into RAM
    reads = bamfile.fetch(chrX, start_pos, end_pos)

    # compile regexes before iterating to save time        
    segment_re = re.compile(r"\d+[IDMS]")
    soft_end_re = re.compile(r"\d+S$")
    soft_start_re = re.compile(r"\d+S")

    ####################################################################
    profile_memory_snapshots = 0
    # h = hpy()
    ####################################################################

    recorded_indels = {}
    positions_dict = {}
    previous_pos_list = [0 for i in range(15)] # buffer size of 15 => we expect absolute maximum of 15 indels to be found in backwards order

    for read in reads:
        cig = read.cigarstring
        
        # handle case where read has no cigarstring
        if cig != None and ('I' in cig or 'D' in cig) and is_usable_read(read):
            soft_clip_end = 0
            soft_clip_start = 0
            if 'S' in cig:
                if cig.endswith('S'):
                    soft_clip_end = sum(map(int, re.split(r"[A-Z]", cig)[:-1])) - int(soft_end_re.search(cig).group(0)[:-1])
                if 'S' in cig[:-1]:
                    soft_clip_start = int(soft_start_re.match(cig).group(0)[:-1])

            # the curr_index is the index in this read of an indel. start at 0
            ins_bases = 0
            del_bases = 0
            curr_index = 0

            while True:
                # seg is either "4I", "7D", "24S" or "121M" (4, 7, 24 and 121 could be other arbitrary ints)
                try:
                    seg = segment_re.match(cig).group(0)
                except AttributeError:
                    raise AttributeError("segment_re did not match cig: %s in chrX %s, pos %s" % (cig, chrX, str(read.reference_start + curr_index)))
                
                if seg[-1] == 'I':
                    # if there is an insertion at the current index, record it
                    position = read.reference_start + curr_index - ins_bases
                    quals = read.query_qualities[curr_index-del_bases:curr_index-del_bases+int(seg[:-1])]
                    
                    # make sure position is within searching range, and satisfies filtering requirements
                    if position >= start_pos and position < end_pos and sum(quals)/len(quals) >= c.MIN_AVG_BQ and read.mapping_quality >= c.MIN_MQ[0]:
                        if position in positions_dict:
                            # already encountered this position
                            recorded_indels[position].append(('I', read.query_sequence[curr_index-del_bases:curr_index-del_bases+int(seg[:-1])], min(quals), sum(quals)/len(quals), read.mapping_quality, soft_clip_proximity(soft_clip_end, soft_clip_start, curr_index, 0, end_pos-start_pos)))
                            positions_dict[position][0] += 1
                        else:
                            # first time encountering this position
                            recorded_indels[position] = [('I', read.query_sequence[curr_index-del_bases:curr_index-del_bases+int(seg[:-1])], min(quals), sum(quals)/len(quals), read.mapping_quality, soft_clip_proximity(soft_clip_end, soft_clip_start, curr_index, 0, end_pos-start_pos))]
                            if any(x < position for x in previous_pos_list):
                                positions_dict[position] = [1, 0, max([x for x in previous_pos_list if x < position] or [0]), 1]
                                if any([x > position for x in previous_pos_list]):
                                    positions_dict[min(x for x in previous_pos_list if x > position)][2] = position
                            else:
                                positions_dict[position] = [1, 0, 0, 1]
                            previous_pos_list.append(position)
                            previous_pos_list.remove(min(previous_pos_list))
                    ins_bases += int(seg[:-1])

                elif seg[-1] == 'D':
                    # if there is a deletion, record it AND advance curr_index to the end of the deletion
                    position = read.reference_start + curr_index - ins_bases

                    # make sure position is within searching range, and satisfies filtering requirements
                    if position >= start_pos and position < end_pos and read.mapping_quality >= c.MIN_MQ[1]:
                        if position in positions_dict:
                            # already encountered this position
                            recorded_indels[position].append(('D', 'D'*int(seg[:-1]), 0, 0, read.mapping_quality, soft_clip_proximity(soft_clip_end, soft_clip_start, curr_index, int(seg[:-1]), end_pos-start_pos)))
                            positions_dict[position][1] += 1
                        else:
                            # first time encountering this position
                            recorded_indels[position] = [('D', 'D'*int(seg[:-1]), 0, 0, read.mapping_quality, soft_clip_proximity(soft_clip_end, soft_clip_start, curr_index, int(seg[:-1]), end_pos-start_pos))]
                            if any(x < position for x in previous_pos_list):
                                positions_dict[position] = [0, 1, max(x for x in previous_pos_list if x < position), 1]
                                if any(x > position for x in previous_pos_list):
                                    positions_dict[min(x for x in previous_pos_list if x > position)][2] = position
                            else:
                                positions_dict[position] = [0, 1, 0, 1]
                            previous_pos_list.append(position)
                            previous_pos_list.remove(min(previous_pos_list))
                    del_bases += int(seg[:-1])

                    ####################################################################
                    if position > start_pos + profile_memory_snapshots*(end_pos-start_pos)/5:
                        print(("PROFILING: del at pos {} from start ({}% through)".format(position-start_pos, (100*(position-start_pos))/(end_pos-start_pos))))
                        print((h.heap()))
                    ####################################################################
                
                # chop off the part of cig we already read
                cig = cig[len(seg):]
                curr_index += int(seg[:-1])

                # escape the loop if all indels have been recorded
                if not ('I' in cig or 'D' in cig):
                    break

    # iterate through all positions starting at the last, calculated nearest indel neighbor for each one
    # in addition, reject all positions with count < MIN_FREQ
    position = (max(previous_pos_list), 2*end_pos) # current position, previous position    
    counted_positions = 0
    
    while positions_dict[position[0]][2] > 0:
        if positions_dict[position[0]][0] < c.MIN_FREQ[0] and positions_dict[position[0]][1] < c.MIN_FREQ[1]:
            del recorded_indels[position[0]]
        else:
            nearest_neighbour_prox = min(position[1] - position[0], position[0] - positions_dict[position[0]][2])
        position = (positions_dict[position[0]][2], position[0])
        counted_positions += 1
        positions_dict[position[1]][2] = nearest_neighbour_prox
    
    if counted_positions != len(positions_dict):
        print(("\nWARNING: could not find nearest neighbor for {} out of {} positions ({}%)".format(len(positions_dict) - counted_positions, len(positions_dict), ((len(positions_dict) - counted_positions)*100.0)/len(positions_dict))))

    return recorded_indels, positions_dict

def record_indels_simple(bamfile, chrX, start_pos, end_pos):
    """
    Iterate through all reads between start_pos and end_pos in chrX of bamfile, and identify
    all positions where at least one c.MIN_FREQ reads contain an insertion or deletion.
    Positions are defined as the reference position at which the indel begins.

    Arguments:
        bamfile:    an opened pysam AlignmentFile
        is_tumor:   True if bamfile is tumor sample, False if bamfile is normal sample
        chrX, start_pos, end_pos:   the chrX label, starting position and ending position
                                    to fetch reads from bamfile
    Returns:
        recorded_indels : 
            a set of ints, corresponding to the ref pos of an indel
    """
    # load reads into RAM
    reads = bamfile.fetch(chrX, start_pos, end_pos)

    # compile regexes before iterating to save time        
    segment_re = re.compile(r"\d+[IDMS]")

    recorded_indels = {}

    positions_dict = {}

    POSITIONS_COVERAGE = {} # e.g. { 1234: 20, .. }

    for read in reads:
        cig = read.cigarstring

        # handle case where read has no cigarstring
        if cig is not None and ('I' in cig or 'D' in cig) and is_usable_read(read):
            # the curr_index is the index in this read of an indel. start at 0
            curr_index = 0
            qual_index = 0
            
            while True:
                # seg is either "4I", "7D", "24S" or "121M" (4, 7, 24 and 121 could be other arbitrary ints)
                try:
                    seg = segment_re.match(cig).group(0)
                except AttributeError:
                    raise AttributeError("segment_re did not match cig: %s in chrX %s, pos %s" % (cig, chrX, str(read.reference_start + curr_index)))
                
                if seg[-1] == 'M' or seg[-1] == 'S':
                    # if there is a mapped segment at curr_index, advance curr_index to the end of the segment
                    curr_index += int(seg[:-1])
                    qual_index += int(seg[:-1])
               
                    if seg[-1] == 'M':
                        positions = [x for x in range(read.reference_start + curr_index - int(seg[:-1]), read.reference_start + curr_index)] 
                        for position in positions:
                            if position not in POSITIONS_COVERAGE:
                                POSITIONS_COVERAGE[position] = 1
                            else:
                                POSITIONS_COVERAGE[position] += 1 
 
                elif seg[-1] == 'I':
                    # if there is an insertion at the current index, record it
                    position = read.reference_start + curr_index
                    quals = read.query_qualities[qual_index:qual_index+int(seg[:-1])]
                    
                    #if position == start_pos: 
                    #    print((position, seg, read.query_sequence[qual_index:qual_index+int(seg[:-1])], read.mapping_quality, sum(quals)/len(quals)))

                    if position not in POSITIONS_COVERAGE:
                        POSITIONS_COVERAGE[position] = 1
                    else:
                        POSITIONS_COVERAGE[position] += 1
                   
                    # COLLECTING STATISTICS OF TRUE POSITIONS 
                    if 'true_positions' in globals():
                        assert len(list(true_positions.keys())) > 0
                        key = 'chrom%spos%s' % (chrX, position) 
                        if key in true_positions:
                            TRUE_MQS.append(read.mapping_quality)                
 
                    # make sure position is within searching range, and satisfies filtering requirements
                    if position >= start_pos and position < end_pos and sum(quals)/len(quals) >= c.MIN_AVG_BQ and read.mapping_quality >= c.MIN_MQ[0]:   
                        recorded_indels[position] = True
                        if position in positions_dict:
                            # we have encountered this position before, so increment its count
                            positions_dict[position][0] += 1 
                        else:
                            # we have never encountered this position before, so create a count for it
                            positions_dict[position] = [1, 0]
                    qual_index += int(seg[:-1])
                
                elif seg[-1] == 'D':
                    # if there is a deletion, record it AND advance curr_index to the end of the deletion
                    position = read.reference_start + curr_index
                    
                    #if position == start_pos: 
                    #    print((seg, read.query_sequence[qual_index:qual_index+int(seg[:-1])], read.mapping_quality))              

                    if position not in POSITIONS_COVERAGE:
                        POSITIONS_COVERAGE[position] = 1
                    else:   
                        POSITIONS_COVERAGE[position] += 1
 
                    # COLLECTING STATISTICS OF TRUE POSITIONS 
                    if 'true_positions' in globals():
                        assert len(list(true_positions.keys())) > 0
                        key = 'chrom%spos%s' % (chrX, position)
                        if key in true_positions:
                            TRUE_MQS.append(read.mapping_quality)

                    # make sure position is within searching range, and satisfies filtering requirements
                    if position >= start_pos and position < end_pos and read.mapping_quality >= c.MIN_MQ[1]:
                        recorded_indels[position] = False
                        if position in positions_dict:
                            # we have encountered this position before, so increment its count
                            positions_dict[position][1] += 1                            
                        else:
                            # we have never encountered this position before, so create a count for it
                            positions_dict[position] = [0, 1]
                            
                    curr_index += int(seg[:-1])

                # chop off the part of cig we already read
                cig = cig[len(seg):]

                # escape the loop if all indels have been recorded
                if not ('I' in cig or 'D' in cig):
                    break
        
    #recorded_indels = list(recorded_indels.keys())

    # reject all positions with count < MIN_FREQ
    for position, count in list(positions_dict.items()):

        if 'true_positions' in globals():
            key = 'chrom%spos%s' % (chrX, position)
            if key in true_positions:
                TRUE_FREQS.append(max(count)) # FREQ, count = [insertion count, deletion count]

        if count[0] < c.MIN_FREQ[0] and count[1] < c.MIN_FREQ[1]:
            recorded_indels.pop(position)
            #del recorded_indels[recorded_indels.index(position)]
        else:
            # AF
            #recorded_indels[position] = (count[0] + count[1])/POSITIONS_COVERAGE[position]
            # READ COUNT
            recorded_indels[position] = count[0] + count[1]

    return recorded_indels

def filter_indels(candidates_folder, bamfile_n_path, bamfile_t_path, regions, batch_num):

    output_file = os.path.join(candidates_folder, 'batch_%s.csv' % str(batch_num) )
    
    if os.path.exists(output_file):
        print(("INDEL BATCH COMPLETE: %s" % output_file))
        return

    bamfile_n = pysam.AlignmentFile(bamfile_n_path, "rb")
    bamfile_t = pysam.AlignmentFile(bamfile_t_path, "rb")

    recorded_indels = []

    for region in regions:
        chrom, start, end = region[0], region[1], region[2]

        if c.ONLY_FILTER_TUMOR:
            recorded_indels = record_indels_simple(bamfile_t, chrom, start, end)

        else:
            # record_indels_simple returns dict
            # positions returned by record_indels_simple are the 0-indexed position of the beginning of actual indel
            # not one base before indel starts

            recorded_indels_n = record_indels_simple(bamfile_n, chrom, start, end)
            recorded_indels_t = record_indels_simple(bamfile_t, chrom, start, end)

            #recorded_indels_n = []
            #for position in recorded_indels_t:
            #    recorded_indels_n.extend(record_indels_simple(bamfile_n, chrom, position, position+1))

            # indels recorded in t should not be in n
            for p in list(recorded_indels_t.keys()):
                # accept if p not in normal or, 

                # if p is in normal, the type of indel should be different. Insertions are assigned True, deletions are False
                #if p not in recorded_indels_n or recorded_indels_t[p] != recorded_indels_n[p]:

                # if p in normal, freqs and in tumor and normal should be different by some margin 
                #if p in recorded_indels_n: margin = (recorded_indels_t[p] + recorded_indels_n[p])/15
                margin = 3 # 4 reads
                #margin = 0.02 # 2% AF difference
                if p not in recorded_indels_n or abs(recorded_indels_t[p] - recorded_indels_n[p]) > margin:
                    recorded_indels.append((chrom, p))            

            # indels recorded in n should not be in t
            #for p in recorded_indels_n.keys():
            #    if p not in recorded_indels_t:
            #        recorded_indels.append(p)
        
    with open(output_file, 'a') as f:
        for pos in recorded_indels:
            f.write('%s\t%d\n' % (pos[0], int(pos[1])-1)) # 0-indexed position of the base before indel starts

    print(("COMPLETED INDEL BATCH", output_file))

    if 'true_positions' in globals():
        return [TRUE_FREQS, TRUE_MQS] 


def divide_chromosomes_into_batches(superbatch, num_threads):
    """
    create a list of batches from the chromosomes in superbatch. each batch will be run on its own thread
    if there are unused threads, the largest chromosomes will be split among them
    batches is a list of 4-tuples:
        [
            (chrX, bamfile_index, start_pos, end_pos, split_chrom_num),
            ...
        ]
    where split_chrom_num is an integer, to identify chromosomes that have been split into multiple batches
    i.e. if chr2 of bamfile 0 has been split into 3 batches, the batches will have split_chrom_num 0, 1 and 2
    split_chrom_num is None if the chromosome has not been split
    """
    batches = [[] for i in range(num_threads)]
    split_chrom_dict = {}

    for k, pair in enumerate(superbatch):
        batches[k] = (pair[0], pair[1], 0, get_ref_chrom_end_index(pair[0]), None)
        split_chrom_dict[(pair[0], pair[1])] = 1

    batch_index = len(superbatch) # the index of the current empty slot in batches

    while batch_index < num_threads:
        # get the index of the batch in batches that maximizes end_pos - start_pos
        batch_to_split = batches.index(max(batches[:batch_index], key=lambda x: x[3] - x[2]))
        start_pos = batches[batch_to_split][2]
        end_pos = batches[batch_to_split][3]

        # split the batch in half, and place the top half in the empty slot
        focal_batch = batches[batch_to_split] 
        batches[batch_index] = (focal_batch[0], focal_batch[1], start_pos + (end_pos - start_pos)/2, end_pos, split_chrom_dict[(focal_batch[0], focal_batch[1])])
        split_chrom_dict[(focal_batch[0], focal_batch[1])] += 1
        
        if focal_batch[4] == None:
            batches[batch_to_split] = (focal_batch[0], focal_batch[1], start_pos, start_pos + (end_pos - start_pos)/2, 0)
        else:
            batches[batch_to_split] = (focal_batch[0], focal_batch[1], start_pos, start_pos + (end_pos - start_pos)/2, focal_batch[4])

        # move on to the next empty slot
        batch_index += 1

    return batches

def get_ref_chrom_end_index(label):
    """
    Produce the final index in the reference chromosome corresponding to char index ('1', 'Y', etc.)
    """
    if label == 'X':
        return ref_file.lengths[22] - 1
    elif label == 'Y':
        return ref_file.lengths[23] - 1
    elif label == 'MT':
        return ref_file.lengths[24] - 1
    else:
        try:
            return ref_file.lengths[int(label) - 1] - 1
        except ValueError:
            error("{} is not a valid chromosome label".format(label))

def get_chromosome_bins(num_nodes, num_bams):
    """
    Return a list of bins containing (chrX, bamfile index) pairs:
    e.g.    [
                [('1', 0), ('5', 1), ('X', 2)],
                ...
            ]
    """
    # first, create chromosome_bins with chromosomes indexed largest to smallest
    chromosome_bins = [[] for i in range(num_nodes)]
    for chrx in range(c.num_chromosomes * num_bams):
        chromosome_bins[chrx % num_nodes] += [(c.chromosomes_list[chrx / num_bams], chrx % num_bams)]

    if all(chromosome_bins):

        # if some bins have one fewer chromosome, send the largest chromosomes to those bins to compensate
        for chrx in range(min((c.num_chromosomes * num_bams) % num_nodes, num_nodes - ((c.num_chromosomes * num_bams) % num_nodes))):
            temp_pair = chromosome_bins[chrx][0]
            chromosome_bins[chrx][0] = chromosome_bins[-chrx - 1][-1]
            chromosome_bins[-chrx - 1][-1] = temp_pair

    return chromosome_bins

def get_candidates_folder(data_name, num_nodes, postprocessing=False):
    """
    returns the folder for this set of candidates data, given the candidates_path and data_name
    If run during postprocessing, the candidates folder should already exist
    If not run during postprocessing, creates the folder if it does not exist
    """
    candidates_folder = os.path.join(c.candidates_path, "{}_{}".format(data_name, c.candidates_encoding_name))
    
    if postprocessing:
        if not os.path.exists(candidates_folder):
            raise IOError("candidates folder expected at {} does not exist".format(candidates_folder))
    else:
        if os.path.exists(candidates_folder):
            print(("\nCandidate positions folder already exists: {}".format(candidates_folder)))
        else:        
            if int(args.num_nodes) > 1:
                try:
                    os.makedirs(candidates_folder)
                    print(("\nCreating folder for candidate positions: {}".format(candidates_folder)))
                except OSError:
                    print(("\nCandidate positions folder was created by another node: {}".format(candidates_folder)))
            else:
                print(("\nCreating folder for candidate positions: {}".format(candidates_folder)))
                os.makedirs(candidates_folder)
    return candidates_folder


# === EXECUTE MAIN === #

if __name__ == '__main__':
    from .analyze_candidates import analyze
 
    TIME = time()
    print("\nPREFILTER SETTINGS:")
    for setting in c.PREFILTER_SETTINGS:
        print((setting[0], setting[1]))

    # this script has two different functions, depending on whether node_no == 'final'
    if postprocessing:
        # === Concatenate csv files and validate candidate positions === #

        # get candidate positions folder and analysis output file
        candidates_folder = get_candidates_folder(args.data_name, args.num_nodes, True)
        print("")

        # analyze_candidates.py performs three functions:
        #  - concatenate separate csv files into a single file per patient
        #  - produce Analysis file for each patient to summarize indel distribution data
        #  - validate candidate positions against ncomms file (if it exists)
        analyze(candidates_folder, num_threads, json_name=args.json_name, patients_path=args.data_path)
        #os.system("python analyze_candidates.py --json_name {} --candidates_folder {} --num_processes {} --running_alone no".format(args.json_name, candidates_folder, num_threads))        

    else:
        # === Open bamfiles in data_path, find all candidate positions and write them to csv files === #    
        
        # throw IOError if data_path does not exist
        if not os.path.exists(args.data_path):
            raise IOError("bamfile path not found: {}".format(args.data_path))

        # create candidate positions folder
        candidates_folder = get_candidates_folder(args.data_name, args.num_nodes, False)

        # find all the patients in the data path
        patients_prelim = []
        for patient in os.listdir(args.data_path):
            if os.path.isdir(os.path.join(args.data_path, patient)):
                normal_bams = [bam for bam in os.listdir(os.path.join(args.data_path, patient)) if bam.endswith('.bam') and ('-N' in bam or 'normal' in bam)]
                tumor_bams = [bam for bam in os.listdir(os.path.join(args.data_path, patient)) if bam.endswith('.bam') and ('-T' in bam or 'tumor' in bam)]
                if normal_bams or tumor_bams:
                    if not normal_bams:
                        raise IOError("found no normal bamfiles in folder {}".format(os.path.join(args.data_path, patient)))
                    elif not tumor_bams:
                        raise IOError("found no tumor bamfiles in folder {}".format(os.path.join(args.data_path, patient)))
                    elif len(normal_bams) > 1:
                        raise IOError("found more than one normal bamfile in folder {}, expected only one".format(os.path.join(args.data_path, patient)))
                    elif len(tumor_bams) > 1:
                        raise IOError("found more than one tumor bamfile in folder {}, expected only one".format(os.path.join(args.data_path, patient)))
                    else:
                        patients_prelim.append((patient, normal_bams[0], tumor_bams[0]))
                else:
                    print(("warning: found a folder in data_path that did not contain any bam files: {}. skipping".format(os.path.join(args.data_path, patient))))

        # each patient will recieve their own candidate positions file
        # however, to avoid parallel IO issues, we create separate files for each batch and then concatenate them at the end
        # first check if a final concatenated training data file already exists
        patients = []
        
        for patient in patients_prelim:
            candidates_file = os.path.join( candidates_folder, patient[0] + '.csv' )
            
            if os.path.exists(candidates_file):
                print(("Candidate positions file already exists for patient {}. Skipping".format(patient[0])))
            else:
                patients.append(patient)

        if len(patients) == 0:
            print("\nNo unprocessed patients were found. Exiting")
        else:
            # there are num_chromosomes*len(patients) chromosomes to filter
            # divide them into num_nodes bins such that all bins are approximately equal in size
            chromosome_bins = get_chromosome_bins(int(args.num_nodes), len(patients))

            # get the subset of chromosomes that have been assigned to this node, and remove any that have already been prefiltered
            target_chromosomes = [pair for pair in chromosome_bins[int(args.node_no) - 1]]

            if target_chromosomes:
                print("\nCHROMOSOMES TO PRFILTER ON THIS NODE:")     
                
                for pair in target_chromosomes:
                    print(("chr {} of patient {}: {}".format(pair[0], pair[1], patients[pair[1]][0])))

                # if there are more chromosomes than threads, split target_chromosomes into superbatches
                if len(target_chromosomes) > num_threads:
                    num_superbatches = int(ceil(len(target_chromosomes) / float(num_threads)))
                    superbatches = [target_chromosomes[i::num_superbatches] for i in range(num_superbatches)]
                else:
                    superbatches = [target_chromosomes]

                # spawn num_threads threads running filter_indels for each superbatch
                for superidx, superbatch in enumerate(superbatches):

                    print("\n=========================================================================")
                    print(("superbatch {} out of {}:".format(superidx + 1, len(superbatches))))

                    # assign one thread to each chromosome in superbatch
                    # if there are unused threads, the largest chromosomes will be split among them
                    # batches is a list of 5-tuples:
                    #       [  (chrX, patient_index, start_pos, end_pos, split_chrom_num),  ...  ]
                    # where split_chrom_num is an integer:
                    #       0 if that chromosome has not been split up between multiple batches
                    #       n if it has, where each batch it was split into has a unique n
                    
                    batches = divide_chromosomes_into_batches(superbatch, num_threads)

                    for batch in batches:
                        print(("chr {} of bamfile {}: batch {} (from pos {} to {})".format(batch[0], batch[1], batch[4], batch[2], batch[3])))

                    # WARNING: don't put any print statements inside filter_indels
                    #          or any of the functions it calls (other than temporary ones for debugging)
                    #          the parallelization makes print statements interrupt eachother
                    #
                    # any output bound for the logfile in filter_indels will be output as log_string
                    # parallel_output is:
                    #   [log_string_thread_1, log_string_thread_2, ... ]
                    
                    parallel_output = Parallel(n_jobs=num_threads, max_nbytes=1)(delayed(filter_indels)(args.data_path, patients[batch[1]], batch[0], batch[2], batch[3], batch[4], candidates_folder) for batch in batches)
                    
                    print(('\n'.join(parallel_output)))
                    print("\nCompleted batches")
            
            else:
                print("\nNO CHROMOSOMES TO PREFILTER ON THIS NODE. EXITING")

    print(("\nTIME ELAPSED ON NODE {}: {} SECONDS\n".format(args.node_no, time() - TIME)))




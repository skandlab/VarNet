# ===te_input_tensor_for_position PROGRAM NOTES ===

# NOTE: We set up an array with 7 channels where:
    #   0 => sequence comparision of reference to normal
    #   1 => base quality of normal
    #   2 => strand direction of normal
    #   3 => reference sequence
    #   4 => sequence comparision of reference to tumor
    #   5 => base quality of tumor
    #   6 => strand direction of tumor

# NOTE: Each image consists of 3 channels: (r,g,b)

# Create one image for the 3 normal channels:
    # (r) sequence comparison to ref
    # (g) base quality
    # (b) strand direction

# Create one image for the 3 tumor channels:
    # (r) sequence comparison to ref
    # (g) base quality
    # (b) strand direction

"""
Soft-clipped bases are not used in variant caling. Soft clip bases are bases at the beginning or end of the read that do not align anywhere on the reference.
These bases are likely introduced due to sequencing errors and are not using in variant calling.

Bases marked as 'N' in the read have a very low base quality and are certain to be wrong.

"""
import os
import numpy as np
import pandas as pd
import pysam
import re
import argparse
#from PIL import Image
from time import time
from joblib import Parallel, delayed, __version__
import snvs.constants as c
import datetime
import random

from snvs.compress_npy_helper import save_compressed_npy, load_compressed_npy
from utils import sample_reads_fn

# === FUNCTIONS FOR RUNNING PROGRAM IN COMMAND LINE ===

def parse_args():
    parser = argparse.ArgumentParser(description="Image and NDarray Generator")
    parser.add_argument('--path_to_bam_n', default='')
    parser.add_argument('--path_to_bam_t', default='')
    parser.add_argument('--path_to_labels', default='')
    parser.add_argument('--environment', default='aquila')
    parser.add_argument('--generate_all', default='no')
    parser.add_argument('--crc_data', default=False)
    parser.add_argument('--gastric_data', default=False)
    parser.add_argument('--liver_data', default=False)
    parser.add_argument('--lung_data', default=False)
    parser.add_argument('--sarcoma_data', default=False)
    parser.add_argument('--thyroid_data', default=False)
    parser.add_argument('--lymphoma_data', default=False)
    parser.add_argument('--goldset_data', default=False)
    parser.add_argument('--num_nodes')
    parser.add_argument('--node_no')
    parser.add_argument('--num_processes', default=1)
    
    parser.parse_args().num_nodes = int(parser.parse_args().num_nodes)
    parser.parse_args().node_no = int(parser.parse_args().node_no)

    if parser.parse_args().crc_data == 'yes':
        parser.parse_args().crc_data = True

    if parser.parse_args().gastric_data == 'yes':
        parser.parse_args().gastric_data = True

    if parser.parse_args().liver_data == 'yes':
        parser.parse_args().liver_data = True

    if parser.parse_args().goldset_data == 'yes':
        parser.parse_args().goldset_data = True

    if parser.parse_args().lung_data == 'yes':
        parser.parse_args().lung_data = True

    if parser.parse_args().sarcoma_data == 'yes':
        parser.parse_args().sarcoma_data = True

    if parser.parse_args().thyroid_data == 'yes':
        parser.parse_args().thyroid_data = True

    if parser.parse_args().lymphoma_data == 'yes':
        parser.parse_args().lymphoma_data = True

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.environment == 'workstation':
        ref_path = c.ref_path_on_workstation

    elif args.environment == 'aquila':
        ref_path = c.ref_path_on_aquila

    elif args.environment == 'nscc':
        ref_path = c.ref_path_on_nscc

    ref_file = pysam.FastaFile(ref_path) 

else:
    pass
    # ref_path = c.ref_path_on_nscc
    # ref_file = pysam.FastaFile(ref_path) 

# === PROGRAM HELPERS === #

def get_ref_base(pos, chrX, ref_file):
    """
    Positions are 0-indexed in pysam and the fetch function returns bases for the half-open interval (includes start position, excludes end position)
    http://pysam.readthedocs.io/en/latest/api.html#pysam.FastaFile.fetch
    """
    try:
        return ref_file.fetch(chrX, pos, pos + 1).upper() # base at pos in reference
    except KeyError:
        if chrX == 'chrM' or chrX == 'M':
            chrX = 'MT'
        else:
            chrX = chrX.replace('chr','')

        return ref_file.fetch(chrX, pos, pos + 1).upper() # base at pos in reference

    except Exception as e:
        print(('Err in get_ref_base:', e))
        raise

def get_base_val(base):
    # NOTE: could make the degenerate bases that encode 2 bases get a value of 10 and those that encode 3 bases get a value of 5, whereas any results in a value of 0
    base_to_color = {'A': 245, 'G': 175, 'T': 105, 'C': 35, 'D': 0, 'N': 0, 'W': 0,
                     'S': 0, 'M': 0, 'K': 0, 'R': 0, 'Y': 0, 'B': 0, 'H': 0, 'V': 0}
    if base not in list(base_to_color.keys()):
        raise KeyboardInterrupt
    return base_to_color.get(base, 0)

def is_usable_read(read):
    """
    Let's not filter reads with bad mapping quality. We'll let the net learn to use mapping quality values appropriately.
    There is no clarity on the range of mapping quality values that alignment software produce and they don't appear to comply with the SAM specification.
    """
    usable_read = (not (read.is_duplicate or read.is_supplementary or read.is_unmapped or read.is_secondary))
    return usable_read

def get_positions_to_fill(read):
    """
    This function removes soft clip bases from the beginning and end of the read.
    Soft clip bases are likely from sequencing errors and are not used by variant callers
    """
    # list of read/query and reference positions
    all_positions = read.get_aligned_pairs()
    
    soft_clip_bases_on_the_left = read.cigartuples[0][1] if read.cigartuples[0][0] == 4 else 0
    soft_clip_bases_on_the_right = read.cigartuples[-1][1] if read.cigartuples[-1][0] == 4 else 0

    if soft_clip_bases_on_the_right:
        all_positions = all_positions[soft_clip_bases_on_the_left:-soft_clip_bases_on_the_right]
    else:
        all_positions = all_positions[soft_clip_bases_on_the_left:]

    return all_positions

def get_reference(ref_file, chrX, pos, ref_dict=None, NUM_READS=None):
    start_pos = pos - c.FLANK
    end_pos = pos + c.FLANK + 1
    
    if start_pos < 0: start_pos = 0

    if ref_dict is None:
        try:
            ref_sequence = ref_file.fetch(chrX, start_pos, end_pos).upper() # corresponding reference sequence

        except KeyError:
            if chrX == 'chrM' or chrX == 'M':
                chrX = 'MT'
            else:
                chrX = chrX.replace('chr','')
        
            ref_sequence = ref_file.fetch(chrX, start_pos, end_pos).upper() # corresponding reference sequence        
  
        except Exception as e:
            print(('Err in get_reference:', e))
            raise
 
    else:
        ref_sequence = ref_dict[chrX][start_pos:end_pos]
    
    ref_nucs = list(ref_sequence) # nucleotides in reference

    if c.VARIABLE_INPUT_SIZE:
        if NUM_READS:
            ref_channel = np.zeros((NUM_READS, c.SEQ_LENGTH), dtype=np.float32)
        else:
            raise Exception("Variable input size but input height not provided")
    else:
        ref_channel = np.zeros((c.NUM_READS, c.SEQ_LENGTH), dtype=np.float32)

    for i, nuc in enumerate(ref_nucs):
        ref_channel[:, i] = get_base_val(nuc)

    return ref_channel

def get_mapping_quality_color(map_quality):
    if np.isclose(map_quality, 255.): # 255. means no available mapping quality
        return 0.
    else:
        return float(int(255.0 * (min(45, map_quality) / 45.0)))

def get_base_quality_color(quality):
    return int(255.0 * (min(45, quality) / 45.0))

def get_strand_color(on_positive_strand):
    return 125 if on_positive_strand else 250

def get_match_val(cigar_tag, read_base, ref_base):
    if (cigar_tag == 'M'):  # this means that this base aligns to reference
        if c.COMPARE_REF_BASE:
            if (read_base == ref_base): # base match
                return 0.
            else: # base mismatch
                return get_base_val(read_base)
        else:
            return get_base_val(read_base)

    elif (cigar_tag == 'S'): # soft clip
        return get_base_val(read_base) # this is okay, MQ will be 0
    elif (cigar_tag == 'D'): # deletion
        return 10.
    else:
        print((cigar_tag, read_base, ref_base))
        raise KeyboardInterrupt
        return

def encode_base_qualities(quals):
    out_arr = np.array([get_base_quality_color(qual) for qual in quals], dtype=np.float32)
    return out_arr

def encode_strand_dir(on_positive_strand, length):
    return np.ones((length,)) * get_strand_color(on_positive_strand)

def encode_mapping_quality(mapping_qual, length):
    return np.ones((length,)) * get_mapping_quality_color(mapping_qual)

def aligned_pair_for_reference_position(ref_pos, read):
    for pair in read.get_aligned_pairs():
        if pair[1] == ref_pos:
            return pair
    return (None, None)

def get_average_neighbouring_base_quality(read, ref_pos):
    """Calculates average base quality of bases on either side of a deletion in a read"""
    neighbouring_base_quality_amount, neighbouring_bases_count = 0.0, 0.0

    # get base quality of nearest available base on the left (not inserted base)
    pos = ref_pos - 1
    while(pos >= read.reference_start):
        pair = aligned_pair_for_reference_position(pos, read)
        if pair[0]: # check that it's not another deletion to the left
            neighbouring_base_quality_amount += read.query_qualities[pair[0]] 
            neighbouring_bases_count += 1
            break
        pos -= 1

    # get base quality of nearest available base on the right (not inserted base)
    pos = ref_pos + 1
    while(pos < read.reference_end):
        pair = aligned_pair_for_reference_position(pos, read)
        if pair[0]: # check that it's not another deletion to the left
            neighbouring_base_quality_amount += read.query_qualities[pair[0]] 
            neighbouring_bases_count += 1
            break
        pos += 1

    return neighbouring_base_quality_amount/neighbouring_bases_count

def get_bases_to_fill(read, chromosome, reference_start, reference_end, mutate=None):
    seq = ''
    encoded_seq = []
    quals = []

    all_pairs = get_positions_to_fill(read) # list of read/query and reference positions, soft clip bases removed
    read_starts_at = 0

    insert_indices = []

    'query_qualities is missing in reads sometimes. Fill it up with 45 (min base qual)'
    if read.query_qualities is None:
        read.query_qualities = [45 for u in range(len(read.query_sequence))]

    for i, pair in enumerate(all_pairs):
        read_pos = pair[0]
        ref_pos = pair[1]

        if ref_pos is None: # Ref pos is None for insertions
            # what is the best way to encode insertions for SNV calling?
            if c.ENCODE_INSERTIONS:
                # modify preceeding base
                if len(encoded_seq):
                    if all_pairs[i-1][1]: # this is the previous position's ref_pos. don't modify the preceeding base more once per insertion
                        encoded_seq[-1] += 20
                    
            # insert_indices += [ read_pos - 1 ] if read_pos != 0 else []

        elif ref_pos >= reference_start and ref_pos < reference_end:
            
            if not len(encoded_seq): # if this is the first base we are adding to the encoded read
                read_starts_at = ref_pos - reference_start

            if read_pos is None: # This means that there is a deletion
                seq += 'D'
                
                encoded_seq.append(get_match_val('D', None, None)) # ref_base not needed unless COMPARE_REF_BASE=TRUE. get_ref_base(ref_pos, chromosome)))

                # base quality for deletions
                # use average of the base qualities neighbouring it
                quals.append(get_average_neighbouring_base_quality(read, ref_pos))

            else:
                seq += read.query_sequence[read_pos]
                
                encoded_seq.append(get_match_val('M', read.query_sequence[read_pos], None)) # ref_base not needed unless COMPARE_REF_BASE=TRUE. get_ref_base(ref_pos, chromosome)))
    
                quals.append(read.query_qualities[read_pos])

    assert(len(seq) == len(quals))
    return seq, encoded_seq, quals, read_starts_at, insert_indices

def stack_read_in_image(read, img, row_num, col_i, start_pos, end_pos, chrX, ref_dict=None):
    sequence, encoded_bases, base_qualities, read_starts_at, insert_indices = get_bases_to_fill(read, chrX, start_pos, end_pos)
    encoded_quals = encode_base_qualities(base_qualities)

    processed_read_length = len(encoded_bases)
    
    assert(len(encoded_bases) == len(encoded_quals))

    img[row_num, read_starts_at : read_starts_at + processed_read_length, 0] = encoded_bases
    img[row_num, read_starts_at : read_starts_at + processed_read_length, 1] = encoded_quals
    img[row_num, read_starts_at : read_starts_at + processed_read_length, 2] = encode_strand_dir(not read.is_reverse, processed_read_length)
    img[row_num, read_starts_at : read_starts_at + processed_read_length, 3] = encode_mapping_quality(read.mapping_quality, processed_read_length)

def stack_read_in_image_tetris_mode(read, img, row_num, col_i, start_pos, end_pos, chrX, BASE_COUNTS_PER_POS):
    sequence, encoded_bases, base_qualities, read_starts_at, insert_indices = get_bases_to_fill(read, chrX, start_pos, end_pos)
    encoded_quals = encode_base_qualities(base_qualities)
    processed_read_length = len(encoded_bases)

    strand_color, mapping_quality = get_strand_color(not read.is_reverse), get_mapping_quality_color(read.mapping_quality)

    insert_base = np.zeros((1, 1, c.NUM_CHANNELS_PER_IMAGE))

    for idx, base in enumerate(encoded_bases):
        insert_base[0, 0, 0] = base
        insert_base[0, 0, 1] = encoded_quals[idx]
        insert_base[0, 0, 2] = strand_color
        insert_base[0, 0, 3] = mapping_quality

        column = read_starts_at + idx
        row = BASE_COUNTS_PER_POS[column]

        if row < c.NUM_READS:
            img[row, column, :] = insert_base
            BASE_COUNTS_PER_POS[column] += 1

def generate_image_tetris_mode(chrX, position, bamfile, ref, ref_dict=None, is_negative_gen=False, sample_reads=False):
    start_time = time()

    row_i = 0
    col_i = 0

    fetch_region_flank =  (c.SEQ_LENGTH - 1) / 2
    fetch_region_start = position - fetch_region_flank
    fetch_region_end = position + fetch_region_flank + 1

    # === INITIALIZATION ===
    img = np.zeros((c.NUM_READS, c.SEQ_LENGTH, c.NUM_CHANNELS_PER_IMAGE), dtype=np.float32)

    # keep a count of bases populated at each position in the image
    BASE_COUNTS_PER_POS = np.zeros((c.SEQ_LENGTH), dtype=int)

    names_of_stacked_reads = {}

    reads = fetch_reads_from_bam(bamfile, chrX, fetch_region_start, fetch_region_end)

    for read in reads:
        if is_usable_read(read) and read.query_name not in names_of_stacked_reads:
            names_of_stacked_reads[read.query_name] = True
            stack_read_in_image_tetris_mode(read, img, row_i, col_i, fetch_region_start, fetch_region_end, chrX, BASE_COUNTS_PER_POS)
            #row_i += 1
            #col_i += 1

        # if row_i >= c.NUM_READS:
        #     break

    return img

def fetch_reads_from_bam(bamfile, chrX, start_pos, end_pos, sample_reads=False):
    # fetch is 0-indexed, inclusive of start_pos, exclusive of end_pos
    try:
        bam_file = bamfile.fetch(chrX, start_pos if start_pos >= 0 else 0, end_pos, multiple_iterators=True)
    except ValueError:
        chrX = 'chr%s' % chrX
        bam_file = bamfile.fetch(chrX, start_pos if start_pos >= 0 else 0, end_pos, multiple_iterators=True)

    return bam_file

def get_reads(bamfile, chrX, start, end):
    """
    Gets all reads in bam that are not duplicate, supplementary,
    unmapped, secondary or repeated.

    Args:
        bamfile - Pysam Alignment File to get reads of.
        chrX - Chromosome/section string to get reads of.
        start - Start place to fetch reads.
        end - End place to fetch reads.
    Returns:
        usable_reads - The list of usable reads.
    """

    usable_reads = []
    reads = fetch_reads_from_bam(bamfile, chrX, start, end) # fetch returns an iterator, which you can go through only once so convert to list()
    names_of_stacked_reads = {}

    for read in reads:
        if is_usable_read(read) and read.query_name not in names_of_stacked_reads:
            names_of_stacked_reads[read.query_name] = True
            usable_reads.append(read)

    return usable_reads

def generate_image(chrX, position, bamfile, ref, reads, ref_dict=None, is_negative_gen=False, sample_reads=False, mutate=None, seed=1):
    row_i = 0
    col_i = 0

    fetch_region_flank =  int((c.SEQ_LENGTH - 1) / 2)
    fetch_region_start = position - fetch_region_flank
    fetch_region_end = position + fetch_region_flank + 1

    num_usable_reads = len(reads)

    if c.SAMPLE_READS and len(reads) > c.NUM_READS: # sample only if there are more than c.NUM_READS reads
        assert not c.VARIABLE_INPUT_SIZE
        reads = sample_reads_fn(reads, c.NUM_READS, seed=seed) # sample reads while preserving order in reads list

    if c.VARIABLE_INPUT_SIZE:
        '''
        Minimum input height is still c.NUM_READS (100). so if there are less than 100 usable reads, the input height will be 100. 
        '''
        if len(reads) > c.MAX_READS:
            reads = sample_reads_fn(reads, c.MAX_READS) # upper limit for reads

        INPUT_HEIGHT = max(c.NUM_READS, len(reads))
        img = np.zeros((INPUT_HEIGHT, c.SEQ_LENGTH, c.NUM_CHANNELS_PER_IMAGE), dtype=np.float32)

    else:
        img = np.zeros((c.NUM_READS, c.SEQ_LENGTH, c.NUM_CHANNELS_PER_IMAGE), dtype=np.float32)
        
    if mutate is not None:
        coverage = len(reads)
 
        # uniform random between 5% and 100%
        #vaf = np.random.uniform(0.05, 1.)

        # beta dist alpha=2, beta=5
        vaf = np.random.beta(2,5)

        num_variant_reads = int(round(vaf*coverage))
        num_normal_reads = coverage - num_variant_reads

        N=0

        #print 'num variants: %d, num normal: %d' % (num_variant_reads, num_normal_reads)

        # MODIFY READS TO INSERT MUTATION
        for read in reads:
            for pos in read.get_aligned_pairs():
                if pos[1] == mutate['ref_pos'] and pos[0] is not None:
                    prev_base = read.query_sequence[pos[0]]
                    if read.query_sequence[pos[0]] == mutate['normal_allele']:
                        if N >= num_normal_reads:
                            read.query_sequence = read.query_sequence[:pos[0]] + mutate['variant_allele'] + read.query_sequence[pos[0]+1:]
                        else:
                            N+=1
                    #print '%s -> %s' % (prev_base, read.query_sequence[pos[0]])
                    break

    if sample_reads:
        num_reads = 0

        for read in reads:
            num_reads+=1

        if num_reads < 100:
            sample_reads = False
        else:
            read_indices = list(range(num_reads))
            sample_read_indices = random.sample(read_indices, c.NUM_READS)
            sample_read_indices.sort()
    
    for idx, read in enumerate(reads):
        if sample_reads: #randomly skip reads
            if idx not in sample_read_indices:
                continue        

        stack_read_in_image(read, img, row_i, col_i, fetch_region_start, fetch_region_end, chrX, ref_dict)
        row_i += 1
        col_i += 1

        if row_i >= c.NUM_READS and not c.VARIABLE_INPUT_SIZE:
            break

    if c.INCLUDE_ALLELE_FREQUENCY:
        print("Calculating coverage ")
        s=time()

        try:
            coverage = bamfile.count_coverage(chrX, fetch_region_start, fetch_region_end)
        except ValueError:
            chrX = 'chr%s' % chrX
            coverage = bamfile.count_coverage(chrX, fetch_region_start, fetch_region_end)

        print(("count coverage af %.10f" % (time() - s)))

        print("Pre-processing coverage... ")
        # [ (#A, #C, #G, #T), (#A, #C, #G, #T), (#A, #C, #G, #T), ] at each position in normal
        coverage_list = [(coverage[0][i], coverage[1][i], coverage[2][i], coverage[3][i]) 
            for i in range(len(coverage[0]))]
        del coverage

        for idx, p in enumerate(coverage_list):
            DEPTH_AT_POSITION = sum(p)
            A_AF, C_AF, G_AF, T_AF = 0., 0., 0., 0.
        
            if DEPTH_AT_POSITION>0:        
                A_AF = float(p[0])/DEPTH_AT_POSITION
                C_AF = float(p[1])/DEPTH_AT_POSITION
                G_AF = float(p[2])/DEPTH_AT_POSITION
                T_AF = float(p[3])/DEPTH_AT_POSITION

            img[:, idx, 4] = A_AF
            img[:, idx, 5] = T_AF
            img[:, idx, 6] = G_AF
            img[:, idx, 7] = C_AF

    return img

def save_array(filename, samples_array, batch_num, sample_name):
    #print "Saving array in: ", sample_name
    np.save(sample_name + "/" + filename + "-b" + str(batch_num), samples_array)

def save_image(tensor):
    normal = Image.fromarray(tensor[:, :, 0:c.NUM_CHANNELS_PER_IMAGE].astype('uint8'), mode='RGB')
    normal.save('normal.png', quality=100)

    tumor = Image.fromarray(tensor[:, :, c.NUM_CHANNELS_PER_IMAGE + 1:c.NUM_CHANNELS].astype('uint8'), mode='RGB')
    tumor.save('tumor.png', quality=100)

    reference = Image.fromarray(tensor[:, :, c.NUM_CHANNELS_PER_IMAGE].astype('uint8'), mode='L')
    reference.save('reference.png', quality=100)

    normal = Image.fromarray(tensor.astype('uint8'), mode='RGB')
    normal.save('image.png', quality=100)

def populate_array(i, image_n, image_t, ref_channel, samples_array):
    middle_pos = c.FLANK

    # if variable_input_size is on, samples_array will take the height of the taller image between normal and tumor (i.e. max(normal_height, tumor_height)). minimum height will be 100bp

    # === Insert the normal == image_n's height may be less than samples_array due to c.VARIABLE_INPUT_SIZE, otherwise it will be c.NUM_READS
    normal_height = image_n.shape[0]

    # no different for TUMOR_NORMAL_ADJACENT
    # before candidate site
    samples_array[i, :normal_height, :middle_pos, 0:c.NUM_CHANNELS_PER_IMAGE] = image_n[:, :middle_pos, :]
    
    # repeat candidate site
    for pos in range(middle_pos, middle_pos + c.CTR_DUP):
        samples_array[i, :normal_height, pos, 0:c.NUM_CHANNELS_PER_IMAGE] = image_n[:, middle_pos, :]

    # after candidate site
    samples_array[i, :normal_height, middle_pos + c.CTR_DUP: c.PER_IMAGE_WIDTH, 0:c.NUM_CHANNELS_PER_IMAGE] = image_n[:, middle_pos + 1:, :]

    # === Insert the reference == ref_channel and samples_array should have same MAX_INPUT_HEIGHT

    if c.TUMOR_NORMAL_ADJACENT:
        # before candidate site
        samples_array[i, :, :middle_pos, c.NUM_CHANNELS_PER_IMAGE] = ref_channel[:, :middle_pos]

        # repeat candidate site
        for pos in range(middle_pos, middle_pos + c.CTR_DUP):
            samples_array[i, :, pos, c.NUM_CHANNELS_PER_IMAGE] = ref_channel[:, middle_pos]
        
        # after candidate site
        samples_array[i, :, middle_pos + c.CTR_DUP: c.PER_IMAGE_WIDTH, c.NUM_CHANNELS_PER_IMAGE] = ref_channel[:, middle_pos + 1:]

        # duplicate the ref image for the tumor part

        # before candidate site
        samples_array[i, :, c.PER_IMAGE_WIDTH: c.PER_IMAGE_WIDTH + middle_pos, c.NUM_CHANNELS_PER_IMAGE] = ref_channel[:, :middle_pos]

        # repeat candidate site
        for pos in range(c.PER_IMAGE_WIDTH + middle_pos, c.PER_IMAGE_WIDTH + middle_pos + c.CTR_DUP):
            samples_array[i, :, pos, c.NUM_CHANNELS_PER_IMAGE] = ref_channel[:, middle_pos]
        
        # after candidate site
        samples_array[i, :, c.PER_IMAGE_WIDTH + middle_pos + c.CTR_DUP: , c.NUM_CHANNELS_PER_IMAGE] = ref_channel[:, middle_pos + 1:]

    else:
        samples_array[i, :, :middle_pos, c.NUM_CHANNELS_PER_IMAGE] = ref_channel[:, :middle_pos]
        for pos in range(middle_pos, middle_pos + c.CTR_DUP):
            samples_array[i, :, pos, c.NUM_CHANNELS_PER_IMAGE] = ref_channel[:, middle_pos]
        
        samples_array[i, :, middle_pos + c.CTR_DUP: , c.NUM_CHANNELS_PER_IMAGE] = ref_channel[:, middle_pos + 1:]

    # === Insert the tumor  == image_t's height may be less than samples_array due to c.VARIABLE_INPUT_SIZE

    tumor_height = image_t.shape[0]

    if c.TUMOR_NORMAL_ADJACENT:
        # before candidate site
        samples_array[i, :tumor_height, c.PER_IMAGE_WIDTH: c.PER_IMAGE_WIDTH + middle_pos, 0:c.NUM_CHANNELS_PER_IMAGE] = image_t[:, :middle_pos, :]
        
        # repeat candidate site
        for pos in range(c.PER_IMAGE_WIDTH + middle_pos, c.PER_IMAGE_WIDTH + middle_pos + c.CTR_DUP):
            samples_array[i, :tumor_height, pos, 0:c.NUM_CHANNELS_PER_IMAGE] = image_t[:, middle_pos, :]
        
        # after candidate site
        samples_array[i, :tumor_height, c.PER_IMAGE_WIDTH + middle_pos + c.CTR_DUP: , 0:c.NUM_CHANNELS_PER_IMAGE] = image_t[:, middle_pos + 1:, :]
    else:
        samples_array[i, :, :middle_pos, c.NUM_CHANNELS_PER_IMAGE + 1:c.NUM_CHANNELS] = image_t[:, :middle_pos, :]
        
        for pos in range(middle_pos, middle_pos + c.CTR_DUP):
            samples_array[i, :, pos, c.NUM_CHANNELS_PER_IMAGE + 1:c.NUM_CHANNELS] = image_t[:, middle_pos, :]
        
        samples_array[i, :, middle_pos + c.CTR_DUP: , c.NUM_CHANNELS_PER_IMAGE + 1:c.NUM_CHANNELS] = image_t[:, middle_pos + 1:, :]

def create_input_tensor_for_position(chromosome, position, bamfile_n, bamfile_t, ref_file, sample_reads=False, mutate=None):
    """
    Returns the input tensor for given position
    The shape of the output is (1, INPUT_SHAPE), as used during prediction by keras.

    """
    # get fetch positions
    fetch_region_flank =  int((c.SEQ_LENGTH - 1) / 2)
    fetch_region_start = position - fetch_region_flank
    fetch_region_end = position + fetch_region_flank + 1
    
    # fetch normal and tumor reads
    normal_reads = get_reads(bamfile_n, chromosome, fetch_region_start, fetch_region_end)
    tumor_reads = get_reads(bamfile_t, chromosome, fetch_region_start, fetch_region_end)

    # how many times to sample c.NUM_READS reads from bams
    n_read_samples = 1 

    # sample reads n times if using c.SAMPLE_READS AND if there are more than 100 reads in either normal or tumor
    if c.SAMPLE_READS and (len(normal_reads) > c.NUM_READS or len(tumor_reads) > c.NUM_READS):
        n_read_samples = c.SAMPLE_READS_COUNT
    
    # if using all reads, set the height accordingly
    if c.VARIABLE_INPUT_SIZE:
        INPUT_HEIGHT = max(len(normal_reads), len(tumor_reads))
    else:
        INPUT_HEIGHT = c.INPUT_SHAPE[0] # default

    X = np.zeros((n_read_samples, INPUT_HEIGHT, c.INPUT_SHAPE[1], c.INPUT_SHAPE[2]), dtype=np.float32) # shape = (n, height, width, channels)

    # get ref channel. no need to do this n times. this must be done only if the batch contains a single site (i.e. c.SAMPLE_READS_COUNT > 1). NOT to be done if batch contains multiple sites
    try:
        ref_channel = get_reference(ref_file,  chromosome, position, NUM_READS=INPUT_HEIGHT)
    
    except KeyError:
        print(("%s %s. location not in reference" % (chromosome, str(position))))
        raise
        
    except ValueError:
        print(("%s %s, failed to retrieve sequence" % (chromosome, str(position))))
        raise

    for sample in range(1, n_read_samples + 1): # samples [1,2...,n]
        # normal
        try:
            if c.TETRIS_MODE:
                image_n = generate_image_tetris_mode(chromosome, position, bamfile_n, ref_file, sample_reads=sample_reads)
            else:
                image_n = generate_image(chromosome, position, bamfile_n, ref_file, normal_reads, sample_reads=sample_reads, seed=sample)

        except ValueError:
            print(("%s %s. location not in normal bam" % (chromosome, str(position))))
            raise

        # tumor
        try:
            if c.TETRIS_MODE:
                image_t = generate_image_tetris_mode(chromosome, position, bamfile_t, ref_file, sample_reads=sample_reads)
            else:
                image_t = generate_image(chromosome, position, bamfile_t, ref_file, tumor_reads, sample_reads=sample_reads, mutate=mutate, seed=sample) # mutate only for tumor sample

        except ValueError:
            print(("%s %s. location not in tumor bam" % (chromosome, str(position))))
            raise
        
        # populate each sample in the input
        populate_array(sample-1, image_n, image_t, ref_channel, X)

    return X

def generate_images_for_positions(sample_name, positions_to_generate, normal_bam_path, tumor_bam_path, training_data_file, batch_num=None):
    """
    positions_to_generate = [ [ 'X', 2999, 1.0 ], [ 'X', 2998, 0.0 ] ... ]
    """
    print(("Sample Name: %s" % sample_name))
    print(("Number of candidate positions %s" % str(len(positions_to_generate))))

    error_positions = 0

    try:
        print(("NORMAL BAM: %s" % normal_bam_path))
        bamfile_n = pysam.AlignmentFile(normal_bam_path, "rb") # normal bamfile
    except IOError:
        print(("IOError: %s" % normal_bam_path))
        return
    except OSError as e:
        print((e, normal_bam_path))
        return

    except OSError as e:
        print((e, normal_bam_path))
        return

    try:
        print(("TUMOR BAM: %s" % tumor_bam_path))
        bamfile_t = pysam.AlignmentFile(tumor_bam_path, "rb") # tumor bamfile
    except IOError:
        print(("IOError: %s" % tumor_bam_path))
        return
    except OSError as e:
        print((e, tumor_bam_path))
        return

    Y = []
    X_arr = []

    num_done, num_errors = 0,0

    for idx, pos in enumerate(positions_to_generate):
        chromosome, position, y = str(pos[0]), int(pos[1]), float(pos[2])

        try:
            X = create_input_tensor_for_position(chromosome, position, bamfile_n, bamfile_t)
            X_arr.append(X[0, :, :, :])
            del X
        
            Y.append(y)

            num_done += 1

        except ValueError:
            num_errors += 1
            continue

    print(("error positions %d " % error_positions))

    print(("done %d, errors %d" % (num_done, num_errors)))

    assert len(X_arr) == len(Y)
    # convert to tensor with shape like (num_samples, 100, 35, 9) for X and (num_samples, 1) for Y
    x_length, y_length = len(X_arr), len(Y)
    X_arr = np.asarray(X_arr) # np.asarray converts a list of ndarrays to a single tensor of shape (len(X_arr) ,100, 35, 9)
    Y = np.asarray(Y)
    Y = Y.reshape((y_length, 1)) # 

    assert X_arr.shape == tuple([x_length] + c.INPUT_SHAPE)
    assert Y.shape == (y_length, 1)
    
    print(("output length: %d" % len(Y)))
    
    if batch_num is None: # generating patient in a single batch
        save_compressed_npy(training_data_file, X_arr, Y)
        print(("Saved %s" % training_data_file))
    else:
        save_compressed_npy(training_data_file + '.' + str(batch_num), X_arr, Y)
        print(("Saved %s batch %s" % (training_data_file, str(batch_num))))

def parse_predictions_file(predictions_path, args, is_smurf=False, is_goldset=False, goldset_trues_path=None):
    """
    Returns an array with Chromosome, Position, 1/0 (mutation probability) for each position in the file.
    The positions are returned 0-indexed. Smurf and goldset positions are 1-indexed
    """

    print(("Parsing %s" % predictions_path))

    positives_in_file, negatives_in_file = 0,0 

    positions_in_file, balanced_positions_to_generate = [], []

    if is_smurf:
        with open(predictions_path) as f:
            for idx, line in enumerate(f):

                if not idx:
                    continue

                line = line.strip()
                split_cols = line.split('\t')

                chromosome, position_1_indexed, prediction = split_cols[0], int(split_cols[1]), split_cols[2]
                output = float(prediction)

                if output: positives_in_file += 1
        
                position_0_indexed = position_1_indexed - 1
                positions_in_file.append( [chromosome, position_0_indexed, output] )
    
    elif is_goldset:
        positions = {}
    
        # get the trues
        with open(goldset_trues_path) as f:
            for line in f:
                line = line.strip()
                split_cols = line.split('\t')
                chromosome, position_1_indexed = split_cols[0], int(split_cols[1])
                position_0_indexed = position_1_indexed - 1
                positions_in_file.append( [chromosome, position_0_indexed, 1.0] )
                pos_key = 'chrom%spos%d' % (chromosome, position_0_indexed)
                positions[pos_key] = True

        # get the remaining negative positions from smurf output
        with open(predictions_path) as f:
            for idx, line in enumerate(f):
                if not idx:
                    continue
                line = line.strip()
                split_cols = line.split('\t')
                chromosome, position_1_indexed = split_cols[1], int(split_cols[2])
                position_0_indexed = position_1_indexed - 1
                pos_key = 'chrom%spos%d' % (chromosome, position_0_indexed)
    
                if pos_key not in positions:
                    positions_in_file.append( [chromosome, position_0_indexed, 0.0] )
    else:
        print("Is it SMURF predictions or GOLDSET data?")

    print(("%d positions in file" % len(positions_in_file)))
    print(("positives in file %s" % positives_in_file))
 
    if not is_goldset:
        # balance positive and negative positions
        for pos in positions_in_file:
            if pos[2]:
                balanced_positions_to_generate.append(pos)
            else:
                if negatives_in_file < positives_in_file:
                    balanced_positions_to_generate.append(pos)
                    negatives_in_file += 1

    else:
        balanced_positions_to_generate = positions_in_file

    print(("%d positives, %d negatives" % (positives_in_file, negatives_in_file)))

    return balanced_positions_to_generate

if __name__ == '__main__':
    if args.generate_all == 'no':

        TIME = time()
        labels = pd.read_csv(args.path_to_labels, delimiter='\t', header=0)

        # Sort the labels file by position and chromosome and then reindex
        labels = labels.sort_values(['X.CHROM', 'POS'], ascending=[True, True]).reset_index(drop=True)

        # Number of labels
        print(("Number of labels: ", len(labels)))

        print("=== Processing BAM files")
        num_samples = (len(labels) / BATCH_SIZE) * BATCH_SIZE  # truncates the last few labels. Gotta fix this too
        labels = labels.iloc[:num_samples]

        batches = []
        for i in range(BATCH_SIZE, num_samples + BATCH_SIZE, BATCH_SIZE):
            batches.append(labels.iloc[i-BATCH_SIZE:i])

        print(("Number of batches: ", len(batches)))

        # process all the batches
        Parallel(n_jobs=8)(delayed(genBatch)(batches[i], i, args) for i in range(0, len(batches)))

        print(("==========" * 50))
        print(("TOTAL TIME ELAPSED: ", time() - TIME))

        print(("# of labels: ", len(labels)))
        print(("# of batches: ", len(batches)))
    
    else:
        encoding_name = c.encoding_name
        if args.environment == 'nscc':
            training_data_folder = os.path.join(c.training_data_folder_on_nscc, encoding_name)
        elif args.environment == 'aquila':
            training_data_folder = os.path.join(c.training_data_folder_on_aquila, encoding_name)
    
        if os.path.exists(training_data_folder):
            print(("Training data already exists: %s" % training_data_folder))
        else:
            print(("Creating folder for training data: %s" % training_data_folder))
            os.makedirs(training_data_folder)

        training_data_folder = os.path.join(training_data_folder, c.all_data_folder_name)
    
        if os.path.exists(training_data_folder):
            print(("All data folder already exists: %s" % training_data_folder))
        else:
            print(("Creating folder for training data: %s" % training_data_folder))
            os.makedirs(training_data_folder)

        patient_files = [] # list of (positions, normal_bam, tumor_bam) paths for each patient
       
        if args.liver_data:
            patients = [ patient.replace('.csv','') for patient in os.listdir(c.liver_patients_smurf_predictions) ]
            patients.sort()
            print((len(patients)))
 
            patients = np.array_split(patients, int(args.num_nodes))[int(args.node_no)]

            for patient in patients:
                patient_files.append( (patient, parse_predictions_file(os.path.join(c.liver_patients_smurf_predictions, patient + '.csv'), args, is_smurf=True, is_goldset=False), c.liver_patients_normal_bam_file_path % (patient), c.liver_patients_tumor_bam_file_path % (patient) ) )

        elif args.goldset_data:
            goldset_smurf_predictions_root = '/home/users/astar/gis/krishnak/scratch/SMURF_GOLDSET_PREDICTIONS'

            goldset_samples = [ x for x in c.goldset_files_on_nscc ]
            goldset_samples.sort()
            print((len(goldset_samples)))

            goldset_samples = np.array_split(goldset_samples, int(args.num_nodes))[int(args.node_no)]
 
            for goldset_sample in goldset_samples:
                smurf_predictions_file = os.path.join(goldset_smurf_predictions_root, goldset_sample[0])
                patient_files.append( ( goldset_sample[0], parse_predictions_file(smurf_predictions_file, args, is_smurf=False, is_goldset=True, goldset_trues_path=goldset_sample[1]), str(goldset_sample[2]), str(goldset_sample[3]) ) )

        elif args.goldset_finetune:
            goldset_samples = [ x for x in c.goldset_files_on_nscc ]
            goldset_samples.sort()
        
            goldset_samples = np.array_split(goldset_samples, int(args.num_nodes))[int(args.node_no)]
            for goldset_sample in goldset_samples:
                smurf_predictions_file = goldset_sample[2]
                patient_files.append( ( goldset_sample[0], parse_predictions_file(smurf_predictions_file, args, is_smurf=False, is_goldset=True, goldset_trues_path=goldset_sample[1]), str(goldset_sample[3]), str(goldset_sample[4]) ) )

        elif args.crc_data:
            patients = [ patient.replace('.csv','') for patient in os.listdir(c.crc_patients_smurf_predictions) ]
            patients.sort()
            print((len(patients)))

            patients = np.array_split(patients, int(args.num_nodes))[int(args.node_no)]

            for patient in patients:
                patient_files.append( (patient, parse_predictions_file(os.path.join(c.crc_patients_smurf_predictions, patient + '.csv'), args, is_smurf=True, is_goldset=False), c.crc_patients_normal_bam_file_path % patient, c.crc_patients_tumor_bam_file_path % patient ) )

        elif args.gastric_data:
            patients = [ patient.replace('.csv','') for patient in os.listdir(c.gastric_patients_smurf_predictions) ]
            patients.sort()
            print((len(patients)))

            patients = np.array_split(patients, int(args.num_nodes))[int(args.node_no)]

            for patient in patients:
                patient_files.append( (patient, parse_predictions_file(os.path.join(c.gastric_patients_smurf_predictions, patient + '.csv'), args, is_smurf=True, is_goldset=False), c.gastric_patients_normal_bam_file_path % patient, c.gastric_patients_tumor_bam_file_path % patient ) )

        elif args.lung_data:
            patients = [ patient.replace('.csv','') for patient in os.listdir(c.lung_patients_smurf_predictions) ]
            patients.sort()

            patients = np.array_split(patients, int(args.num_nodes))[int(args.node_no)]
    
            for patient in patients:
                patient_files.append( (patient, parse_predictions_file(os.path.join(c.lung_patients_smurf_predictions, patient + '.csv'), args, is_smurf=True, is_goldset=False), c.lung_patients_normal_bam_file_path % (patient), c.lung_patients_tumor_bam_file_path % (patient) ) )

        elif args.sarcoma_data:
            patients = [ patient.replace('.csv','') for patient in os.listdir(c.sarcoma_patients_smurf_predictions) ]
            patients.sort()

            patients = np.array_split(patients, int(args.num_nodes))[int(args.node_no)]

            for patient in patients:
                patient_files.append( (patient, parse_predictions_file(os.path.join(c.sarcoma_patients_smurf_predictions, patient + '.csv'), args, is_smurf=True, is_goldset=False), c.sarcoma_patients_normal_bam_file_path % (patient), c.sarcoma_patients_tumor_bam_file_path % (patient) ) )

        elif args.thyroid_data:
            patients = [ patient.replace('.csv','') for patient in os.listdir(c.thyroid_patients_smurf_predictions) ]
            patients.sort()

            patients = np.array_split(patients, int(args.num_nodes))[int(args.node_no)]

            for patient in patients:
                patient_files.append( (patient, parse_predictions_file(os.path.join(c.thyroid_patients_smurf_predictions, patient + '.csv'), args, is_smurf=True, is_goldset=False), c.thyroid_patients_normal_bam_file_path % (patient), c.thyroid_patients_tumor_bam_file_path % (patient) ) )

        elif args.lymphoma_data:
            patients = [ patient.replace('.csv','') for patient in os.listdir(c.lymphoma_patients_smurf_predictions) ]
            patients.sort()

            patients = np.array_split(patients, int(args.num_nodes))[int(args.node_no)]

            for patient in patients:
                patient_files.append( (patient, parse_predictions_file(os.path.join(c.lymphoma_patients_smurf_predictions, patient + '.csv'), args, is_smurf=True, is_goldset=False), c.lymphoma_patients_normal_bam_file_path % patient, c.lymphoma_patients_tumor_bam_file_path % patient ) )

        if not len(patient_files):
            pass
        else:
            for patient in patient_files:
                split_positions = np.array_split(patient[1], (len(patient[1])/200000) + 1) # can't fit more than 200k positions in memory, so split the list

                training_data_file = os.path.join( training_data_folder, patient[0])
            
                """
                for idx, batch in enumerate(split_positions):
                    print '%s.%d.npz' % (patient[0], idx)

                    if not os.path.exists(os.path.join(training_data_folder, '%s.%d.npz' % (patient[0], idx) )):
                        print '%s.%d.npz' % (patient[0], idx)
                        continue

                    X,Y = load_compressed_npy(os.path.join(training_data_folder, '%s.%d.npz' % (patient[0], idx) ))
            
                    if Y.shape[0] != len(batch):
                        print os.path.join(training_data_folder, '%s.%d.npz' % (patient[0], idx) )
                        print "%d in npz, %d in batch" % (Y.shape[0], len(batch))

                    del X                    
                """

                Parallel(n_jobs=int(args.num_processes))(delayed(generate_images_for_positions)(patient[0], batch, patient[2], patient[3], training_data_file, idx) for idx, batch in enumerate(split_positions) if not os.path.exists(os.path.join(training_data_folder, '%s.%d.npz' % (patient[0], idx) )) )
 

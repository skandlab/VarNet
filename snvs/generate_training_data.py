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
import sys
sys.path.append('/home/project/13002420/smudl/')

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

from utils import sample_reads_fn

# === FUNCTIONS FOR RUNNING PROGRAM IN COMMAND LINE ===

def parse_args():
    parser = argparse.ArgumentParser(description="Image and NDarray Generator")
    parser.add_argument('--path_to_bam_n', default='')
    parser.add_argument('--path_to_bam_t', default='')
    parser.add_argument('--path_to_labels', default='')
    parser.add_argument('--environment', default='nscc')
    parser.add_argument('--generate_all', default='yes')
    parser.add_argument('--crc_data', default=False)
    parser.add_argument('--gastric_data', default=False)
    parser.add_argument('--liver_data', default=False)
    parser.add_argument('--lung_data', default=False)
    parser.add_argument('--sarcoma_data', default=False)
    parser.add_argument('--thyroid_data', default=False)
    parser.add_argument('--lymphoma_data', default=False)
    parser.add_argument('--goldset_data', default=False)
    parser.add_argument('--ffpe', action='store_true')
    parser.add_argument('--num_nodes')
    parser.add_argument('--node_no')
    parser.add_argument('--num_processes', default=1)
    parser.add_argument('--ref_file')
    parser.add_argument('--mutect2_calls_on_ffpe', action='store_true')
    parser.add_argument('--strelka2_calls_on_ffpe', action='store_true')
    parser.add_argument('--ffpe_wgs_wes_training', action='store_true') 
    parser.add_argument('--tcga_wxs', action='store_true') 
    parser.add_argument('--tumor_only', action='store_true')
    parser.add_argument('--compute_stats', action='store_true')
    parser.add_argument('--parse_ffpe_calls', action='store_true')

    parser.add_argument('--tcga_wxs_convnet', action='store_true') 

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

# === PROGRAM HELPERS === #

def get_ref_base(start_pos, chrX, ref_file, end_pos=None):
    """
    Positions are 0-indexed in pysam and the fetch function returns bases for the half-open interval (includes start position, excludes end position)
    http://pysam.readthedocs.io/en/latest/api.html#pysam.FastaFile.fetch
    """
    if start_pos < 0:
        start_pos = 0
        
    if end_pos is None:
        end_pos = start_pos + 1

    try:
        return ref_file.fetch(chrX, start_pos, end_pos).upper() # base at pos in reference
    except KeyError:
        if chrX == 'chrM' or chrX == 'M':
            chrX = 'MT'
        elif 'chr' in chrX:
            chrX = chrX.replace('chr','')
        else:
            chrX = 'chr%s' % chrX

        return ref_file.fetch(chrX, start_pos, end_pos).upper() # base at pos in reference

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

def get_reference(ref_file, chrX, pos, flank, return_sequence=False, ref_dict=None, NUM_READS=None):
    start_pos = pos - flank # c.FLANK
    end_pos = pos + flank + 1 # c.FLANK + 1
    
    if start_pos < 0: start_pos = 0

    if ref_dict is None:
        try:
            ref_sequence = ref_file.fetch(chrX, start_pos, end_pos).upper() # corresponding reference sequence

        except KeyError:
            if chrX == 'chrM' or chrX == 'M':
                chrX = 'MT'
            elif 'chr' in chrX:
                chrX = chrX.replace('chr','')
            else:
                chrX = 'chr%s' % chrX

            ref_sequence = ref_file.fetch(chrX, start_pos, end_pos).upper() # corresponding reference sequence        
  
        except Exception as e:
            print(('Err in get_reference:', e))
            raise
 
    else:
        ref_sequence = ref_dict[chrX][start_pos:end_pos]
            
    ref_nucs = list(ref_sequence) # nucleotides in reference

    if return_sequence:
        return ref_nucs

    if c.VARIABLE_INPUT_SIZE:
        if NUM_READS:
            ref_channel = np.zeros((NUM_READS, c.SEQ_LENGTH), dtype=np.float32)
        else:
            raise Exception("Variable input size but input height not provided")
    else:
        ref_channel = np.zeros((c.NUM_READS, c.SEQ_LENGTH), dtype=np.float32)

    # expected len of ref sequence is (2*flank+1). if len is not equal to this, then either start_pos was set to 0 or the sequence was truncated at the end of the chromosome (automatically by pysam). 
    if len(ref_nucs) == (2*flank+1) or start_pos == 0:
        # expected length or start_pos was set to 0 (fill it up in reverse)
        for i in range(len(ref_nucs)):
            ref_channel[:, len(ref_nucs)-1-i] = get_base_val(ref_nucs[len(ref_nucs)-1-i])
    else:
        # sequence truncated due to end of chromsome
        # fill it up from the first element
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

def encode_strand_dir(on_positive_strand, length, is_read1):
    if c.ENCODE_READ_ORIENTATION:
        if is_read1:
            return np.ones((length,)) * get_strand_color(on_positive_strand) + 50
        else:
            return np.ones((length,)) * get_strand_color(on_positive_strand) - 50
    else:
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

def record_read_orientation(read, positions_stats, ref_pos, base):
    # process only paired reads
    if not read.is_paired:
        return

    if 'F1R2' not in positions_stats[ref_pos][base]: # initialize
        positions_stats[ref_pos][base]['F1R2'] = 0
        positions_stats[ref_pos][base]['F2R1'] = 0

    if read.is_read1:
        # READ 1
        if (not read.is_reverse): # forward
            positions_stats[ref_pos][base]['F1R2'] += 1 # read 1 from forward strand
        else:
            positions_stats[ref_pos][base]['F2R1'] += 1 # read 1 from reverse strand
    else:
        # READ 2
        if (not read.is_reverse): 
            positions_stats[ref_pos][base]['F2R1'] += 1 # read 2 from forward strand
        else:
            positions_stats[ref_pos][base]['F1R2'] += 1 # read 2 from reverse strand

def record_distance_from_read_end(read, positions_stats, ref_pos, base):
    """
    calculates the minimum distance of ref_pos to either end of the read
    Deaminations have been shown to be enriched at the ends of molecules due to an increased sensitivity to deaminate of overhanging ends
    Sources: 
    1) Briggs,A.W. et al. (2007) Patterns of damage in genomic DNA sequences from a Neandertal. Proc. Natl Acad. Sci. U.S.A., 104, 14616–14621
    2) Lindahl,T. and Nyberg,B. (1972) Rate of depurination of native deoxyribonucleic acid. Biochemistry, 11, 3610–3618
    """
    if 'read_end_distances' not in positions_stats[ref_pos][base]:
        positions_stats[ref_pos][base]['read_end_distances'] = []

    read_end_distance = min(abs(ref_pos - read.reference_start), abs(ref_pos - read.reference_end))
    read_length = read.reference_end - read.reference_start
    read_end_distance = read_end_distance/read_length # divide by read length
    positions_stats[ref_pos][base]['read_end_distances'].append(read_end_distance)

def record_fragment_length(read, positions_stats, ref_pos, base):
    """
    Calculates the fragment length of the read pair. Shorter fragments maybe associated with DNA damage
    Fragment length can be quite large for outliers so use median fragment length divided by 500.
    """
    # process only properly paired reads, i.e. the mates are mapped and on the same chromosome within a small range
    # TLEN is accurate only for properly paired reads
    if not read.is_proper_pair:
        return None

    if 'fragment_lengths' not in positions_stats[ref_pos][base]:
        positions_stats[ref_pos][base]['fragment_lengths'] = []
        
    positions_stats[ref_pos][base]['fragment_lengths'].append(abs(read.template_length)) # use absolute value of TLEN as it can be negative when read2 is aligned before read1 in the reference genome
    return read.template_length

def get_bases_to_fill_tumor_only(read, chromosome, reference_start, reference_end, positions_stats):
    """
    positions_stats = { 'pos1': {'attr_1': val, 'attr_2': val} ..., 'pos2': {} }
    """
    seq = ''
    encoded_seq = []
    quals = []

    all_pairs = get_positions_to_fill(read) # list of read/query and reference positions, soft clip bases removed
    read_starts_at = 0

    insert_indices = []

    # query_qualities is missing in reads sometimes. Fill it up with 45 (min base qual)
    if read.query_qualities is None:
        read.query_qualities = [45 for u in range(len(read.query_sequence))]

    last_ref_pos = None # keep track of the last valid ref_pos position within input window parsed in the read
    INSERT_POSITIONS_IN_READ = {} # record positions where inserts have been found in this read

    for i, pair in enumerate(all_pairs):
        read_pos = pair[0]
        ref_pos = pair[1]

        if ref_pos is None: # Ref pos is None for insertions            
            if last_ref_pos is not None and last_ref_pos not in INSERT_POSITIONS_IN_READ:
                # Ensure the insertion isn't at the beginning of the read i.e. last_ref_pos is not None          
                # Keep a count of insertions at the previous valid ref_pos position i.e. last_ref_pos
                # Even if the insertion has multiple bases e.g. 'ATGCCC', this will record only one insertion at last_ref_pos using INSERT_POSITIONS_IN_READ
                if 'INS' not in positions_stats[last_ref_pos]:
                    positions_stats[last_ref_pos]['INS'] = {}
                    positions_stats[last_ref_pos]['INS']['count'] = 1
                    positions_stats[last_ref_pos]['INS']['base_qualities'] = [read.query_qualities[read_pos]/100] # divide BQ phred score by 100
                    positions_stats[last_ref_pos]['INS']['mapping_qualities'] = [read.mapping_quality/100] # divide MQ phred score by 100
                    # initialize forward/reverse counts for insertion at last_ref_pos
                    positions_stats[last_ref_pos]['INS']['forward_reads'], positions_stats[last_ref_pos]['INS']['reverse_reads'] = 0,0
                    
                else:
                    positions_stats[last_ref_pos]['INS']['count'] += 1
                    positions_stats[last_ref_pos]['INS']['base_qualities'].append(read.query_qualities[read_pos]/100) # divide BQ phred score by 100
                    positions_stats[last_ref_pos]['INS']['mapping_qualities'].append(read.mapping_quality/100) # divide MQ phred score by 100

                # update forward/reverse counts for insertion at last_ref_pos
                if not read.is_reverse:
                    positions_stats[last_ref_pos]['INS']['forward_reads'] += 1
                else:
                    positions_stats[last_ref_pos]['INS']['reverse_reads'] += 1

                # read orientation
                record_read_orientation(read, positions_stats, last_ref_pos, 'INS')
                record_distance_from_read_end(read, positions_stats, last_ref_pos, 'INS')
                record_fragment_length(read, positions_stats, last_ref_pos, 'INS')

                INSERT_POSITIONS_IN_READ[last_ref_pos] = True

        elif ref_pos >= reference_start and ref_pos < reference_end:
            last_ref_pos = ref_pos
            
            if ref_pos not in positions_stats:
                positions_stats[ref_pos] = {}
            
            if not len(encoded_seq): # if this is the first base we are adding to the encoded read
                read_starts_at = ref_pos - reference_start

            if read_pos is None:
                # This means that there is a deletion
                base = 'DEL'
            else:
                # match
                base = read.query_sequence[read_pos]

            if base not in positions_stats[ref_pos]:
                positions_stats[ref_pos][base] = {}
                positions_stats[ref_pos][base]['count'] = 1

                # initialize forward/reverse counts for base at ref_pos
                positions_stats[ref_pos][base]['forward_reads'], positions_stats[ref_pos][base]['reverse_reads'] = 0,0

                if base == 'DEL':
                    # deletions don't have BQ so use average of neighboring bases for DEL
                    positions_stats[ref_pos][base]['base_qualities'] = [get_average_neighbouring_base_quality(read, ref_pos)/100] # divide BQ phred score by 100
                else:
                    positions_stats[ref_pos][base]['base_qualities'] = [read.query_qualities[read_pos]/100] # divide BQ phred score by 100

                positions_stats[ref_pos][base]['mapping_qualities'] = [read.mapping_quality/100] # divide MQ phred score by 100
            else:
                positions_stats[ref_pos][base]['count'] += 1
                    
                if base == 'DEL':
                    # deletions don't have BQ so use average of neighboring bases for DEL
                    positions_stats[ref_pos][base]['base_qualities'].append(get_average_neighbouring_base_quality(read, ref_pos)/100) # divide BQ phred score by 100
                else:
                    positions_stats[ref_pos][base]['base_qualities'].append(read.query_qualities[read_pos]/100) # divide BQ phred score by 100
                    
                positions_stats[ref_pos][base]['mapping_qualities'].append(read.mapping_quality/100) # divide MQ phred score by 100

            # update forward/reverse counts for this base at ref_pos
            if not read.is_reverse:
                positions_stats[ref_pos][base]['forward_reads'] += 1
            else:
                positions_stats[ref_pos][base]['reverse_reads'] += 1

            record_read_orientation(read, positions_stats, ref_pos, base)
            record_distance_from_read_end(read, positions_stats, ref_pos, base)
            record_fragment_length(read, positions_stats, ref_pos, base)

def stack_read_in_image(read, img, row_num, col_i, start_pos, end_pos, chrX, ref_dict=None):
    sequence, encoded_bases, base_qualities, read_starts_at, insert_indices = get_bases_to_fill(read, chrX, start_pos, end_pos)
    encoded_quals = encode_base_qualities(base_qualities)

    processed_read_length = len(encoded_bases)
    
    assert(len(encoded_bases) == len(encoded_quals))

    img[row_num, read_starts_at : read_starts_at + processed_read_length, 0] = encoded_bases
    img[row_num, read_starts_at : read_starts_at + processed_read_length, 1] = encoded_quals
    img[row_num, read_starts_at : read_starts_at + processed_read_length, 2] = encode_strand_dir(not read.is_reverse, processed_read_length, read.is_read1)
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

def generate_image_tetris_mode(chrX, position, bamfile, ref, ref_dict=None, is_negative_gen=False):
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

def fetch_reads_from_bam(bamfile, chrX, start_pos, end_pos):
    # fetch is 0-indexed, inclusive of start_pos, exclusive of end_pos
    try:
        bam_file = bamfile.fetch(chrX, start_pos if start_pos >= 0 else 0, end_pos, multiple_iterators=True)
    except ValueError:
        chrX = 'chr%s' % chrX
        bam_file = bamfile.fetch(chrX, start_pos if start_pos >= 0 else 0, end_pos, multiple_iterators=True)

    return bam_file

def get_reads(bamfile, chrX, start, end):
    """
    *** legacy function for convnet tumor-normal. ***

    read.query_name is template name and is shared by read mates in a pair.

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

def get_reads_tumor_only(bamfile, chrX, start, end):
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

    for read in reads:
        if is_usable_read(read):
            usable_reads.append(read)

    return usable_reads

def get_gc_bias(chrom, start_pos, end_pos, ref_file):
    """
    returns percentage of GC content in 300bp window (150 on each side) of the input window
    """
    gc_window = 150 # 150bp on each side of the input, total is 300bp + INPUT_WIDTH
    ref_window = get_ref_base(start_pos-gc_window, chrom, ref_file, end_pos=end_pos+gc_window)
    from collections import Counter
    counts = Counter(ref_window) # {'A': 40, 'T': 32, 'G': 4, 'C': 50, 'N': 10}    
    gc_content = round((counts['G']+counts['C'])/len(ref_window), 3) # round the percentage to three decimal places
    return gc_content

def phred_score_to_probability(score):
    return 10**(-score/10)

def generate_image_tumor_only(chrX, position, bamfile, ref_file, reads, ref_sequence, args=None, y=None):
    """
    All values in the returned input are in the range [-1,1] so input normalization not needed
    ref_sequence is a list of ref bases i.e. ['A', 'T', 'G', 'N', 'N', 'N' ...]
    """        
    fetch_region_start = position - c.TUMOR_ONLY_FLANK
    fetch_region_end = position + c.TUMOR_ONLY_FLANK + 1

    gc_bias = get_gc_bias(chrX, fetch_region_start, fetch_region_end, ref_file)

    HEIGHT, WIDTH = c.TUMOR_ONLY_SHAPE

    img = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    positions_stats = {} #  { pos1: {'attr_1': val, 'attr_2': val} ..., pos2: {} } 

    for idx, read in enumerate(reads):
        get_bases_to_fill_tumor_only(read, chrX, fetch_region_start, fetch_region_end, positions_stats)

    BASES = ['A', 'T', 'G', 'C', 'DEL', 'INS'] 

    ## pad the ref sequence in case the start_pos < 0 or the end_pos is after end of chromosome (ref_sequence will be trunctated in both cases)
    if len(ref_sequence) == WIDTH:
        pass # no need to pad
    else:
        diff = WIDTH - len(ref_sequence)
        assert diff > 0, 'ref_sequence is larger than window'
        
        if fetch_region_start<0:
            # position too close to beginning of chromosome so the ref_sequence was truncated on the left
            for _ in range(diff):
                ref_sequence.insert(0,'N') # prepend 'N' to the beginning of the list
        else:
            # fetch_region_end must be after end of chromosome so ref_sequence was truncated on the right
            for _ in range(diff):
                ref_sequence.append('N') # append 'N' to the end of the list

        assert len(ref_sequence) == WIDTH # check that ref_sequence equals width of window
        
    # divide the base values by 245 to normalize them to [0,1]
    encoded_ref_seq = [get_base_val(nuc)/245 for nuc in ref_sequence]

    for col, pos in enumerate(range(fetch_region_start, fetch_region_end)):
        if args is not None and args.compute_stats:
            if pos != position:
                continue # skip non-candidate positions when computing stats

        curr_row = 0
        
        if pos in positions_stats:
            # encode normalized counts of alleles (including DEL and INS)
            total_allele_count = 0
            for B in BASES:
                if B in positions_stats[pos]:
                    img[curr_row, col] = positions_stats[pos][B]['count']
                    curr_row += 1

                    if B != 'INS':
                        # don't count reads with INS as this will double count the reads
                        # the reads supporting the INS would have been counted already for the base at the last_ref_pos
                        total_allele_count += positions_stats[pos][B]['count']
                else:
                    # base not observed in pos so set count to 0
                    img[curr_row, col] = 0             
                    curr_row += 1

            if 'N' in positions_stats[pos]:
                # indeterminate base 'N' at position, count it towards read depth though.
                total_allele_count += positions_stats[pos]['N']['count']

            # normalize allele counts
            if total_allele_count>0:
                # don't divide by 0/0
                img[:curr_row, col] = img[:curr_row, col]/total_allele_count
            else:
                if pos == position and y == 1.0:
                    # zero coverage at candidate site for a mutation so don't use for training (y is provided only during training)
                    print('zero coverage'); import pdb; pdb.set_trace()
                    return None

            # BQ stats for all bases (including DEL and INS)
            # BQ Phred scores are divided by 100 in get_bases_to_fill_tumor_only() to be in range [0,1] roughly
            # Phred scores are usually <60
            for B in BASES:
                if B in positions_stats[pos]:
                    img[curr_row, col] = np.mean(positions_stats[pos][B]['base_qualities'])
                    curr_row += 1
                    img[curr_row, col] = np.median(positions_stats[pos][B]['base_qualities']) 
                    curr_row += 1
                    img[curr_row, col] = np.min(positions_stats[pos][B]['base_qualities']) 
                    curr_row += 1
                    img[curr_row, col] = np.max(positions_stats[pos][B]['base_qualities'])
                    curr_row += 1
                    img[curr_row, col] = np.quantile((positions_stats[pos][B]['base_qualities']), 0.05) # 5th percentile
                    curr_row += 1
                    img[curr_row, col] = np.quantile((positions_stats[pos][B]['base_qualities']), 0.95) # 95th percentile
                    curr_row += 1
                else:
                    # base not observed in pos so set base quality stats to 0
                    img[curr_row:curr_row+6, col] = -1 # 6 stats above
                    curr_row += 6 # 6 stats

            # MQ stats for reads at this pos with for all bases (including DEL and INS)
            # MQ Phred scores are divided by 100 in get_bases_to_fill_tumor_only() to be in range [0,1] roughly
            # Phred scores are usually <60
            for B in BASES:
                if B in positions_stats[pos]:
                    img[curr_row, col] = np.mean(positions_stats[pos][B]['mapping_qualities'])
                    curr_row += 1
                    img[curr_row, col] = np.median(positions_stats[pos][B]['mapping_qualities']) 
                    curr_row += 1
                    img[curr_row, col] = np.min(positions_stats[pos][B]['mapping_qualities']) 
                    curr_row += 1
                    img[curr_row, col] = np.max(positions_stats[pos][B]['mapping_qualities']) 
                    curr_row += 1
                    img[curr_row, col] = np.quantile((positions_stats[pos][B]['mapping_qualities']), 0.05) # 5th percentile
                    curr_row += 1
                    img[curr_row, col] = np.quantile((positions_stats[pos][B]['mapping_qualities']), 0.95) # 95th percentile
                    curr_row += 1
                else:
                    # base not observed in pos so set base quality stats to 0
                    img[curr_row:curr_row+6, col] = -1 # 6 stats above
                    curr_row += 6 # 6 stats

            # Strand bias for reads at this pos with base (include DEL and INS)
            for B in BASES:
                if B in positions_stats[pos]:
                    img[curr_row, col] = positions_stats[pos][B]['forward_reads']/(positions_stats[pos][B]['forward_reads']+positions_stats[pos][B]['reverse_reads'])
                    curr_row += 1
                else:
                    # base not observed in pos so set strand bias to -1
                    img[curr_row, col] = -1
                    curr_row += 1 # 1 stat
            
            # read orientation SOB in range [-1,1] for each base (include DEL and INS)
            # using abs(SOB) will be in range [0,1] to avoid distributional imbalance about 0
            for B in BASES:
                if B in positions_stats[pos] and 'F1R2' in positions_stats[pos][B] and 'F2R1' in positions_stats[pos][B]: # possible that there were no paired reads found for this allele
                    img[curr_row, col] = abs((positions_stats[pos][B]['F1R2']-positions_stats[pos][B]['F2R1'])/(positions_stats[pos][B]['F1R2']+positions_stats[pos][B]['F2R1']))
                    curr_row += 1
                else:
                    # base not observed in pos so set SOB to -1
                    img[curr_row, col] = -1
                    curr_row += 1 # 1 stat

            # encode GC content of reference sequence in the window surrounding input. use pysamstats to test
            img[curr_row, col] = gc_bias
            curr_row+=1

            # encode properties of the ALT allele (SNV or indel) at each position
            # identify the ALT allele based on whether it is snvs/filter.py or indels/filter.py
            # encode median fragment length supporting ALT allele (divided by 500), median distance from read end of supporting reads
            # SOB score of ALT allele
            ALT = None
            MAX_ALT_COUNT = -1

            # find alt allele that has highest AF (SNV or INDEL) in each column
            # if no ALT allele found for non-candidate position, -1 is encoded
            REF = ref_sequence[col]
            for B in BASES:
                if B != REF and B in positions_stats[pos] and positions_stats[pos][B]['count']>MAX_ALT_COUNT:
                    MAX_ALT_COUNT = positions_stats[pos][B]['count']
                    ALT=B
                
            if ALT in positions_stats[pos]:
                # ENCODE VAF OF ALT (SNV OR indel)
                img[curr_row, col] = positions_stats[pos][ALT]['count']/total_allele_count
                curr_row += 1
                # BQ stats of ALT (SNV or indel)
                img[curr_row, col] = np.mean(positions_stats[pos][ALT]['base_qualities'])
                curr_row += 1
                img[curr_row, col] = np.median(positions_stats[pos][ALT]['base_qualities'])
                curr_row += 1
                img[curr_row, col] = np.min(positions_stats[pos][ALT]['base_qualities']) 
                curr_row += 1
                img[curr_row, col] = np.max(positions_stats[pos][ALT]['base_qualities'])
                curr_row += 1
                img[curr_row, col] = np.quantile((positions_stats[pos][ALT]['base_qualities']), 0.05) # 5th percentile
                curr_row += 1
                img[curr_row, col] = np.quantile((positions_stats[pos][ALT]['base_qualities']), 0.95) # 95th percentile
                curr_row += 1
                # MQ stats of ALT (SNV or indel)
                img[curr_row, col] = np.mean(positions_stats[pos][ALT]['mapping_qualities'])
                curr_row += 1
                img[curr_row, col] = np.median(positions_stats[pos][ALT]['mapping_qualities'])
                curr_row += 1
                img[curr_row, col] = np.min(positions_stats[pos][ALT]['mapping_qualities']) 
                curr_row += 1
                img[curr_row, col] = np.max(positions_stats[pos][ALT]['mapping_qualities']) 
                curr_row += 1
                img[curr_row, col] = np.quantile((positions_stats[pos][ALT]['mapping_qualities']), 0.05) # 5th percentile
                curr_row += 1
                img[curr_row, col] = np.quantile((positions_stats[pos][ALT]['mapping_qualities']), 0.95) # 95th percentile
                curr_row += 1
                # strand bias of ALT (SNV or indel)
                img[curr_row, col] = positions_stats[pos][ALT]['forward_reads']/(positions_stats[pos][ALT]['forward_reads']+positions_stats[pos][ALT]['reverse_reads'])
                curr_row += 1
                # SOB score in range [-1,1] of ALT (SNV or indel)
                # abs(SOB) will be in range [0,1] to avoid distributional imbalance about 0
                if 'F1R2' in positions_stats[pos][ALT] and 'F2R1' in positions_stats[pos][ALT]:
                    img[curr_row, col] = abs((positions_stats[pos][ALT]['F1R2']-positions_stats[pos][ALT]['F2R1'])/(positions_stats[pos][ALT]['F1R2']+positions_stats[pos][ALT]['F2R1']))
                else:
                    # possible that ALT has no properly paired reads to compute read orientation bias
                    img[curr_row, col] = -1
                curr_row += 1
                # median read end distance of ALT reads (SNV or indel)
                img[curr_row, col] = np.median(positions_stats[pos][ALT]['read_end_distances'])
                curr_row += 1

                # record stats
                if args is not None and args.compute_stats:
                    if pos == position: # candidate position
                        stats = {}
                        if 'fragment_lengths' in positions_stats[pos][ALT]:
                            stats['fragment_lengths'] = np.median(positions_stats[pos][ALT]['fragment_lengths'])
                        else:
                            # no proper read pairs found for this ALT
                            stats['fragment_lengths'] = None
                        stats['VAF'] =  positions_stats[pos][ALT]['count']/total_allele_count
                        stats['DP'] =  total_allele_count
                        stats['strand_bias'] = positions_stats[pos][ALT]['forward_reads']/(positions_stats[pos][ALT]['forward_reads']+positions_stats[pos][ALT]['reverse_reads'])
                        stats['gc_bias'] = gc_bias
                        if 'F1R2' in positions_stats[pos][ALT] and 'F2R1' in positions_stats[pos][ALT]:
                            stats['SOB_scores'] = abs((positions_stats[pos][ALT]['F1R2']-positions_stats[pos][ALT]['F2R1'])/(positions_stats[pos][ALT]['F1R2']+positions_stats[pos][ALT]['F2R1']))
                        else:
                            stats['SOB_scores'] = None
                        stats['read_end_distances'] = np.median(positions_stats[pos][ALT]['read_end_distances'])
                        return stats
                        
            else:
                # no ALT found in col
                NUM_ALT_STATS = 16 # stats above
                img[curr_row:curr_row+NUM_ALT_STATS, col] = -1
                curr_row += NUM_ALT_STATS

                if pos == position and y == 1.0:
                    # no ALT allele found at candidate site for a mutation in training data, so return None to skip site
                    # print('no ALT coverage');
                    # import pdb; pdb.set_trace()
                    return None

            assert curr_row == (HEIGHT-1) # last row reserved for REF sequence

        else:
            # set column to -1 if pos does not have any overlapping reads
            img[:, col] = -1

    if args is not None and args.compute_stats:
        # in case no stats returned for ALT
        return None

    # encode REF sequence once for entire input in the last row
    img[-1] = encoded_ref_seq

    ## duplicate candidate site?

    # round img to 4 decimals
    img = np.round(img, 4)
    
    # import pdb; pdb.set_trace()

    if args is not None:
        # training data
        if np.isnan(img).any():
            from numpy import unravel_index
            print('nan found in index:', unravel_index(np.argmin(img), img.shape))
            import pdb; pdb.set_trace()

        if img.max() > 1 or img.min() < -1:
            from numpy import unravel_index
            print('max found in index:', unravel_index(np.argmax(img), img.shape), img.max())
            print('min found in index:', unravel_index(np.argmin(img), img.shape), img.min())
            import pdb; pdb.set_trace()

    # convert (rows,cols) to (cols,rows) for transformer model to compute encoding for each position
    img = img.transpose() # (HEIGHT,WIDTH) -> (WIDTH,HEIGHT)

    return img

def generate_image(chrX, position, bamfile, ref, reads, ref_dict=None, is_negative_gen=False, mutate=None, seed=1):
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

    for idx, read in enumerate(reads):
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

def create_tumor_only_input_tensor_for_position(chromosome, position, bamfile_t, ref_file, args=None, y=None):
    """
    """
    # get fetch positions
    fetch_region_start = position - c.TUMOR_ONLY_FLANK
    fetch_region_end = position + c.TUMOR_ONLY_FLANK + 1
    
    tumor_reads = get_reads_tumor_only(bamfile_t, chromosome, fetch_region_start, fetch_region_end)
    
    try:
        # list of ref seqs
        ref_sequence = get_reference(ref_file, chromosome, position, c.TUMOR_ONLY_FLANK, return_sequence=True)
    except KeyError:
        print(("%s %s. location not in reference" % (chromosome, str(position))))
        raise
    except ValueError:
        print(("%s %s, failed to retrieve sequence" % (chromosome, str(position))))
        raise

    # tumor
    try:
        image_t = generate_image_tumor_only(chromosome, position, bamfile_t, ref_file, tumor_reads, ref_sequence, args=args, y=y)
    except ValueError:
        print(("%s %s. location not in tumor bam" % (chromosome, str(position))))
        raise
    
    if args and args.compute_stats:
        # return stats dictionary
        return image_t

    image_t = np.expand_dims(image_t, axis=0) # (W,H) -> (1,W,H)
    return image_t

def create_input_tensor_for_position(chromosome, position, bamfile_n, bamfile_t, ref_file, mutate=None):
    """
    Returns the input tensor for given position
    The shape of the output is (1, INPUT_SHAPE), as used during prediction by keras.

    """
    # get fetch positions
    fetch_region_flank =  int((c.SEQ_LENGTH - 1) / 2)
    fetch_region_start = position - fetch_region_flank
    fetch_region_end = position + fetch_region_flank + 1
    
    # fetch normal and tumor reads
    if bamfile_n:
        normal_reads = get_reads(bamfile_n, chromosome, fetch_region_start, fetch_region_end)
    else:
        # tumor-only mode
        normal_reads = []

    tumor_reads = get_reads(bamfile_t, chromosome, fetch_region_start, fetch_region_end)

    # how many times to sample c.NUM_READS reads from bams
    n_read_samples = 1 

    # sample reads n times if using c.SAMPLE_READS AND if there are more than 100 reads in either normal or tumor
    if c.SAMPLE_READS and c.MULTIPLE_READ_SAMPLES and (len(normal_reads) > c.NUM_READS or len(tumor_reads) > c.NUM_READS):
        n_read_samples = c.SAMPLE_READS_COUNT # round(max(len(normal_reads), len(tumor_reads)) / c.NUM_READS)
        print('read depth (max):', max(len(normal_reads), len(tumor_reads)))
    
    # if using all reads, set the height accordingly
    if c.VARIABLE_INPUT_SIZE:
        INPUT_HEIGHT = max(len(normal_reads), len(tumor_reads))
    else:
        INPUT_HEIGHT = c.INPUT_SHAPE[0] # default

    X = np.zeros((n_read_samples, INPUT_HEIGHT, c.INPUT_SHAPE[1], c.INPUT_SHAPE[2]), dtype=np.float32) # shape = (n, height, width, channels)

    # get ref channel. no need to do this n times. this must be done only if the batch contains a single site (i.e. c.SAMPLE_READS_COUNT > 1). NOT to be done if batch contains multiple sites
    try:
        ref_channel = get_reference(ref_file,  chromosome, position, c.FLANK, NUM_READS=INPUT_HEIGHT)
    
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
                image_n = generate_image_tetris_mode(chromosome, position, bamfile_n, ref_file)
            else:
                image_n = generate_image(chromosome, position, bamfile_n, ref_file, normal_reads, seed=sample)

        except ValueError:
            print(("%s %s. location not in normal bam" % (chromosome, str(position))))
            raise

        # tumor
        try:
            if c.TETRIS_MODE:
                image_t = generate_image_tetris_mode(chromosome, position, bamfile_t, ref_file)
            else:
                image_t = generate_image(chromosome, position, bamfile_t, ref_file, tumor_reads, mutate=mutate, seed=sample) # mutate only for tumor sample

        except ValueError:
            print(("%s %s. location not in tumor bam" % (chromosome, str(position))))
            raise
        
        # populate each sample in the input
        populate_array(sample-1, image_n, image_t, ref_channel, X)

    if not bamfile_n:
        # tumor-only mode, remove normal
        X = X[:, :, c.PER_IMAGE_WIDTH:, :]

    return X

def generate_images_for_positions(sample_name, positions_to_generate, normal_bam_path, tumor_bam_path, ref_file_path, training_data_file, args, batch_num=None, LIMIT_SAMPLES=None):
    """
    positions_to_generate = [ [ 'X', 2999, 1.0 ], [ 'X', 2998, 0.0 ] ... ]
    """
    print('Ref File:', ref_file_path)
    ref_file = pysam.FastaFile(ref_file_path) # pysam.FastaFile(args.ref_file)

    print(("Sample Name: %s" % sample_name))
    print(("Number of candidate positions %s" % str(len(positions_to_generate))))

    error_positions = 0

    if normal_bam_path is None:
        bamfile_n = None

    if not args.tumor_only and normal_bam_path is not None:
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
    skipped_sites = 0

    if args.compute_stats:
        global stats # modify the global stats defined below
        
    for idx, pos in enumerate(positions_to_generate):
        chromosome, position, y = str(pos[0]), int(pos[1]), float(pos[2])

        try:
            if args.ffpe:
                c.ENCODE_READ_ORIENTATION = False
                # generate multiple read samples only for the +ve sites in FFPE
                if y>0.5:
                    c.MULTIPLE_READ_SAMPLES = True
                else:
                    c.MULTIPLE_READ_SAMPLES = False

            if args.tumor_only:
                # tumor only transformer encoding                   
                X = create_tumor_only_input_tensor_for_position(chromosome, position, bamfile_t, ref_file, args=args, y=y)

                if args.compute_stats:
                    if X is None:
                        # no stats returned
                        continue

                    for stat in X.keys():
                        if stat not in stats:
                            stats[stat] = {}
                            stats[stat]['data'] = []
                            stats[stat]['Y'] = []

                        if X[stat] is not None:
                            stats[stat]['data'].append(X[stat]) # stat
                            stats[stat]['Y'].append(y) # label for stat
                    
                    if idx>LIMIT_SAMPLES:
                        # limit samples for computing stats, the input positions are randomly shuffled for args.compute_stats
                        break
                        
                else:
                    if X[0] is None:
                        skipped_sites += 1
                    else:
                        X_arr.append(X[0]) # tumor only input is (1,H,W)
                        Y.append(y)
            else:
                X = create_input_tensor_for_position(chromosome, position, bamfile_n, bamfile_t, ref_file)

                if X.shape[0] > 1:
                    print('Multiple read samples generated for +ve site')
                    # multiple read samples at site
                    for _ in range(X.shape[0]):
                        X_arr.append(X[_])
                        Y.append(y) # same label for all read samples
                else:
                    X_arr.append(X[0]) # squeeze (1,:,:,:) -> (:,:,:)
                    Y.append(y)
    
            del X
        
            num_done += 1

        except ValueError:
            num_errors += 1
            continue

    if args.compute_stats:
        return
        
    print('<<< Ignored sites >>>:', skipped_sites)

    print(("error positions %d " % error_positions))

    print(("done %d, errors %d" % (num_done, num_errors)))

    # import pdb; pdb.set_trace()
    
    assert len(X_arr) == len(Y)
    # convert to tensor with shape like (num_samples, 100, 35, 9) for X and (num_samples, 1) for Y
    x_length, y_length = len(X_arr), len(Y)
    X_arr = np.asarray(X_arr) # np.asarray converts a list of ndarrays to a single tensor of shape (len(X_arr) ,100, 35, 9)
    Y = np.asarray(Y)
    Y = Y.reshape((y_length, 1)) # 

    # assert X_arr.shape == tuple([x_length] + c.INPUT_SHAPE), '%s, %s' % (X_arr.shape, tuple([x_length] + c.INPUT_SHAPE))
    print(X_arr.shape)
    assert Y.shape == (y_length, 1)
    
    print(("output length: %d" % len(Y)))

    if batch_num is None: # generating patient in a single batch
        save_compressed_npy(training_data_file, X_arr, Y)
        print(("Saved %s" % training_data_file))
    else:
        save_compressed_npy(training_data_file + '.' + str(batch_num), X_arr, Y)
        print(("Saved %s batch %s" % (training_data_file, str(batch_num))))

def parse_vcf(vcf_path, args, is_negatives=False, snv=True):
    """
    Returns an array with Chromosome, Position, 1/0 (mutation probability) for each position in the file.
    The positions are returned 0-indexed
    """
    import vcf

    print('Parsing %s' % vcf_path)
    if snv:
        print('Extracting SNVs from Varnet VCF')
    else:
        print('Extracting indels from Varnet VCF')

    vcf_reader = vcf.Reader(open(vcf_path), 'r')

    sites = []
    for record in vcf_reader:
        if (snv and record.is_snp) or (not snv and record.is_indel):
            if not len(record.FILTER): # only use PASS calls
                sites.append([record.CHROM, record.POS-1, 0. if is_negatives else 1.]) # return 0-indexed POS, one-hot target
                #sites.append([record.CHROM, record.POS-1, float(record.INFO['SCORE']) ]) # return 0-indexed POS and varnet SCORE (soft target)

    print('Variant count:', len(sites))
    return sites


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
                if line.startswith('#'): continue
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
        positions_in_file.shuffle() # shuffle list before subsampling negatives
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

def parse_predictions_v2(predictions, ground_truth):
    """
    predictions is 0-indexed
    ground_truth is 1-indexed
    """
    trues = {}
    positions_in_file = []
    positives_in_file, negatives_in_file = 0,0

    # get the trues
    with open(ground_truth) as f:
        for line in f:
            if line.startswith('#'): continue
            line = line.strip()
            split_cols = line.split('\t')
            chromosome, position_1_indexed = split_cols[0], int(split_cols[1]) # 1-indexed trues
            position_0_indexed = position_1_indexed - 1
            pos_key = 'chrom%spos%d' % (chromosome, position_0_indexed)
            trues[pos_key] = True

    with open(predictions) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            split_cols = line.split('\t')
            chromosome, position_0_indexed = split_cols[0], int(split_cols[1]) # 0-indexed positions
            pos_key = 'chrom%spos%d' % (chromosome, position_0_indexed)

            if pos_key in trues:
                positives_in_file += 1
                positions_in_file.append( [chromosome, position_0_indexed, 1.0] )
            else:
                negatives_in_file += 1
                positions_in_file.append( [chromosome, position_0_indexed, 0.0] )

    print(("%d Total, %d positives, %d negatives" % (len(positions_in_file), positives_in_file, negatives_in_file)))
    return positions_in_file

def parse_positions(positions, fixed_label=None, one_indexed=False):
    """
    set one_indexed=True if file has 1 indexed positions 
    returns 0-indexed positions
    if fixed_label is not provided, it will get it from the file
    """
    positives_in_file, negatives_in_file = 0,0
    positions_in_file = []
    counts = {}

    # parse file with chrom, pos and label
    with open(positions) as f:
        for idx, line in enumerate(f):
            if line.startswith('#'): continue
            sp=line.strip().split()

            if fixed_label is None:
                # get label from file
                chrom, pos, label = sp[0], int(sp[1]), float(sp[2]) 
            else:
                chrom, pos, label = sp[0], int(sp[1]), fixed_label

            if one_indexed:
                pos = pos - 1 # 0-indexed position

            positions_in_file.append([chrom,pos,label]) # 0-indexed position

            if label in counts:
                counts[label] += 1
            else:
                counts[label] = 1
    
    print('Loading:', positions)
    print('Number of positions:', len(positions_in_file), counts)
    return positions_in_file

if __name__ == '__main__':
    from snvs.compress_npy_helper import save_compressed_npy, load_compressed_npy
    args = parse_args()

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

        if args.tumor_only:
            encoding_name = c.TUMOR_ONLY_ENCODING_NAME

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
       
        if args.ffpe:
            patients = [ patient for patient in c.FFPE_SAMPLES ]
            patients.sort()
            print('Num patients:', (len(patients)))
 
            patients = np.array_split(patients, int(args.num_nodes))[int(args.node_no)]

            for patient in patients:
                if type(patient) is np.ndarray: # list gets converted to ndarray in the np.array_split operation above
                    # for WGS GDC lung bams list
                    sample, normal_bam, tumor_bam = str(patient[0]), str(patient[1]), str(patient[2])
                else:
                    # for WES lung sample list
                    sample = patient

                positive_sites = parse_vcf(os.path.join(c.ffpe_vcfs_folder, sample, '%s_positives_v3.vcf' % sample), args, is_negatives=False, snv=True)
                negative_sites = parse_vcf(os.path.join(c.ffpe_vcfs_folder, sample, '%s_negatives_v3.vcf' % sample), args, is_negatives=True, snv=True)        
                
                # start internal WES lung Samples, generate train data on FFPE samples
                normal_bam = os.path.join(c.ffpe_bams_folder, '%s-FFPE-EH-N-ready.bam' % sample)
                tumor_bam = os.path.join(c.ffpe_bams_folder, '%s-FFPE-EH-T-ready.bam' % sample)
                # end internal WES lung
 
                if not len(positive_sites + negative_sites):
                    continue
                
                patient_files.append( (sample, negative_sites + positive_sites, normal_bam, tumor_bam ) )

        elif args.liver_data:
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

        elif args.mutect2_calls_on_ffpe:
            print('Parsing mutect2 calls on ffpe')
            patients = c.mutect2_calls_on_ffpe

            for patient in patients:
                patient_files.append( (patient[0], parse_predictions_v2(patient[1], patient[2]), patient[3], patient[4]) ) 

        elif args.strelka2_calls_on_ffpe:
            print('Parsing strelka2 calls on ffpe')
            patients = c.strelka2_calls_on_ffpe

            for patient in patients:
                patient_files.append( (patient[0], parse_predictions_v2(patient[1], patient[2]), patient[3], patient[4]) )
            
        elif args.ffpe_wgs_wes_training:
            print('FFPE WGS + WES training on SEQC2 and TCGA')
            patients = c.ffpe_wgs_wes_training
            #c.ENCODE_READ_ORIENTATION = True

            args.tumor_only = True
            patients = np.array_split(patients, int(args.num_nodes))[int(args.node_no)]

            from pprint import pprint
            pprint(patients)
            
            for patient in patients:
                patient_files.append((patient[0], parse_positions(patient[1]), patient[2], patient[3], patient[4])) 
        
        elif args.tcga_wxs:
            print('Tumor-only mode')
            samples_root = '/scratch/users/astar/gis/krishnak/TCGA/samples_with_context'
            patients = os.listdir(samples_root)

            patients.sort()
            random.seed(0)
            random.shuffle(patients)
            
            patients = np.array_split(patients, int(args.num_nodes))[int(args.node_no)]
            ref_file = '/scratch/users/astar/gis/krishnak/project/GRCh38.d1.vd1.fa'

            args.tumor_only = True

            for patient in patients:
                files = os.listdir(os.path.join(samples_root, patient))
                print(patient)
                tumor_bam_file = [x for x in files if x.endswith('.bam')][0]
                tumor_bam_file = os.path.join(samples_root, patient, tumor_bam_file)
                all_positions_per_patient = []

                # SNVS and indels
                if 'somatic_snvs_indels.txt' in files:
                    somatic_snvs_indels = os.path.join(samples_root, patient, 'somatic_snvs_indels.txt')
                    somatic_snvs_indels = parse_positions(somatic_snvs_indels, fixed_label=1, one_indexed=True)
                    all_positions_per_patient += somatic_snvs_indels

                # if 'artifact_snvs_indels.txt' in files:
                if 'artifact_snvs_indels_v2.txt' in files:
                    # artifact_snvs_indels = os.path.join(samples_root, patient, 'artifact_snvs_indels.txt') # artifact filters defined by callers (no PASS calls)
                    artifact_snvs_indels = os.path.join(samples_root, patient, 'artifact_snvs_indels_v2.txt') # PASS calls from callers not in ground-truth are artifacts
                    artifact_snvs_indels = parse_positions(artifact_snvs_indels, fixed_label=0, one_indexed=True)
                    all_positions_per_patient += artifact_snvs_indels

                # germline snvs
                # use intersection of germline calls on tumor and matched normal in case haplotypecaller misidentified somatic variants in tumor as germline
                if 'gatk.tumor_normal_intersection.ready_snps.vcf' in files:
                    germline_snvs = os.path.join(samples_root, patient, 'gatk.tumor_normal_intersection.ready_snps.vcf')
                    germline_snvs = parse_positions(germline_snvs, fixed_label=2, one_indexed=True)
                    all_positions_per_patient += germline_snvs

                # germline indels
                # use intersection of germline calls on tumor and matched normal in case haplotypecaller misidentified somatic variants in tumor as germline
                if 'gatk.tumor_normal_intersection.ready_indels.vcf' in files:
                    germline_indels = os.path.join(samples_root, patient, 'gatk.tumor_normal_intersection.ready_indels.vcf')
                    germline_indels = parse_positions(germline_indels, fixed_label=2, one_indexed=True)
                    all_positions_per_patient += germline_indels

                patient_files.append((patient, all_positions_per_patient, None, tumor_bam_file, ref_file))

            Parallel(n_jobs=int(args.num_processes))(delayed(generate_images_for_positions)(patient[0], patient[1], patient[2], patient[3], patient[4], os.path.join(training_data_folder, patient[0]), args) for patient in patient_files if not os.path.exists(os.path.join(training_data_folder, '%s.npz' % (patient[0]))))
        
            sys.exit()

        elif args.tcga_wxs_convnet:
            print('TCGA WXS Convnet Train Data')
            samples_root = '/scratch/users/astar/gis/krishnak/TCGA/samples_with_context'
            patients = os.listdir(samples_root)

            patients.sort()
            random.seed(0)
            random.shuffle(patients)
            
            patients = np.array_split(patients, int(args.num_nodes))[int(args.node_no)]
            ref_file = '/scratch/users/astar/gis/krishnak/project/GRCh38.d1.vd1.fa'

            for patient in patients:
                files = os.listdir(os.path.join(samples_root, patient))
                print(patient)
                tumor_bam_file = [x for x in files if x.endswith('.bam')][0]
                tumor_bam_file = os.path.join(samples_root, patient, tumor_bam_file)
                all_positions_per_patient = []

                # SNVS
                if 'somatic_snvs.txt' in files:
                    somatic_snvs = os.path.join(samples_root, patient, 'somatic_snvs.txt')
                    somatic_snvs = parse_positions(somatic_snvs, fixed_label=1, one_indexed=True)
                    all_positions_per_patient += somatic_snvs

                if 'artifact_snvs.txt' in files:
                    # artifact_snvs = os.path.join(samples_root, patient, 'artifact_snvs.txt') # artifact filters defined by callers (no PASS calls)
                    artifact_snvs = os.path.join(samples_root, patient, 'artifact_snvs_v2.txt') # SNV PASS calls from callers not in ground-truth are artifacts
                    artifact_snvs = parse_positions(artifact_snvs, fixed_label=0, one_indexed=True)
                    all_positions_per_patient += artifact_snvs

                # germline snvs
                # use intersection of germline calls on tumor and matched normal in case haplotypecaller misidentified somatic variants in tumor as germline
                if 'gatk.tumor_normal_intersection.ready_snps.vcf' in files:
                    germline_snvs = os.path.join(samples_root, patient, 'gatk.tumor_normal_intersection.ready_snps.vcf')
                    germline_snvs = parse_positions(germline_snvs, fixed_label=2, one_indexed=True)
                    all_positions_per_patient += germline_snvs

                patient_files.append((patient, all_positions_per_patient, None, tumor_bam_file, ref_file))

            Parallel(n_jobs=int(args.num_processes))(delayed(generate_images_for_positions)(patient[0], patient[1], patient[2], patient[3], patient[4], os.path.join(training_data_folder, patient[0]), args) for patient in patient_files if not os.path.exists(os.path.join(training_data_folder, '%s.npz' % (patient[0]))))
        
            sys.exit()
            
        elif args.compute_stats:
            print('>>> Computing stats for training sites')
            # compute stats for FFPE samples (true vs artifacts)
            # use smudl/plot_stats.py to plot the data

            # <start> SEQC2 FFPE training data 
            patients = c.ffpe_wgs_wes_training 
            # filename = 'SEQC2_FFPE_combined_stats.npy' # original seqc2 ffpe wgs + wxs
            # filename = 'SEQC2_FFPE_with_diluted_tumor_combined_stats.npy' # includes 24h diluted to 50%
            filename = 'SEQC2_FFPE_with_more_diluted_tumor_combined_stats.npy' #  includes 24h diluted to 50%, 40%, 30%
            LIMIT_SAMPLES = 0.20 # percentage of samples to use per sample. the positions are shuffled below for each sample
            # for patient in patients:
            #     patient_files.append((patient[0], parse_positions(patient[1]), patient[2], patient[3], patient[4])) 
            # </start> SEQC2 FFPE training data 

            # <start> FFPE lung WES cohort
            # patients = c.ffpe_lung_cohort_calls # A001, ...
            # filename = 'FFPE_lung_cohort_combined_stats.npy'
            # LIMIT_SAMPLES = 0.20 # percentage of samples to use per sample. the positions are shuffled below for each sample
            # for patient in patients:
            #     patient_files.append((patient[0], parse_positions(patient[1]), patient[2], patient[3], patient[4])) 
            # </start> FFPE lung WES cohort

            # <start> TCGA 10k WES training data 
            # samples_root = '/scratch/users/astar/gis/krishnak/TCGA/samples_with_context'
            # patients = os.listdir(samples_root)
            # filename = 'TCGA_combined_stats.npy' # somatic, germline and artifacts
            # LIMIT_SAMPLES = 0.005 # percentage of samples to use per sample. the positions are shuffled below for each sample
            # ref_file = '/scratch/users/astar/gis/krishnak/project/GRCh38.d1.vd1.fa'

            # for patient in patients:
            #     files = os.listdir(os.path.join(samples_root, patient))
            #     tumor_bam_file = [x for x in files if x.endswith('.bam')][0]
            #     tumor_bam_file = os.path.join(samples_root, patient, tumor_bam_file)
            #     all_positions_per_patient = []

            #     # SNVS and indels
            #     if 'somatic_snvs_indels.txt' in files:
            #         somatic_snvs_indels = os.path.join(samples_root, patient, 'somatic_snvs_indels.txt')
            #         somatic_snvs_indels = parse_positions(somatic_snvs_indels, fixed_label=1, one_indexed=True)
            #         all_positions_per_patient += somatic_snvs_indels
            #     if 'artifact_snvs_indels.txt' in files:
            #         artifact_snvs_indels = os.path.join(samples_root, patient, 'artifact_snvs_indels.txt')
            #         artifact_snvs_indels = parse_positions(artifact_snvs_indels, fixed_label=0, one_indexed=True)
            #         all_positions_per_patient += artifact_snvs_indels
            #     # germline snvs
            #     # use intersection of germline calls on tumor and matched normal in case haplotypecaller misidentified somatic variants in tumor as germline
            #     if 'gatk.tumor_normal_intersection.ready_snps.vcf' in files:
            #         germline_snvs = os.path.join(samples_root, patient, 'gatk.tumor_normal_intersection.ready_snps.vcf')
            #         germline_snvs = parse_positions(germline_snvs, fixed_label=2, one_indexed=True)
            #         all_positions_per_patient += germline_snvs
            #     # germline indels
            #     # use intersection of germline calls on tumor and matched normal in case haplotypecaller misidentified somatic variants in tumor as germline
            #     if 'gatk.tumor_normal_intersection.ready_indels.vcf' in files:
            #         germline_indels = os.path.join(samples_root, patient, 'gatk.tumor_normal_intersection.ready_indels.vcf')
            #         germline_indels = parse_positions(germline_indels, fixed_label=2, one_indexed=True)
            #         all_positions_per_patient += germline_indels

            #     patient_files.append((patient, all_positions_per_patient, None, tumor_bam_file, ref_file))
            # <start> TCGA 10k WES training data 

            # <start> ICGC CLL/MBL and TGEN
            # CLL hg38
            # patients = [['icgc_cll_hg38_snvs', '/scratch/users/astar/gis/krishnak/project/workstation_backup/smudl/ground_truths/icgc_cll_hg38_snvs.txt', '/scratch/users/astar/gis/krishnak/hg38-cll-mbl-tgen/icgc_cll-hg38-WH-N-ready.bam', '/scratch/users/astar/gis/krishnak/hg38-cll-mbl-tgen/icgc_cll-hg38-WH-T-ready.bam', '/scratch/users/astar/gis/krishnak/hg38.fa']]
            # filename = 'CLL_snvs_combined_stats.npy'

            # # MBL hg38
            # patients = [['icgc_mbl_hg38_snvs', '/scratch/users/astar/gis/krishnak/project/workstation_backup/smudl/ground_truths/icgc_mbl_hg38_snvs.txt', '/scratch/users/astar/gis/krishnak/hg38-cll-mbl-tgen/icgc_mbl-hg38-WH-N-ready.bam', '/scratch/users/astar/gis/krishnak/hg38-cll-mbl-tgen/icgc_mbl-hg38-WH-T-ready.bam', '/scratch/users/astar/gis/krishnak/hg38.fa']]
            # filename = 'MBL_snvs_combined_stats.npy'

            # # TGEN hg38
            patients = [['tgen_hg38_snvs', '/scratch/users/astar/gis/krishnak/project/workstation_backup/smudl/ground_truths/tgen_hg38_snvs.txt', '/scratch/users/astar/gis/krishnak/hg38-cll-mbl-tgen/tgen_colo829-hg38-WH-N-ready.bam', '/scratch/users/astar/gis/krishnak/hg38-cll-mbl-tgen/tgen_colo829-hg38-WH-T-ready.bam', '/scratch/users/astar/gis/krishnak/hg38.fa']]
            filename = 'TGEN_snvs_combined_stats.npy'

            LIMIT_SAMPLES = 1.00 # percentage of samples to use per sample. the positions are shuffled below for each sample
            for patient in patients:
                patient_files.append((patient[0], parse_positions(patient[1], one_indexed=True, fixed_label=1), patient[2], patient[3], patient[4])) 
            # </start> ICGC CLL/MBL and TGEN

            args.tumor_only = True

            # global variable
            stats = {} # combined across all patient_files

            for patient in patient_files:
                # the positions are shuffled using random.shuffle as we will only compute stats for a sample of the training data
                # see generate_images_for_positions() for the loop cut off using idx > LIMIT_SAMPLES for args.compute_stats
                random.shuffle(patient[1])
                generate_images_for_positions(patient[0], patient[1], patient[2], patient[3], patient[4], filename, args, LIMIT_SAMPLES=int(LIMIT_SAMPLES*len(patient[1])))

            for stat in stats.keys():
                assert len(stats[stat]['data']) == len(stats[stat]['Y'])
            
            # save global variable stats
            np.save(filename, stats)
            print('Saved:', filename)

            sys.exit()

        elif args.parse_ffpe_calls:
            # parse bcbio-1pct ffpe calls made by varnet, strelka2 and mutect2
            # 0-indexed snv and indel combined. created using ~/scratch/ffpe_parse_vcfs/create_ffpe_ground_truth.py
            patients = c.ffpe_lung_cohort_calls 

            patients.sort()            
            patients = np.array_split(patients, int(args.num_nodes))[int(args.node_no)]

            args.tumor_only = True # generate tumor only encoding

            for patient in patients:
                filename = os.path.basename(patient[1])
                target_file = os.path.join(training_data_folder, filename.replace('.csv', '.npz'))
                if not os.path.exists(target_file):
                    generate_images_for_positions(patient[0], parse_positions(patient[1]), patient[2], patient[3], patient[4], target_file, args)
                    print('Saved:', target_file)
                else:
                    print('File already exists:', target_file)

            sys.exit()

        for patient in patient_files:
            split_positions = np.array_split(patient[1], (len(patient[1])/200000) + 1) # can't fit more than 200k positions in memory, so split the list

            training_data_file = os.path.join(training_data_folder, patient[0])
            print(training_data_file)        
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
            Parallel(n_jobs=int(args.num_processes))(delayed(generate_images_for_positions)(patient[0], batch, patient[2], patient[3], patient[4], training_data_file, args, idx) for idx, batch in enumerate(split_positions) if not os.path.exists(os.path.join(training_data_folder, '%s.%d.npz' % (patient[0], idx) )))
 

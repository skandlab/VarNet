import os
import sys
import pysam
import argparse
import numpy as np
from joblib import Parallel, delayed, __version__
from indels import constants as c
from snvs import generate_training_data
from snvs.generate_training_data import get_reads
from utils import sample_reads_fn

import pickle as pickle

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
    parser.add_argument('--goldset_data', default='')
    parser.add_argument('--num_nodes', default=1)
    parser.add_argument('--node_no', default=0)
    parser.add_argument('--num_processes', default=1)
    parser.add_argument('--balance_positive_negative', default=True)

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

    if parser.parse_args().balance_positive_negative == 'yes':
        parser.parse_args().balance_positive_negative = True

    return parser.parse_args()


class BaseVals(object):
    """
    Keep these dictionaries in memory
    expensive to recreate dict for every base
    """
    reg = {'A': 250, 'G': 220, 'T': 190, 'C': 160, 'D': 0, 'N': 0, 'W': 0,
           'S': 0, 'M': 0, 'K': 0, 'R': 0, 'Y': 0, 'B': 0, 'H': 0, 'V': 0}
    insert = {'A': 130, 'G': 100, 'T': 70, 'C': 40, 'D': 0, 'N': 0, 'W': 0,
              'S': 0, 'M': 0, 'K': 0, 'R': 0, 'Y': 0, 'B': 0, 'H': 0, 'V': 0}
    delete = 10

    @classmethod
    def get_base_val(cls, base):
        try:
            return cls.reg[base]
        except:
            return 0

    @classmethod
    def get_insertion_val(cls, base):
        try:
            return cls.insert[base]
        except:
            return 0

    @classmethod
    def get_deletion_val(cls):
        return cls.delete


def get_color(quality):
    return int(255.0 * (min(45, quality) / 45.0))


def get_strand_dir(on_positive_strand):
    return 125 if on_positive_strand else 250


def get_mapping_quality(mapping_qual):
    return 0 if np.isclose(mapping_qual, 255.) else get_color(mapping_qual)


def encode_base_qualities(quals):
    return np.array([get_color(qual) for qual in quals], dtype=np.uint8)


def encode_strand_dir(on_positive_strand, length):
    return np.full(length, get_strand_dir(on_positive_strand), dtype=np.uint8)


def encode_mapping_quality(mapping_qual, length):
    return np.full(length, get_mapping_quality(mapping_qual), dtype=np.uint8)


def get_reference(chrX, pos, config, insertions, num_rows, img, ref_sequence):
    start_pos = config['start']
    end_pos = config['end']
    encoded_seq = []

    if start_pos < 0:
        # Start position is before ref file starts
        encoded_seq += [0] * (-start_pos)
        start_pos = 0

    read_index = 0
    ref_index = start_pos

    while(read_index < len(ref_sequence)):
        encoded_seq.append(BaseVals.get_base_val(ref_sequence[read_index]))
        if ref_index in insertions:
            encoded_seq += [0] * insertions[ref_index]
        read_index += 1
        ref_index += 1
    if config['insertions_before_middle']:
        del encoded_seq[:config['insertions_before_middle']]
    if config['insertions_after_middle']:
        del encoded_seq[-config['insertions_after_middle']:]
    if len(encoded_seq) < c.SEQ_LENGTH:
        # End position is after end of chromosome - fill end with zeros
        encoded_seq += [0] * (c.SEQ_LENGTH - len(encoded_seq))
    elif len(encoded_seq) > c.SEQ_LENGTH:
        raise ValueError

    img[:num_rows, :c.SEQ_LENGTH] = encoded_seq
    img[:num_rows, c.SEQ_LENGTH:] = encoded_seq


def encode_filler(max_insertion_size, curr_insertion_size, encoded_seq, quals):
    encoded_seq += [0] * (max_insertion_size - curr_insertion_size)
    quals += [0] * (max_insertion_size - curr_insertion_size)


def encode_insertion(read_pos, max_insertion_size, curr_insertion_size,
                     read, encoded_seq, quals):
    for pos in range(read_pos, read_pos+curr_insertion_size):
        base = read.query_sequence[pos]
        encoded_seq.append(BaseVals.get_insertion_val(base))
        quals.append(read.query_qualities[pos])
    # If max insertion size at position > read insertion size
    encode_filler(max_insertion_size, curr_insertion_size, encoded_seq, quals)


def encode_deletion(ref_pos, read, encoded_seq, quals):
    encoded_seq.append(BaseVals.get_deletion_val())
    quals.append(
        generate_training_data.get_average_neighbouring_base_quality(
                                                                read, ref_pos)
    )


def encode_match(read_pos, read, encoded_seq, quals):
    encoded_seq.append(BaseVals.get_base_val(read.query_sequence[read_pos]))
    quals.append(read.query_qualities[read_pos])


def get_starting_position_in_image(ref_pos, start_pos, insertions):
    read_starts_at = ref_pos - start_pos
    for pos in insertions:
        if pos < ref_pos:
            read_starts_at += insertions[pos]
    return read_starts_at


def get_bases_to_fill(read, config, insertions):
    encoded_seq = []
    quals = []
    read_starts_at = 0

    # List of read/query and reference positions, soft clip bases removed
    all_pairs = generate_training_data.get_positions_to_fill(read)

    curr_insertions = {}
    find_insertion_positions(read, curr_insertions, config['start'],
                             config['end'], config['middle'])

    for idx, pair in enumerate(all_pairs):
        read_pos = pair[0]
        ref_pos = pair[1]
    
        if ref_pos is None and idx > 0:  # Ref pos is None for insertions
            last_ref_pos = all_pairs[idx - 1][1]

            if(last_ref_pos is None or last_ref_pos < config['start'] or
               last_ref_pos >= config['end'] or
               last_ref_pos not in insertions):
                continue

            curr_insertion_size = curr_insertions[last_ref_pos]
            max_insertion_size = insertions[last_ref_pos]
            encode_insertion(read_pos, max_insertion_size,
                             curr_insertion_size, read, encoded_seq, quals)

        elif (ref_pos is not None and
              ref_pos >= config['start'] and ref_pos < config['end']):

            if len(encoded_seq) == 0:
                read_starts_at = get_starting_position_in_image(
                                        ref_pos, config['start'], insertions)
            if read_pos is None:
                encode_deletion(ref_pos, read, encoded_seq, quals)
            else:
                encode_match(read_pos, read, encoded_seq, quals)

            if ref_pos in insertions and ref_pos not in curr_insertions:
                encode_filler(insertions[ref_pos], 0, encoded_seq, quals)

        elif ref_pos is not None and ref_pos >= config['end']:
            break

    if config['insertions_before_middle']:
        del encoded_seq[:config['insertions_before_middle']]
        del quals[:config['insertions_before_middle']]

    if read_starts_at + len(encoded_seq) > c.SEQ_LENGTH:
        to_delete = (read_starts_at + len(encoded_seq)) - c.SEQ_LENGTH
        del encoded_seq[-to_delete:]
        del quals[-to_delete:]

    return encoded_seq, quals, read_starts_at


def stack_read_in_image(read, img, i, config, insertions):
    encoded_bases, base_qualities, j = get_bases_to_fill(read, config, insertions)

    if(len(encoded_bases) == 0):
        return False

    if(len(encoded_bases) != len(base_qualities)):
        raise ValueError

    size = len(encoded_bases)
    img[i, j:j+size, 0] = encoded_bases
    img[i, j:j+size, 1] = encode_base_qualities(base_qualities)
    img[i, j:j+size, 2] = encode_strand_dir(not read.is_reverse, size)
    img[i, j:j+size, 3] = encode_mapping_quality(read.mapping_quality, size)

    return True


def generate_image(reads, config, insertions, start_at_read, img):
    image_row = 0

    for i in range(start_at_read, len(reads)):
        if stack_read_in_image(reads[i], img, image_row, config, insertions):
            image_row += 1

            if image_row >= c.NUM_READS and not c.VARIABLE_INPUT_SIZE:
                break

    return image_row


def find_insertion_positions(read, insertions, start, end, position):
    """
    Finds the longest insertions at each given reference position in the read.

    Args:
        read       - The pysam read to find insertions of.
        insertions - A dictionary mapping starting reference position to
                     length of insertion.
        start      - Start place to search for insertions in the read.
        end        - End place to look for insertions in the read.
    Modifies:
        insertions - Adds insertion if never seen at that position before.
                   - Replaces insertion if longer insertion is found at same
                     position. (For multiple reads)
    """
    if "I" in read.cigarstring:
        ref_pos = read.reference_start - 1
        for i, items in enumerate(read.cigartuples):
            if (i == 0 and (items[0] == 4 or items[0] == 1)):
                # Don't include soft clip/insertions at the front of the read
                continue
            if items[0] == 1:
                if(ref_pos >= start and ref_pos < end):
                    if(ref_pos not in insertions
                       or items[1] > insertions[ref_pos]):
                        insertions[ref_pos] = items[1]
            else:
                ref_pos += items[1]


def get_insertions(normal_reads, tumor_reads, position, config):
    insertions = {}
    for read in normal_reads:
        find_insertion_positions(read, insertions,
                                 config['start'], config['end'], position)
    for read in tumor_reads:
        find_insertion_positions(read, insertions,
                                 config['start'], config['end'], position)
    return insertions

def get_start(position):
    return position - c.FLANK + 1

def get_end(position):
    return position + c.FLANK + 2


def get_config(position, start_pos, end_pos):
    config = {}
    config['insertions_before_middle'] = 0
    config['insertions_after_middle'] = 0
    config['n_start'] = 0
    config['t_start'] = 0
    config['start'] = start_pos
    config['end'] = end_pos
    config['middle'] = int((start_pos + end_pos) / 2 - 1)
    return config


def update_config(config, n_length, t_length, insertions):
    if n_length > c.NUM_READS:
        config['n_start'] = int((n_length - c.NUM_READS) / 2)
    if t_length > c.NUM_READS:
        config['t_start'] = int((t_length - c.NUM_READS) / 2)
    for ref_pos in insertions:
        if ref_pos < config['middle']:
            config['insertions_before_middle'] += insertions[ref_pos]
        else:
            config['insertions_after_middle'] += insertions[ref_pos]


def create_input_tensor_for_position(chrX, position, bamfile_n, bamfile_t, ref_file, dtype=np.float32):
    start_pos = get_start(position)
    end_pos = get_end(position)

    normal_reads = get_reads(bamfile_n, chrX, start_pos, end_pos)
    tumor_reads = get_reads(bamfile_t, chrX, start_pos, end_pos)

    # how many times to sample c.NUM_READS reads from bams
    n_read_samples = 1

    # sample reads n times if c.SAMPLE_READS AND if there are more than c.NUM_READS reads in either normal or tumor
    if c.SAMPLE_READS and (len(normal_reads) > c.NUM_READS or len(tumor_reads) > c.NUM_READS):
        n_read_samples = c.SAMPLE_READS_COUNT

    if c.VARIABLE_INPUT_SIZE:
        if len(normal_reads) > c.MAX_READS: # cap on reads even if variable_input_size
            normal_reads = sample_reads_fn(normal_reads, c.MAX_READS)

        if len(tumor_reads) > c.MAX_READS:
            tumor_reads = sample_reads_fn(tumor_reads, c.MAX_READS)

        INPUT_HEIGHT = max(len(normal_reads), len(tumor_reads), c.INPUT_SHAPE[0]) # minimum height is 140, even if normal_reads & tumor_reads < 140
        X = np.zeros((n_read_samples, INPUT_HEIGHT, c.INPUT_SHAPE[1], c.INPUT_SHAPE[2]), dtype=dtype) # shape = (n, height, width, channels)
    else:
        X = np.zeros(tuple([n_read_samples] + c.INPUT_SHAPE), dtype=dtype)
    
    # get ref sequence once per site. for each read sample, the config maybe different due to positions of indels in each sample of reads. simply fetch the ref_sequence and pass to get_reference below for maniupulation
    try:
        ref_sequence = ref_file.fetch(chrX, start_pos if start_pos >=0 else 0, end_pos).upper()   
    except KeyError:
        if chrX == 'chrM':
            chrX = 'MT'
        else:
            chrX = chrX.replace('chr', '')
        ref_sequence = ref_file.fetch(chrX, start_pos if start_pos >=0 else 0, end_pos).upper() # simply fetch the ref_sequence and pass to get_reference for maniupulation
    except:
        print('error in fetch', chrX, position, start_pos, end_pos)
        raise
 
    for sample in range(1, n_read_samples + 1): # samples [1,2...,n]
        normal_reads = sample_reads_fn(normal_reads, c.NUM_READS, seed=sample) # sample reads while preserving order in reads list. default seed=1
        tumor_reads = sample_reads_fn(tumor_reads, c.NUM_READS, seed=sample) # sample reads while preserving order in reads list. default seed=1

        config = get_config(position, start_pos, end_pos)
        insertions = get_insertions(normal_reads, tumor_reads, position, config)
        update_config(config, len(normal_reads), len(tumor_reads), insertions)

        try:
            # Create a view of the image to reduce amount of copying
            normal_img = X[sample-1, :, :c.SEQ_LENGTH, :c.NUM_CHANNELS_PER_IMAGE]
            num_rows_n = generate_image(normal_reads, config, insertions,
                                        config['n_start'], normal_img)
        except ValueError:
            print((
                "Skipping %s %s. location not in normal bam"
                % (chrX, str(position))
            ))
            raise

        try:
            tumor_img = X[sample-1, :, c.SEQ_LENGTH:, :c.NUM_CHANNELS_PER_IMAGE]
            num_rows_t = generate_image(tumor_reads, config, insertions,
                                        config['t_start'], tumor_img)
        except ValueError:
            print((
                "Skipping %s %s. location not in tumor bam"
                % (chrX, str(position))
            ))
            raise

        # ref channel
        try:
            ref_img = X[sample-1, :, :, c.NUM_CHANNELS_PER_IMAGE]
            get_reference(chrX, position, config, insertions, max(num_rows_n, num_rows_t), ref_img, ref_sequence) # not filling ref bases all over image like in snv

        except KeyError:
            print((
                "Skipping %s %s. location not in reference"
                % (chrX, str(position))
            ))
            raise

        except ValueError:
            print((
                "Skipping %s %s, failed to retrieve sequence"
                % (chrX, str(position))
            ))
            raise

    return X

def get_ref_file():
    return pysam.FastaFile(c.ref_path)

def create_sampled_input_tensor_for_position(chrX, position, bamfile_n, bamfile_t, ref_file):

    start_pos = get_start(position)
    end_pos = get_end(position)

    normal_reads = get_reads(bamfile_n, chrX, start_pos, end_pos)
    tumor_reads = get_reads(bamfile_t, chrX, start_pos, end_pos)

    config = get_config(position, start_pos, end_pos)
    insertions = get_insertions(normal_reads, tumor_reads, position, config)
    update_config(config, len(normal_reads), len(tumor_reads), insertions)

    max_reads = max(normal_reads, tumor_reads)
    times_to_sample = int(max_reads/c.NUM_READS) + 1

    X = np.zeros(([times_to_sample] + c.INPUT_SHAPE), dtype=np.float32)

    for i in range(times_to_sample):
        try:
            if len(normal_reads) > c.NUM_READS:
                sampled_reads = np.random.choice(normal_reads, c.NUM_READS, replace=False)
            else:
                sampled_reads = normal_reads
            # Create a view of the image to reduce amount of copying
            normal_img = X[i, :, :c.SEQ_LENGTH, :c.NUM_CHANNELS_PER_IMAGE]
            num_rows_n = generate_image(sampled_reads, config, insertions,
                                        config['n_start'], normal_img)
        except ValueError:
            print((
                "Skipping %s %s. location not in normal bam"
                % (chrX, str(position))
            ))
            raise
        try:
            if len(tumor_reads) > c.NUM_READS:
                sampled_reads = np.random.choice(tumor_reads, c.NUM_READS, replace=False)
            else:
                sampled_reads = tumor_reads
            tumor_img = X[i, :, c.SEQ_LENGTH:, :c.NUM_CHANNELS_PER_IMAGE]
            num_rows_t = generate_image(sampled_reads, config, insertions,
                                        config['t_start'], tumor_img)
        except ValueError:
            print((
                "Skipping %s %s. location not in tumor bam"
                % (chrX, str(position))
            ))
            raise
        try:
            ref_img = X[i, :, :, c.NUM_CHANNELS_PER_IMAGE]
            get_reference(chrX, position, config, insertions,
                          max(num_rows_n, num_rows_t), ref_img, ref_file)
        except KeyError:
            print((
                "Skipping %s %s. location not in reference"
                % (chrX, str(position))
            ))
            raise
        except ValueError:
            print((
                "Skipping %s %s, failed to retrieve sequence"
                % (chrX, str(position))
            ))
            raise

    #print "Done chromosome %s position %s" % (chrX, str(position))
    return X

def generate_images_for_positions(sample_name, positions_to_generate,
                                  normal_bam_path, tumor_bam_path, save_file,
                                  batch_num=None, node_no=None):
    """
    positions_to_generate = [ [ 'X', 2999, 1.0 ], [ 'X', 2998, 0.0 ] ... ]
    """
    ref_file = get_ref_file()

    if batch_num is not None:
        save_file += '.' + str(batch_num)
    if node_no is not None:
        save_file += '.' + str(node_no)
    save_file = save_file + ".npz"

    if os.path.isfile(save_file):
        print(("File already exists: %s" % save_file))
        return

    print(("Sample Name: %s" % sample_name))
    print(("Number of candidate positions %s" % str(len(positions_to_generate))))
    print(("File Name: %s" % save_file))
    try:
        print(("NORMAL BAM: %s" % normal_bam_path))
        bamfile_n = pysam.AlignmentFile(str(normal_bam_path), "rb")
    except IOError:
        print(("IOError: %s" % normal_bam_path))
        return
    try:
        print(("TUMOR BAM: %s" % tumor_bam_path))
        bamfile_t = pysam.AlignmentFile(str(tumor_bam_path), "rb")
    except IOError:
        print(("IOError: %s" % tumor_bam_path))
        return

    Y = []
    X_arr = []
    for c_p_y in positions_to_generate:
        chrX, position, y = str(c_p_y[0]), int(c_p_y[1]), float(c_p_y[2])
        try:
            X = create_input_tensor_for_position(
                                    chrX, position, bamfile_n, bamfile_t, ref_file)
            X_arr.append(X[0, :, :, :])
            del X
            Y.append(y)
        except ValueError:
            print((
                "Error appending training position at "
                "chromosome %s position %s" % (chrX, str(position))
            ))
            continue

    assert len(X_arr) == len(Y)

    length = len(Y)
    X_arr = np.asarray(X_arr)
    Y = np.asarray(Y)
    Y = Y.reshape((length, 1))
    generate_training_data.save_compressed_npy(save_file, X_arr, Y)
    print(("Saved %s" % save_file))

def get_positions_from_set(path_to_set):
    positions = []
    with open(path_to_set, 'rb') as input:
        s = pickle.load(input)
        for item in s:
            sp = item.split('/')
            chrX, position, is_real = sp[0], int(sp[1]), float(sp[2])
            position -= 1
            positions.append([chrX, position, is_real])
    return positions

def balance_positives_and_negatives(positives, negatives):
    if len(negatives) >= len(positives):
        np.random.shuffle(negatives)
        positions_in_file = positives + negatives[:len(positives)]
    else:
        np.random.shuffle(positives)
        positions_in_file = positives[:len(negatives)] + negatives
    return positions_in_file

def parse_smurf_file(predictions_path, true_file, balance):
    # Need a true file for smurf positions as they don't come
    # With actual labeled positions
    positives_in_file = []
    negatives_in_file = []
    goldset_positions = set()
    with open(true_file, "r") as f:
        next(f)
        for line in f:
            line = line.strip()
            split_cols = line.split('\t')
            goldset_positions.add((str(split_cols[0]), int(split_cols[1])))
    with open(predictions_path) as f:
        next(f)
        for line in f:
            line =  line.strip()
            split_cols = line.split('\t')
            chromosome, position_1_indexed = str(split_cols[1]), int(split_cols[2])
            position_0_indexed = position_1_indexed - 1
            if (chromosome, position_1_indexed) in goldset_positions:
                positives_in_file.append([chromosome, position_0_indexed, 1])
            else:
                negatives_in_file.append([chromosome, position_0_indexed, 0])
    if balance:
        positions_in_file = balance_positives_and_negatives(positives_in_file,
                                                            negatives_in_file)
    else:
        positions_in_file = positives_in_file + negatives_in_file

    return positions_in_file

def parse_predictions_file(predictions_path, balance):
    positives_in_file = []
    negatives_in_file = []

    with open(predictions_path) as f:
        next(f)
        for line in f:
            line = line.strip()
            split_cols = line.split('\t')
            chromosome, position_1_indexed, prediction = split_cols[0], int(split_cols[1]), float(split_cols[2])
            position_0_indexed = position_1_indexed - 1
            if prediction:
                positives_in_file.append( [chromosome, position_0_indexed, prediction] )
            else:
                negatives_in_file.append( [chromosome, position_0_indexed, prediction] )
    
    if balance:
        positions_in_file = balance_positives_and_negatives(positives_in_file,
                                                            negatives_in_file)
    else:
        positions_in_file = positives_in_file + negatives_in_file

    np.random.shuffle(positions_in_file)

    #print "%d total positions for %s" % (len(positions_in_file), predictions_path)

    #print "Positives: %d Negatives: %d" % (len(positives_in_file), len(negatives_in_file))

    return positions_in_file



def main():
    args = parse_args()
    encoding_name = c.encoding_name
    training_data_folder = os.path.join(c.training_data_folder, encoding_name)

    if os.path.exists(training_data_folder):
        print(("Training data already exists: %s" % training_data_folder))
    else:
        print(("Creating folder for training data: %s" % training_data_folder))
        os.makedirs(training_data_folder)

    training_data_folder = os.path.join(
                                training_data_folder, c.all_data_folder_name)

    if os.path.exists(training_data_folder):
        print(("All data folder already exists: %s" % training_data_folder))
    else:
        print(("Creating folder for training data: %s" % training_data_folder))
        os.makedirs(training_data_folder)

    patient_files = []
    num_nodes = int(args.num_nodes)
    node_no = int(args.node_no)

    if args.goldset_data:
        for goldset_file in c.goldset_files:
            if goldset_file[0] == str(args.goldset_data):
                goldset_sample = goldset_file
                break
        else:
            print(("NO GOLDSET FILE FOUND UNDER NAME: %s" % args.goldset_data))
            return

        if len(goldset_sample) == 4:
            positions = parse_predictions_file(goldset_sample[1],
                                               args.balance_positive_negative)
        elif len(goldset_sample) == 5:
            positions = parse_smurf_file(
                            goldset_sample[1],
                            goldset_sample[4],
                            args.balance_positive_negative)
        else:
            print("ERROR UNKNOWN GOLDSET SAMPLE")
            return

        patient_files.append((goldset_sample[0],
                              positions,
                              goldset_sample[2],
                              goldset_sample[3]))
        
        for patient in patient_files:
            print((len(patient[1])))
            positions_for_this_node = np.array_split(
                                            patient[1], num_nodes)[node_no]
            n_positions = len(positions_for_this_node)
            print(n_positions)
            split_positions = np.array_split(
                                positions_for_this_node,
                                n_positions/c.reg_batch_size + 1
                                )
            print((len(split_positions)))
            training_data_file = os.path.join(
                                        training_data_folder, patient[0])
            Parallel(n_jobs=int(args.num_processes))(
                delayed(generate_images_for_positions)
                    (
                        patient[0], batch, patient[2], patient[3],
                        training_data_file, idx, node_no
                    )
                for idx, batch in enumerate(split_positions)
            )
    else:
        if args.liver_data:
            smurf_preds = c.liver_patients_smurf_predictions
            normal_bam = c.liver_patients_normal_bam_file_path
            tumor_bam = c.liver_patients_tumor_bam_file_path
        elif args.crc_data:
            smurf_preds = c.crc_patients_smurf_predictions
            normal_bam = c.crc_patients_normal_bam_file_path
            tumor_bam = c.crc_patients_tumor_bam_file_path
        elif args.gastric_data:
            smurf_preds = c.gastric_patients_smurf_predictions
            normal_bam = c.gastric_patients_normal_bam_file_path
            tumor_bam = c.gastric_patients_tumor_bam_file_path
        elif args.lung_data:
            smurf_preds = c.lung_patients_smurf_predictions
            normal_bam = c.lung_patients_normal_bam_file_path
            tumor_bam = c.lung_patients_tumor_bam_file_path
        elif args.sarcoma_data:
            smurf_preds = c.sarcoma_patients_smurf_predictions
            normal_bam = c.sarcoma_patients_normal_bam_file_path
            tumor_bam = c.sarcoma_patients_tumor_bam_file_path
        elif args.thyroid_data:
            smurf_preds = c.thyroid_patients_smurf_predictions
            normal_bam = c.thyroid_patients_normal_bam_file_path
            tumor_bam = c.thyroid_patients_tumor_bam_file_path
        elif args.lymphoma_data:
            smurf_preds = c.lymphoma_patients_smurf_predictions
            normal_bam = c.lymphoma_patients_normal_bam_file_path
            tumor_bam = c.lymphoma_patients_tumor_bam_file_path

        csv_files = os.listdir(smurf_preds)
        patients = [patient.replace('.csv', '') for patient in csv_files]
        patients.sort()
        
        print((len(patients)))
        
        patients = np.array_split(patients, num_nodes)[node_no]
        
        total = 0

        for patient in patients:
            filename = os.path.join(smurf_preds, patient + '.csv')
            positions = parse_predictions_file(filename,
                                               args.balance_positive_negative)
            total += len(positions)
            patient_files.append((patient,
                                  positions,
                                  normal_bam % patient,
                                  tumor_bam % patient))
        #print 'Total', total

        if not len(patient_files):
            pass
        else:
            for patient in patient_files:
                split_positions = np.array_split(
                                    patient[1], (len(patient[1])/c.reg_batch_size) + 1)
                training_data_file = os.path.join(
                                            training_data_folder, patient[0])
                Parallel(n_jobs=int(args.num_processes))(
                    delayed(generate_images_for_positions)
                    (
                        patient[0], batch, patient[2], patient[3],
                        training_data_file, idx
                    )
                    for idx, batch in enumerate(split_positions)
                )


if __name__ == '__main__':
    main()

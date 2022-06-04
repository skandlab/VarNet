import pysam
import pandas as pd
import numpy as np
from time import time
from joblib import Parallel, delayed
import snvs.constants as c
import os
import argparse

from snvs.filter import filter_snvs
from indels.filter import filter_indels

def concat_csv_files(filtering_batches_folder):
    combined_positions_file = os.path.join(filtering_batches_folder, c.filtered_positions_file) 
    batch_files = os.listdir(filtering_batches_folder)
    
    # print "Batch files count %s" % str(len(batch_files))

    if os.path.exists(combined_positions_file):
        print(("Combined positions file already exists, please check. %s" % combined_positions_file))
        return

    with open(combined_positions_file, 'a') as f:
        for batch in batch_files:
            with open(os.path.join(filtering_batches_folder, batch)) as r:
                for line in r:
                    line = line.strip()
                    chrom, pos = line.split('\t')[0], line.split('\t')[1]
                    f.write('%s\t%s\n' % (chrom, pos))

    # delete the batches
    for batch in batch_files:
        os.remove(os.path.join(filtering_batches_folder, batch))

    print(("Candidates file: %s" % combined_positions_file))
    return combined_positions_file


def split_chrom(chrom_range, args):
    cpu_count = 10 # split it into a fixed numbers of batches
    
    chunk_length = int((chrom_range[2]+1)/cpu_count)

    if chunk_length == 0:
        return [chrom_range]

    start, end = 0, chunk_length - 1
    new_ranges = [(chrom_range[0], start, end)]
    done=False
    
    while not done:
        start = start + chunk_length
        end = end + chunk_length

        if end >= chrom_range[2]:
            done = True
            end = chrom_range[2]

        new_ranges += [(chrom_range[0], start, end)]

    return new_ranges

def create_folder(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except:
            # folder may have been created in another process at the same time
            pass

def filter_candidates(args, snv_candidates_folder, indel_candidates_folder, bamname_n, bamname_t, regions, batch_num):
    snv_candidates_file = os.path.join(snv_candidates_folder, c.filtered_positions_file)

    if not args.indel: # if not indel_only
        if os.path.exists(snv_candidates_file):
            print(("SNV candidates already generated. Delete folder to re-run:", snv_candidates_folder))
        else:
            filter_snvs(snv_candidates_folder, bamname_n, bamname_t, regions, batch_num)

    indel_candidates_file = os.path.join(indel_candidates_folder, c.filtered_positions_file)

    if not args.snv: # if not snv_only
        if os.path.exists(indel_candidates_file):
            print(("INDEL candidates already generated. Delete folder to re-run:", indel_candidates_folder))
        else:
            output = filter_indels(indel_candidates_folder, bamname_n, bamname_t, regions, batch_num)
        
            if args.path_to_trues:
                outfile = args.path_to_trues + '.stats.npy'
                if os.path.exists(outfile):
                    data=np.load(outfile)
                    TRUE_FREQS, TRUE_MQS = list(data[0]),list(data[1])
                    TRUE_FREQS.extend(output[0])
                    TRUE_MQS.extend(output[1])
                    np.save(outfile, [TRUE_FREQS, TRUE_MQS])
                else:
                    np.save(outfile, output)
    
def parse_bed(bed_file, limit_per_batch=1000000):
    """
    returns a list
    each element in the list is a batch containing regions that don't have overall length more than limit_per_batch
    e.g. batches = [ [(chrom, start, end), (chrom, start, end) ..], [(chrom, start, end), .. ], ...  ]

    the positions in a bed file are 0-indexed and inclusive of the start and exclusive of the end position.
    https://genome.ucsc.edu/FAQ/FAQformat.html#format1
    https://en.wikipedia.org/wiki/BED_(file_format)#Coordinate_system
    """
    import pybedtools

    regions_bed = pybedtools.BedTool(bed_file)
    batches = []
    batch, batch_length = [], 0

    for region in regions_bed:

        chrom, start, end = region.chrom, region.start, region.end-1 # end-1 because end is not inclusive in BED format, and filter_snvs and filter_indels expects inclusive 0-indexed intervals

        if batch_length < limit_per_batch:
            batch_length += region.length
            batch.append((chrom, start, end)) 
        else:
            batches.append(batch)
            batch, batch_length = [], 0 # reset
            batch.append((chrom, start, end))
            batch_length += region.length
    
    if len(batch):
        batches.append(batch)

    return batches

def main(sample_name, bamname_n, bamname_t, args, goldset=True):

    sample_folder = os.path.join(args.output_dir, sample_name)

    create_folder(sample_folder)
    print(("Sample Output: %s\n" % sample_folder))

    candidates_folder = os.path.join(sample_folder, c.sample_candidates_folder)

    create_folder(candidates_folder)
    print(("Candidates Directory: %s\n" % candidates_folder))

    snv_candidates_folder = os.path.join(candidates_folder, c.snv_candidates_folder)
    indel_candidates_folder = os.path.join(candidates_folder, c.indel_candidates_folder)

    create_folder(snv_candidates_folder)
    create_folder(indel_candidates_folder)

    ref_file = pysam.FastaFile(args.reference)

    # 'MT' is mitochondrial DNA, relevant in cancer
    # we list the portions/chromosomes of DNA we want as there are more unmapped sections in v37 of the reference genome
    # chromosomes_positions_range = [] # range of positions in each chromosome e.g ('X', 0, 155270559), ('Y', 0, 59373565)

    if not args.region_bed:
        print("Scanning reference for regions")

        batches = []

        # get the list of chromosomes in the reference genome and append the position range for each chromosome
        for idx, chrom in enumerate(ref_file.references):
            #if chrom == args.chromosome:
            chrom_range = (chrom, 0, ref_file.lengths[idx] - 1 ) # fetch the length of the chromosome from the reference file, subtract by 1 as positions are 0-indexed in pysam
            batches.append(split_chrom(chrom_range, args)) # split_chrom returns [ (chrom, start1, end1), (chrom, start2, end2), ... ]

    else:
        print("Reading bed file:", args.region_bed)
        batches = parse_bed(args.region_bed)


    Parallel(n_jobs=args.processes)(delayed(filter_candidates)(args, snv_candidates_folder, indel_candidates_folder, bamname_n, bamname_t, batches[i], i) for i in range(len(batches)))
    
    if not args.indel: concat_csv_files(snv_candidates_folder)
    if not args.snv: concat_csv_files(indel_candidates_folder)

    print("============== FILTERING REGIONS COMPLETE ===============")

def check_filtering(path_to_trues, sample_name):  
    # checking all trues
    if goldset:
        trues = pd.read_csv(path_to_trues, sep='\t', header=None, names=['chrom', 'pos'], dtype={'chrom': str, 'pos': int})
        trues['pos'] = trues['pos'] - 1 # goldset positions are 1-indexed and pysam is 0-indexed so subtract by 1 here
        print("Fetched Trues")
    else:
        trues = pd.read_csv(path_to_trues, sep='\t', header=None, skiprows=1, names=['chrom', 'pos'], dtype={'chrom': str, 'pos': int}, usecols=[1, 2])
        trues['pos'] = trues['pos'] - 1 # SMURF positions are 1-indexed and pysam is 0-indexed so subtract by 1 here

    potential_trues = pd.read_csv(os.path.join(c.filtering_folder, c.filtering_batches_folder, sample_name, c.filtered_positions_file), sep='\t', header=None, names=['chrom', 'pos'], dtype={'chrom': str, 'pos': int})
    print("Fetched potential trues")
    intersection = pd.merge(potential_trues, trues, on=['chrom', 'pos'], how='inner')
    print("intersection done")

    print(("Sample Name %s" % sample_name))     
    print(("Number of filtered positions: %s" % potential_trues.shape[0]))
    print(("True positives selected: %s", intersection.shape[0]))
    print(("Number of true positives: %s", trues.shape[0]))

def parse_args():
    parser = argparse.ArgumentParser(description="Filter positions")
    parser.add_argument('--processes', default=5, type=int)
    parser.add_argument('--sample_name', required=True)
    parser.add_argument('--path_to_trues', default=None)
    parser.add_argument('--normal_bam', required=True)
    parser.add_argument('--tumor_bam', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--region_bed', required=False, default=None)
    parser.add_argument('-snv', action='store_true') # read as snv_only
    parser.add_argument('-indel', action='store_true') # read as indel_only

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    if args.path_to_trues:
        ### TRUE INDELS
        true_positions = {}

        with open(args.path_to_trues) as f:
            for line in f:
                line=line.strip()
                chrom, pos = line.split()[0], line.split()[1]
                key='chrom%spos%d' % (chrom, int(pos)) # 1-index
                true_positions[key] = True
    
        filter.true_positions = true_positions
        filter.TRUE_FREQS = []
        filter.TRUE_MQS = []
        ####    

    goldset = True
    filtering_details_file = c.filtering_details_file
    
    # if not os.path.exists(c.filtering_root_folder):
    #     os.makedirs(c.filtering_root_folder)
    
    # if not os.path.exists(c.filtering_folder):
    #     os.makedirs(c.filtering_folder)

    # if not os.path.exists(os.path.join(c.filtering_folder, c.filtering_batches_folder)):
    #     os.makedirs(os.path.join(c.filtering_folder, c.filtering_batches_folder))

    # with open(os.path.join(c.filtering_folder, filtering_details_file), 'w') as f:
    #     f.write('goldset pre-filters from paper\n')

    """ 
    for g in c.dream_challenge_files_on_nscc:# c.goldset_files_on_nscc:
        sample_name, path_to_trues, normal_bamfile, tumor_bamfile = g[0], g[1], g[2], g[3]
        main(sample_name, path_to_trues, normal_bamfile, tumor_bamfile, args, goldset)
    """
    main(args.sample_name, args.normal_bam, args.tumor_bam, args, goldset)
    # check_filtering(args.path_to_trues, args.sample)
    

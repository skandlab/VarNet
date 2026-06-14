import pysam
import pandas as pd
import numpy as np
from time import time
from joblib import Parallel, delayed
import snvs.constants as c
import os
import argparse
import sys
import gc
import ctypes

from utils import load_npz
from snvs.filter import filter_snvs, ffpe_filter_snvs
from indels.filter import filter_indels, ffpe_filter_indels

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
                    f.write(line)

    # delete the batches
    for batch in batch_files:
        os.remove(os.path.join(filtering_batches_folder, batch))

    print(("Candidates file: %s" % combined_positions_file))
    return combined_positions_file

def create_folder(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except:
            # folder may have been created in another process at the same time
            pass

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

def dbSNP_filtering(path_to_candidates, dbSNP_variants_0_indexed, whitelist_vcf):
    """
    filter any somatic variant candidate overlapping any dbSNP COMMON variant
    dbSNP_variants_0_indexed
    """
    candidates_file = open(path_to_candidates)
    filtered_candidates_count = 0
    filtered_candidates_file = path_to_candidates.replace('.csv', '.dbSNP.filtered.csv')
    out = open(filtered_candidates_file, 'w')
    
    whitelist_sites = {}
    if whitelist_vcf:
        if whitelist_vcf.endswith('.gz'):
            import gzip
            f = gzip.open(whitelist_vcf)
        else:
            f = open(whitelist_vcf)

        whitelist_sites = {}

        for line in f:
            line = line.decode("utf-8") # convert bytes to string
            line = line.strip()
            if line.startswith('#'): continue
            CHROM, POS = line.split()[0], int(line.split()[1])-1 # convert to 0-indexed

            whitelist_sites[f'CHROM{CHROM}POS{POS}'] = True

        f.close()

    for line in candidates_file:
        s=line.strip().split()
        CHROM, POS, REF, ALT = s[0], int(s[1]), s[2], s[3] # position is 0-indexed in candidates file

        # Note: To support GRch37, will need to map chromosome names from the fasta file version (e.g. 1, 2, 3, GL000231.1, MT) to (chr1, chr2, chrUn_gl000231, chrM) to match what the dbSNP dictionary is using

        # pos_key = '%spos%s' % (CHROM, POS) # for build 155
        pos_key = f'CHROM{CHROM}POS{POS}' # 0-indexed pos, for build 156
        # pos_key = f'CHROM{CHROM}POS{POS}REF{REF}ALT{ALT}' # 0-indexed pos, for build 156
        
        if pos_key in dbSNP_variants_0_indexed and pos_key not in whitelist_sites:
            filtered_candidates_count += 1
        else:
            out.write(line)

    out.close()
    os.remove(path_to_candidates)
    os.rename(filtered_candidates_file, path_to_candidates)

    print('dbSNP filtered variants count:', filtered_candidates_count)
    print('Saved dbSNP filtered variant candidates:', path_to_candidates)

def load_pon(filename='1000g_pon.hg38.vcf.gz'):
    import gzip
    x=gzip.open('1000g_pon.hg38.vcf.gz', 'rt')
    pon={}
    for line in x:
        if line[0]=='#': continue
        chrom,pos=line.split()[0], line.split()[1] # 1-indexed pos from vcf
        pon[f'{chrom}pos{pos}']=True
    x.close()
    return pon

def pon_filtering(path_to_candidates, pon):
    """
    filter any somatic variant candidate overlapping any pon site
    """
    candidates_file = open(path_to_candidates)
    filtered_candidates_count = 0
    filtered_candidates_file = path_to_candidates.replace('.csv', '.pon.filtered.csv')
    
    out = open(filtered_candidates_file, 'w')
    
    for line in candidates_file:
        s=line.strip().split()
        chrom, pos, REF, ALT = s[0], int(s[1]), s[2], s[3] # position is 0-indexed in candidates file
    
        pos_key = f'{chrom}pos{pos+1}' # convert to 1-indexed pos
        
        if pos_key in pon:
            filtered_candidates_count += 1
        else:
            out.write(line) # write 0-indexed position back to file
        
    out.close()
    os.remove(path_to_candidates)
    os.rename(filtered_candidates_file, path_to_candidates)

    print('PoN filtered variants count:', filtered_candidates_count)
    print('Saved PoN filtered variant candidates:', path_to_candidates)

def load_npz_part_single(filename, part_name):
    """Loads a specific part from the single NPZ file."""
    data = np.load(filename, allow_pickle=True)
    return data[part_name].item()  # Convert back to a dictionary

def gnomAD_filtering(path_to_candidates, gnomAD_variants_1_indexed, whitelist_vcf):
    """
    filter any somatic variant candidate overlapping any gnomAD PASS variant AF > 0
    """
    candidates_file = open(path_to_candidates)
    filtered_candidates_count = 0
    filtered_candidates_file = path_to_candidates.replace('.csv', '.gnomAD.filtered.csv')
    
    out = open(filtered_candidates_file, 'w')

    whitelist_sites = {}
    if whitelist_vcf:
        if whitelist_vcf.endswith('.gz'):
            import gzip
            f = gzip.open(whitelist_vcf)
        else:
            f = open(whitelist_vcf)

        whitelist_sites = {}

        for line in f:
            line = line.decode("utf-8") # convert bytes to string
            line = line.strip()
            if line.startswith('#'): continue
            CHROM, POS, REF, ALT = line.split()[0], int(line.split()[1]), line.split()[3], line.split()[4] # vcf POS is 1-indexed

            whitelist_sites[f'CHROM{CHROM}POS{POS}REF{REF}ALT{ALT}'] = True

        f.close()

    for line in candidates_file:
        s=line.strip().split()
        chrom, pos, REF, ALT = s[0], int(s[1]), s[2], s[3] # position is 0-indexed in candidates file

        # pos_key = 'CHROM%sPOS%d' % (chrom, pos+1) # convert to 1-indexed pos for gnomAD dict
        pos_key = 'CHROM%sPOS%dREF%sALT%s' % (chrom, pos+1, REF, ALT) # convert to 1-indexed pos for gnomAD dict

        if pos_key in gnomAD_variants_1_indexed and pos_key not in whitelist_sites:
            filtered_candidates_count += 1
        else:
            out.write(line) # write 0-indexed position back to file

    out.close()
    os.remove(path_to_candidates)
    os.rename(filtered_candidates_file, path_to_candidates)

    #print('gnomAD filtered variants count:', filtered_candidates_count)
    #print('Saved gnomAD filtered variant candidates:', path_to_candidates)

    return filtered_candidates_count

def parse_germline_vcf(path):
    """
    returns dict of germline variant positions 0-indexed
    """
    if path.endswith('.gz'):
        import gzip
        f = gzip.open(path)
    else:
        f = open(path)
    
    germline_sites = {}

    for line in f:
        line = line.decode("utf-8") # convert bytes to string
        line = line.strip()
        if line.startswith('#'): continue 
        chrom, pos = line.split()[0], int(line.split()[1])-1 # convert to 0-indexed

        germline_sites['%spos%d' % (chrom, pos)] = True

    f.close()
    return germline_sites

def germline_vcf_filter(path_to_candidates, germline_sites):
    """
    filter any somatic variant candidate overlapping any germline variant
    """
    candidates_file = open(path_to_candidates)
    filtered_candidates_count = 0
    filtered_candidates_file = path_to_candidates.replace('.csv', '.germline.filtered.csv')

    if os.path.exists(filtered_candidates_file): os.remove(filtered_candidates_file)
    
    out = open(filtered_candidates_file, 'a')
    
    for line in candidates_file:
        line=line.strip()
        chrom, pos = line.split()[0], line.split()[1] # position is 0-indexed in candidates file

        pos_key = '%spos%s' % (chrom, pos)
        
        if pos_key in germline_sites:
            filtered_candidates_count += 1
        else:
            out.write('%s\t%s\n' % (chrom, pos))

    out.close()
    os.remove(path_to_candidates)
    os.rename(filtered_candidates_file, path_to_candidates)

    print('Number of filtered candidates:', filtered_candidates_count)
    print('Saved germline-filtered variant candidates:', path_to_candidates)

def filter_candidates(args, snv_candidates_folder, indel_candidates_folder, bamname_n, bamname_t, regions, batch_num):
    snv_candidates_file = os.path.join(snv_candidates_folder, c.filtered_positions_file)

    # need to re-define the ref_file here since it cannot be pickled by joblib in Parallel() call
    ref_file = pysam.FastaFile(args.reference)

    try:
        if not args.indel: # if not indel_only
            if os.path.exists(snv_candidates_file):
                pass#print(("SNV candidates already generated. Delete this folder to re-generate candidates:", snv_candidates_folder))
            else:
                filter_snvs(snv_candidates_folder, bamname_n, bamname_t, ref_file, regions, batch_num)

        indel_candidates_file = os.path.join(indel_candidates_folder, c.filtered_positions_file)

        if not args.snv: # if not snv_only
            if os.path.exists(indel_candidates_file):
                pass#print(("INDEL candidates already generated. Delete this folder to re-generate candidates:", indel_candidates_folder))
            else:
                output = filter_indels(indel_candidates_folder, bamname_n, bamname_t, ref_file, regions, batch_num)
    finally:
        ref_file.close()
        
def main(sample_name, bamname_n, bamname_t, args, goldset=True):

    sample_folder = os.path.join(args.output_dir, sample_name)

    output_vcf = os.path.join(sample_folder, sample_name + '.vcf')
    if (os.path.exists(output_vcf) or os.path.exists(output_vcf + '.gz')):
        print("VCF file exists for sample. Use new output_dir to re-run sample or delete the VCF to re-generate in current output dir.")
        print(("VCF:", output_vcf))
        return

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

    # close the reference handle used for chromosome enumeration
    # (each filter_candidates worker opens its own handle)
    ref_file.close()


    Parallel(n_jobs=args.processes)(delayed(filter_candidates)(args, snv_candidates_folder, indel_candidates_folder, bamname_n, bamname_t, batches[i], i) for i in range(len(batches)))
    
    gc.collect()
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)
    
    if not args.indel: concat_csv_files(snv_candidates_folder)
    if not args.snv: concat_csv_files(indel_candidates_folder)

    if args.germline_vcf:
        print('Filtering provided germline variants: %s' % args.germline_vcf)
        germline_sites = parse_germline_vcf(args.germline_vcf)
        
        snv_candidates_file = os.path.join(snv_candidates_folder, c.filtered_positions_file)
        indel_candidates_file = os.path.join(indel_candidates_folder, c.filtered_positions_file)

        if not args.indel: germline_vcf_filter(snv_candidates_file, germline_sites)
        if not args.snv: germline_vcf_filter(indel_candidates_file, germline_sites)

        del germline_sites
        
    # gnomAD-dbSNP filtering for tumor-only mode
    if (not args.no_gnomad_dbsnp and not bamname_n) or args.gnomad_dbsnp:
        # tumor-only mode, fetch dbSNP variant sites
        if args.reference_build == 'grch37':
            # Note: Grch37 currently not supported. To support GRch37, need to map chromosome names from the fasta file version (e.g. 1, 2, 3, GL000231.1, MT) to (chr1, chr2, chrUn_gl000231, chrM) to match what the dbSNP dictionary is using
            dbSNP_variants_0_indexed = load_npz('GrCh37-dbSNP-common-variants-0-indexed.npz') # build 155
        elif args.reference_build == 'grch38':
            # dbSNP_variants_0_indexed = load_npz('GrCh38-dbSNP-common-variants-0-indexed.npz') # build 155, {CHROM}pos{POS}
            dbSNP_variants_0_indexed = load_npz('GrCh38-dbSNP-build-156-common-variants-0-indexed-CHROMPOS.npz') # build 156, CHROM{CHROM}POS{POS}
            # dbSNP_variants_0_indexed = load_npz('GrCh38-dbSNP-build-156-common-variants-0-indexed.npz') # build 156, CHROM{CHROM}POS{POS}REF{REF}ALT{ALT}
        else:
            raise Exception('unrecognized reference build:', args.reference_build)

        snv_candidates_file = os.path.join(snv_candidates_folder, c.filtered_positions_file)
        indel_candidates_file = os.path.join(indel_candidates_folder, c.filtered_positions_file)

        if not args.indel: dbSNP_filtering(snv_candidates_file, dbSNP_variants_0_indexed, args.whitelist_vcf)
        if not args.snv: dbSNP_filtering(indel_candidates_file, dbSNP_variants_0_indexed, args.whitelist_vcf)

        del dbSNP_variants_0_indexed # delete to free up memory

        # load gnomAD 
        # gnomAD_variants_1_indexed = load_npz('combined_gnomAD_v4.1_genomes_exomes_dict.hg38.CHROMPOS.npz') # CHROMPOS: True
        # gnomAD_variants_1_indexed = load_npz('combined_gnomAD_v4.1_genomes_exomes_dict.hg38.with_AF.npz') # CHROMPOSREFALT: AF, needs 100GB memory to load
        #gnomAD_variants_1_indexed = load_npz('combined_gnomAD_v4.1_genomes_exomes_dict.hg38.npz') # CHROMPOSREFALT: True
       
        # load gnomAD in chunks to lower memory need
        filtered_candidates_count = 0
        for part in list(range(1,21)): # [1,...,20]
            gnomAD_variants_1_indexed = load_npz_part_single("combined_gnomAD_v4.1_genomes_exomes_dict.hg38.chunks.npz", f"part_{part}") # CHROMPOSREFALT: True
        
            if not args.indel: filtered_candidates_count += gnomAD_filtering(snv_candidates_file, gnomAD_variants_1_indexed, args.whitelist_vcf)
            if not args.snv: filtered_candidates_count += gnomAD_filtering(indel_candidates_file, gnomAD_variants_1_indexed, args.whitelist_vcf)

            del gnomAD_variants_1_indexed # delete to free up memory
        
        print('gnomAD filtered variants count:', filtered_candidates_count)

        ## load 1000genomes PoN
        #pon = load_pon('1000g_pon.hg38.vcf.gz')
        #pon_filtering(snv_candidates_file, pon)
        #pon_filtering(indel_candidates_file, pon)
        #del pon

    if args.ffpe:
        print('===> ADDITIONAL FFPE FILTERING <===')
        snv_candidates_file = os.path.join(snv_candidates_folder, c.filtered_positions_file)
        indel_candidates_file = os.path.join(indel_candidates_folder, c.filtered_positions_file)
        
        if not args.indel:
            # do SNV FFPE filtering
            ffpe_filter_snvs(bamname_n, bamname_t, snv_candidates_file)       
        
        if not args.snv:
            # indel FFPE filtering
            ffpe_filter_indels(bamname_n, bamname_t, indel_candidates_file)
 
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
    parser.add_argument('--normal_bam', required=False, default=None)
    parser.add_argument('--tumor_bam', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--reference_build', required=False, default='grch38', choices=['grch37', 'grch38'])
    parser.add_argument('--region_bed', required=False, default=None)
    parser.add_argument('-snv', action='store_true') # read as snv_only
    parser.add_argument('-indel', action='store_true') # read as indel_only
    parser.add_argument('-ffpe', action='store_true') # ffpe sample
    parser.add_argument('--germline_vcf', required=False, type=str, default=None)
    parser.add_argument('--gnomad_dbsnp', action='store_true', help='do germline filtering using gnomad and dbsnp')
    parser.add_argument('--no_gnomad_dbsnp', action='store_true', help='dont do germline filtering using gnomad and dbsnp')
    parser.add_argument('--whitelist_vcf', required=False, type=str, default=None, help='list of variants to whitelist during germline filtering using gnomAD/dbSNP (for tumor-only mode)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # if not args.normal_bam:
    #     if not args.reference_build:
    #         print('Please provide --reference_build ("grch37", "grch38") argument in tumor-only mode')
    #         sys.exit()

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
    

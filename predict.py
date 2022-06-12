import pysam
import numpy as np
import pandas as pd
import argparse
from time import time
import joblib
from joblib import Parallel, delayed, __version__
import os
import sys

# Disable tf logging. 1 to filter out INFO logs, 2 to additionally filter out WARNING logs,
# and 3 to additionally filter out ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from filter import create_folder
from utils import get_ref_file

import snvs.constants as c
from snvs.generate_training_data import get_ref_base
from snvs.filter import check_read
from snvs.predict import predict_snvs
from indels.predict import predict_indels

def concatenate_batch_prediction_results(predictions_folder):
    prediction_results_file = os.path.join(predictions_folder, c.combined_predictions_file)

    if os.path.exists(prediction_results_file):
        print(("Predictions file already exists: %s" % prediction_results_file))
        return

    from os import listdir
    from os.path import isfile, join

    batch_prediction_files = [join(predictions_folder, f) for f in listdir(predictions_folder) if isfile(join(predictions_folder, f))]

    for batch in batch_prediction_files:
        p = pd.read_csv(batch, sep='\t', header=None, names=['chrom', 'pos', 'pred_true'])
        p = p.drop_duplicates(subset=['chrom', 'pos'])
        p.to_csv(prediction_results_file, sep='\t', index=False, encoding='utf-8', mode='a', header=False)

    for f in batch_prediction_files:
       os.remove(f)

def make_vcf(sample_folder, snv_predictions_file, indel_predictions_file, args):
    from datetime import datetime

    ref_file = get_ref_file(args.reference)

    output_vcf = os.path.join(sample_folder, args.sample_name + '.vcf')

    if os.path.exists(output_vcf):
        print("VCF file exists for sample. Delete the VCF to re-generate in current output dir.")
        print(("VCF:", output_vcf))
        return

    output_vcf = output_vcf.replace('.vcf', '.vcf.temp')

    if os.path.exists(output_vcf):
        # temp file exists, delete it
        os.remove(output_vcf)

    vcf_write = open(output_vcf, 'a')

    fileDate = datetime.now().strftime("%Y%B%d, %H:%M:%S")

    vcf_header =  "##fileformat=VCFv4.2\n" + \
"""##fileDate=%s
##source=VarNet v%s
##reference=%s
##normalBAM=%s
##tumorBAM=%s
##INFO=<ID=TYPE,Number=.,Type=String,Description="Type of Somatic Event INDEL or SNV">
##INFO=<ID=SCORE,Number=1,Type=Float,Description="Prediction probability score">
##FILTER=<ID=PASS,Description="Accept as somatic mutation with probability score at least 0.5">
##FILTER=<ID=REJECT,Description="Reject somatic mutation with probability score value below 0.5">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth in the tumor">
##FORMAT=<ID=RO,Number=1,Type=Integer,Description="Reference allele observation count in the tumor">
##FORMAT=<ID=AO,Number=A,Type=Integer,Description="Alternate allele observation count in the tumor">
##FORMAT=<ID=AF,Number=1,Type=Float,Description="Allele fractions of alternate alleles in the tumor">
#CHROM  POS ID  REF ALT QUAL    FILTER  INFO    FORMAT  SAMPLE\n""" % (fileDate, c.__VERSION__, args.reference, args.normal_bam, args.tumor_bam)

    vcf_write.write(vcf_header)

    ALLELES = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    ALLELE_INDICES = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    CUT_OFF = 0.3

    bamfile_n = pysam.AlignmentFile(args.normal_bam, "rb", check_sq=False)
    bamfile_t = pysam.AlignmentFile(args.tumor_bam, "rb", check_sq=False)

    def germline_filter(chrom, ref_pos, bamfile_n, ref_file):
        # return True if germlines variant (snp or indel) found in the neighboring 50bp region
        # search to the left of the site (and 1bp to the right), in order to check if there is an indel that overlaps the somatic site but begins in a prior position
        margin = 1 # 1bp. reject a somatic site if there is a germline variant within this margin. 1bp default is based on bedtools subtract behavior
        window = 10

        start, end = ref_pos + margin, ref_pos-window
        if end<0: end = 0 # sanity check

        check_sites = range(start, end, -1) # [ref_pos + 1, ref_pos, ref_pos-1, ref_pos-2, ..., ref_pos-49]

        # filter site if there exists a germline variant with AF > 0.1 AND it overlaps with the somatic site of interest
        for site in check_sites:
            snp=get_snv(chrom, site, bamfile_n, ref_file)
            indel=get_indels(chrom, site, bamfile_n, ref_file)

            snp_AF, indel_AF = snp[-1], indel[-1]
            max_AF = max([snp_AF, indel_AF])

            GERMLINE_FILTER = 0.1

            if max_AF > GERMLINE_FILTER:
                # active germline variant, now check if it overlaps the somatic site
                if abs(site-ref_pos) <= margin:
                    return True # site is close or equal to ref_pos

                # if it is an indel on the left of ref_pos, check if it overlaps the somatic site (ref_pos)
                elif site<ref_pos and indel_AF>snp_AF:
                    indel_length = abs(len(indel[0])-len(indel[1])) # length of insertion or deletion. diff between ref and alt sequence lengths
                    right_end = site + indel_length # right end of indel

                    # if the indel overlaps the somatic site within margin, reject somatic site
                    if (right_end + margin >= ref_pos): 
                        return True

            else:
                # no active germline variant (AF>0.1) at this site, move on to next site
                continue

        return False

    # finds most frequent element in a list
    def most_frequent(List):
        return max(set(List), key = List.count)

    def get_indels(chrom, ref_pos, bamfile, ref_file):
        DEPTH, REFERENCE_ALLELE_COUNT, ALT_ALLELE_READ_COUNT = 0, 0, 0

        reads = bamfile.fetch(chrom, ref_pos, ref_pos+1)
        indels = []

        for read in reads: # extract the insertion/deletion in read at ref_pos
            DEPTH += 1

            insertion, deletion_length = '', 0

            # this looks for an indel anywhere in the read. there may not be an indel at the position of interest
            # for indels, the ref pos is the position before the start of indel
            if read.cigarstring is None or not ('I' in read.cigarstring or 'D' in read.cigarstring):
                REFERENCE_ALLELE_COUNT += 1
                continue

            aligned_pairs = read.get_aligned_pairs() # [ (0, ref_pos), (1, ref_pos+1) .. (4, None), (None, ref_pos + 5) .. ]

            past_ref_pos = False

            for p in aligned_pairs:
                if p[1] == ref_pos: # position right before insertion or deletion
                    past_ref_pos = True
                    continue

                if past_ref_pos:
                    # read pos is None, i.e. deletion
                    if p[0] is None:
                        deletion_length += 1

                    # ref pos is none, insertion
                    elif p[1] is None:
                        inserted_base = read.query_sequence[p[0]]
                        insertion += inserted_base

                    else:
                        # if there is no indel right after ref pos, no evidence for indel in this read
                        # so increment ref allele
                        if p[1] == ref_pos+1:
                            REFERENCE_ALLELE_COUNT += 1

                        # stop parsing this read
                        break

            if len(insertion) > 0:
                indels.append(insertion)

            if deletion_length > 0:
                indels.append(deletion_length)

        reference_allele, alt_allele = '.', '.'
        ALT_ALLELE_FRACTION = 0

        if len(indels) == 0:
            # no indels found in position, weird
            return (reference_allele, alt_allele, DEPTH, REFERENCE_ALLELE_COUNT, ALT_ALLELE_READ_COUNT, ALT_ALLELE_FRACTION)

        most_frequent_indel = most_frequent(indels) # return the most frequent element (insertion 'str' or deletion 'int')
        ALT_ALLELE_READ_COUNT = indels.count(most_frequent_indel)

        if type(most_frequent_indel) is int:
            # deletion length
            reference_allele = ref_file.fetch(chrom, ref_pos, ref_pos + most_frequent_indel + 1).upper() # pos, del1, del2, del3
            alt_allele = reference_allele[0] # if reference is 'ATCG', 'A' is the alt allele since this is deletion

        elif type(most_frequent_indel) is str:
            # insertion
            reference_allele = ref_file.fetch(chrom, ref_pos, ref_pos + 1).upper() # e.g. 'A'
            alt_allele = reference_allele + most_frequent_indel # e.g. 'A' + 'TCT', where 'TCT' is insertion

        if DEPTH > 0:
            ALT_ALLELE_FRACTION = round(float(ALT_ALLELE_READ_COUNT)/DEPTH, 4)

        return (reference_allele, alt_allele, DEPTH, REFERENCE_ALLELE_COUNT, ALT_ALLELE_READ_COUNT, ALT_ALLELE_FRACTION)


    def parse_indel_predictions(f):
        with open(f) as r:
            for line in r:
                line = line.strip()
                s=line.split('\t')
                chrom, pos, pred_true = s[0], int(s[1]), round(float(s[2]), 4)

                if pred_true < CUT_OFF:
                    continue

                FILTER = 'PASS' if pred_true >= 0.5 else 'REJECT'

                REFERENCE_ALLELE, ALT_ALLELE, TUMOR_DEPTH, REFERENCE_ALLELE_COUNT_IN_TUMOR, ALT_ALLELE_READ_COUNT_IN_TUMOR, ALT_ALLELE_FRACTION_IN_TUMOR = get_indels(chrom, pos, bamfile_t, ref_file)
                #REFERENCE_ALLELE, ALT_ALLELE, TUMOR_DEPTH, REFERENCE_ALLELE_COUNT_IN_TUMOR, ALT_ALLELE_READ_COUNT_IN_TUMOR = '.','.',0,0,0

                # filter indels with AF<0.03, since AF filtering is not done in indel pre-filtering (indels/filter.py)
                # snv already has AF filter (3.5%) in snvs/filter.py
                if ALT_ALLELE_FRACTION_IN_TUMOR < 0.03:
                    continue

                if c.GERMLINE_FILTER and germline_filter(chrom, pos, bamfile_n, ref_file):
                    continue # overlapping germline variant identified

                POSITION_1_INDEXED = pos + 1

                INFO = 'TYPE=INDEL;SCORE=%s;DP=%d;RO=%d;AO=%d;AF=%s;' % \
                (str(pred_true), TUMOR_DEPTH, REFERENCE_ALLELE_COUNT_IN_TUMOR, ALT_ALLELE_READ_COUNT_IN_TUMOR, str(ALT_ALLELE_FRACTION_IN_TUMOR))

                FORMAT = 'GT:DP:RO:AO:AF'

                SAMPLE = '0/1:%d:%d:%d:%s' % (TUMOR_DEPTH, REFERENCE_ALLELE_COUNT_IN_TUMOR, ALT_ALLELE_READ_COUNT_IN_TUMOR, str(ALT_ALLELE_FRACTION_IN_TUMOR))

                OUT = (chrom, POSITION_1_INDEXED, '.', REFERENCE_ALLELE, ALT_ALLELE, '.', FILTER, INFO, FORMAT, SAMPLE)

                out_string = ''
                for i in OUT:
                    out_string += str(i) + '\t'
                out_string += '\n'

                vcf_write.write(out_string)

    def get_snv(chrom, ref_pos, bamfile, ref_file):
        coverage = bamfile.count_coverage(chrom, ref_pos, ref_pos+1)#, quality_threshold=c.MIN_BASE_QUALITY, read_callback=check_read)

        # [ (#A, #C, #G, #T) ] at each position in tumor
        coverage_list = [(coverage[0][i], coverage[1][i], coverage[2][i], coverage[3][i])
            for i in range(len(coverage[0]))]

        assert len(coverage_list) == 1
        coverage_list = coverage_list[0] # A C G T
        coverage_list = list(coverage_list) # convert tuple to list

        REFERENCE_ALLELE = get_ref_base(ref_pos, chrom, ref_file)

        try:
            REFERENCE_ALLELE_COUNT = coverage_list[ALLELE_INDICES[REFERENCE_ALLELE]]
        except KeyError:
            # if the reference allele is ambiguous, 'N'
            REFERENCE_ALLELE_COUNT = 0

        DEPTH = sum(coverage_list)

        if REFERENCE_ALLELE in ALLELE_INDICES: # can be 'N'
            # find max count allele in tumor that is not reference allele, maybe an issue if there are two max count alleles
            coverage_list_exclude_ref = coverage_list.copy()
            coverage_list_exclude_ref.pop(ALLELE_INDICES[REFERENCE_ALLELE]) # remove ref allele
            
            ALT_ALLELE = ALLELES[coverage_list.index(max(coverage_list_exclude_ref))] # get index in original coverage_list of the max alt allele
            ALT_ALLELE_READ_COUNT = max(coverage_list_exclude_ref)

            if DEPTH > 0:
                ALT_ALLELE_FRACTION = round(float(ALT_ALLELE_READ_COUNT)/float(DEPTH), 4)
            else:
                ALT_ALLELE_FRACTION = 0
        else:
            # ref allele is 'N' or something not ACGT, can't figure out ALT allele
            ALT_ALLELE = 'N'
            ALT_ALLELE_READ_COUNT, ALT_ALLELE_FRACTION = 0,0

        return (REFERENCE_ALLELE, ALT_ALLELE, DEPTH, REFERENCE_ALLELE_COUNT, ALT_ALLELE_READ_COUNT, ALT_ALLELE_FRACTION)


    def parse_snv_predictions(f):
        with open(f) as r:
            for line in r:
                line = line.strip()
                s=line.split('\t')
                chrom, pos, pred_true = s[0], int(s[1]), round(float(s[2]), 4)

                if pred_true < CUT_OFF:
                    continue

                REFERENCE_ALLELE, ALT_ALLELE, TUMOR_DEPTH, REFERENCE_ALLELE_COUNT_IN_TUMOR, ALT_ALLELE_READ_COUNT_IN_TUMOR, ALT_ALLELE_FRACTION_IN_TUMOR = get_snv(chrom, pos, bamfile_t, ref_file)

                if c.GERMLINE_FILTER and germline_filter(chrom, pos, bamfile_n, ref_file):
                    continue # overlapping germline variant identified

                POSITION_1_INDEXED = pos + 1
                FILTER = 'PASS' if pred_true >= 0.5 else 'REJECT'
                INFO = 'TYPE=SNV;SCORE=%s;DP=%d;RO=%d;AO=%d;AF=%s;' % \
                (str(pred_true), TUMOR_DEPTH, REFERENCE_ALLELE_COUNT_IN_TUMOR, ALT_ALLELE_READ_COUNT_IN_TUMOR, str(ALT_ALLELE_FRACTION_IN_TUMOR))
                FORMAT = 'GT:DP:RO:AO:AF'
                SAMPLE = '0/1:%d:%d:%d:%s' % (TUMOR_DEPTH, REFERENCE_ALLELE_COUNT_IN_TUMOR, ALT_ALLELE_READ_COUNT_IN_TUMOR, str(ALT_ALLELE_FRACTION_IN_TUMOR))

                OUT = (chrom, POSITION_1_INDEXED, '.', REFERENCE_ALLELE, ALT_ALLELE, '.', FILTER, INFO, FORMAT, SAMPLE)

                out_string = ''
                for i in OUT:
                    out_string += str(i) + '\t'
                out_string += '\n'

                vcf_write.write(out_string)

    if not args.indel: # if not indel only
        parse_snv_predictions(snv_predictions_file)

    if not args.snv: # if not snv only
        parse_indel_predictions(indel_predictions_file)

    vcf_write.close()

    os.rename(output_vcf, output_vcf.replace('.vcf.temp', '.vcf'))
    output_vcf = output_vcf.replace('.vcf.temp', '.vcf')

    print(("Output VCF:", output_vcf))

    # 1 index pos
    # make pred_true 4 decimal places
    # fetch ref sequence and alt sequence
    # Sort positions by chromosome number
    # cut off predictions < .10

    # ID is ., QUAL is .
    # FILTER is PASS, REJECT, LowQual
    # INFO IS: SCORE=0.4583;DP=96;RO=93;AO=3;AF=0.0312;
    # FORMAT IS: GT:DP:RO:AO:AF
    # SAMPLE IS: 0/1:96:93:3:0.0312

def check_batches_complete(predictions_folder, candidates_path):
    # checks if the predictions folder has preds for all candidates
    pred_files = os.listdir(predictions_folder)
    candidates = pd.read_csv(candidates_path, sep='\t', header=None, names=['chrom', 'pos'], dtype={'chrom': str, 'pos': int})
    num_candidates = candidates.shape[0]

    num_predicted_positions = 0
    for f in pred_files:
        pred_file = pd.read_csv(os.path.join(predictions_folder, f), sep='\t', header=None, names=['chrom', 'pos', 'pred_true'], dtype={'chrom': str, 'pos': int, 'pred_true': float})
        num_predicted_positions += pred_file.shape[0]

    print(("Num preds", num_predicted_positions))
    print(("Candidates", num_candidates))

    if num_predicted_positions < num_candidates:
        print("Incomplete")
        return False
    else:
        return True

def main():
    sample_folder = os.path.join(args.output_dir, args.sample_name)
    create_folder(sample_folder)

    output_vcf = os.path.join(sample_folder, args.sample_name + '.vcf')
    if os.path.exists(output_vcf):
        print("VCF file exists for sample. Use new output_dir to re-run sample or delete the VCF to re-generate in current output dir.")
        print(("VCF:", output_vcf))
        return

    # split into 100 batches
    split_num = 100

    predictions_folder = os.path.join(sample_folder, c.sample_predictions_folder)
    create_folder(predictions_folder)

    snv_predictions_file, indel_predictions_file = None, None

    if not args.indel: # do snv
        snv_predictions_folder = os.path.join(predictions_folder, c.snv_candidates_folder)
        create_folder(snv_predictions_folder)

        """ CHECK IF Predictions.csv already exists for SNV and INDEL"""
        snv_predictions_file = os.path.join(snv_predictions_folder, c.combined_predictions_file)

        if os.path.exists(snv_predictions_file):
            print(("SNV predictions generated. Delete folder if you wish to re-run:", snv_predictions_folder))
            concatenate_batch_prediction_results(snv_predictions_folder)
        else:
            snv_candidates_path = os.path.join(sample_folder, c.sample_candidates_folder, c.snv_candidates_folder, c.filtered_positions_file)

            if not os.path.exists(snv_candidates_path):
                print("SNV Candidate positions missing. Please run the filter script before prediction.")
                return

            snv_candidates = pd.read_csv(snv_candidates_path, sep='\t', header=None, names=['chrom', 'pos'], dtype={'chrom': str, 'pos': int})

            # Sort the labels file by position and chromosome and then reindex
            snv_candidates = snv_candidates.sort_values(['pos'], ascending=[True]).reset_index(drop=True)

            print(("Number of SNV candidates: ", len(snv_candidates)))

            snv_candidate_batches = np.array_split(snv_candidates, split_num)
            # remove empty batches
            snv_candidate_batches = [_ for _ in snv_candidate_batches if len(_)]

            try:
                Parallel(n_jobs=int(args.processes))( delayed(predict_snvs)(batch, idx, args, snv_predictions_folder) for idx, batch in enumerate(snv_candidate_batches) )
            except joblib.my_exceptions.WorkerInterrupt as e:
                print(('workerinterrupt', e))

            concatenate_batch_prediction_results(snv_predictions_folder)

    if not args.snv: # do indels
        indel_predictions_folder = os.path.join(predictions_folder, c.indel_candidates_folder)
        create_folder(indel_predictions_folder)

        indel_predictions_file = os.path.join(indel_predictions_folder, c.combined_predictions_file)

        if os.path.exists(indel_predictions_file):
            print(("INDEL predictions generated. Delete folder if you wish to re-run:", indel_predictions_folder))
            concatenate_batch_prediction_results(indel_predictions_folder)

        else:
            indel_candidates_path = os.path.join(sample_folder, c.sample_candidates_folder, c.indel_candidates_folder, c.filtered_positions_file)

            if not os.path.exists(indel_candidates_path):
                print("Candidate positions missing. Please run the filter script before predict.")
                return

            indel_candidates = pd.read_csv(indel_candidates_path, sep='\t', header=None, names=['chrom', 'pos'], dtype={'chrom': str, 'pos': int})

            # Sort the labels file by position and chromosome and then reindex
            indel_candidates = indel_candidates.sort_values(['pos'], ascending=[True]).reset_index(drop=True)

            print(("Number of INDEL candidates: ", len(indel_candidates)))

            indel_candidate_batches = np.array_split(indel_candidates, split_num)
            # remove empty batches
            indel_candidate_batches = [_ for _ in indel_candidate_batches if len(_)]

            try:
                Parallel(n_jobs=int(args.processes))( delayed(predict_indels)(batch, idx, args, indel_predictions_folder) for idx, batch in enumerate(indel_candidate_batches) )
            except joblib.my_exceptions.WorkerInterrupt as e:
                print(('workerinterrupt', e))

            concatenate_batch_prediction_results(indel_predictions_folder)

    """ MAKE VCF FILE """
    make_vcf(sample_folder, snv_predictions_file, indel_predictions_file, args)

def parse_args():
    parser = argparse.ArgumentParser(description="Model Predictions")
    parser.add_argument('--path_to_positions_to_predict')
    parser.add_argument('--num_nodes')
    parser.add_argument('--node_no')
    parser.add_argument('--environment', default='aquila') # nscc/aquila cluster/workstation, used to set appropriate file paths
    parser.add_argument('--experiment_id', default=None)
    parser.add_argument('--include_allele_frequency', required=False)

    parser.add_argument('--deep_sequencing', default=False)

    parser.add_argument('--sample_name', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--normal_bam', required=True)
    parser.add_argument('--tumor_bam', required=True)
    parser.add_argument('--processes', default=1, type=int)

    parser.add_argument('-snv', action='store_true') # read as snv_only
    parser.add_argument('-indel', action='store_true') # read as indel_only

    parser.add_argument('--update_batch_norm', default=False) # update batch norm stats for test sample

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.experiment_id:
        c.set_experiment_paths(int(args.experiment_id))

    if args.include_allele_frequency:
        print(('allele freq %s' % args.include_allele_frequency))
        print((args.include_allele_frequency))
        if args.include_allele_frequency == 'true':
            c.set_input_encoding(True)
            print('setting')
        else:
            c.set_input_encoding(False)

    if args.environment == 'workstation':
        ref_path = c.ref_path_on_workstation
        predictions_folder = c.predictions_folder_on_workstation

    elif args.environment == 'aquila':
        ref_path = c.ref_path_on_aquila
        predictions_folder = c.predictions_folder_on_aquila

    elif args.environment == 'nscc':
        ref_path = c.ref_path_on_nscc
        predictions_folder = c.predictions_folder_on_nscc
else:
    ref_path = c.ref_path_on_aquila
    ref_path = c.ref_path_on_nscc
    ref_file = pysam.FastaFile(ref_path)

if __name__ == '__main__':
    main()

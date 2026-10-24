import pysam
import numpy as np
import pandas as pd
import argparse
from time import time
import joblib
from joblib import Parallel, delayed, __version__
import os
import sys
import gc
import gzip
import mmap
import queue
import multiprocessing as mp

# Disable tf logging. 1 to filter out INFO logs, 2 to additionally filter out WARNING logs,
# and 3 to additionally filter out ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from filter import create_folder, resource_path

import snvs.constants as c
from snvs.generate_training_data import get_ref_base
from snvs.filter import check_read, get_snv
from snvs.predict import predict_snvs, get_model as get_snv_model, main_process_model
import snvs.constants as snv_c
from indels.filter import get_indels
from indels.predict import predict_indels, get_model as get_indel_model
import indels.constants as indel_c

def get_coverage(bamfile, chrom, ref_pos):
    coverage = bamfile.count_coverage(chrom, ref_pos, ref_pos+1)#, quality_threshold=c.MIN_BASE_QUALITY, read_callback=check_read)

    # [ (#A, #C, #G, #T) ] at each position in tumor
    coverage_list = [(coverage[0][i], coverage[1][i], coverage[2][i], coverage[3][i])
        for i in range(len(coverage[0]))]

    assert len(coverage_list) == 1
    coverage_list = coverage_list[0] # A C G T
    coverage_list = list(coverage_list) # convert tuple to list
    return coverage_list

def germline_filter(chrom, ref_pos, bamfile_n, ref_file):
    # return True if germline variant (snp or indel) found in the neighboring region
    # search to the left of the site (and 1bp to the right), in order to check if there is an indel that overlaps the somatic site but begins in a prior position
    margin = 0 # previously 1. reject a somatic site if there is a germline variant within this margin
    window = 1 # previously 10

    start, end = ref_pos + margin, ref_pos-window
    if end<0: end = 0 # sanity check

    check_sites = range(start, end, -1) # [ref_pos + 1, ref_pos, ref_pos-1, ref_pos-2, ..., ref_pos-49]

    # filter site if there exists a germline variant with AF > 0.1 AND it overlaps with the somatic site of interest
    for site in check_sites:
        snp=get_snv(get_coverage(bamfile_n, chrom, site), get_ref_base(site, chrom, ref_file))
        indel=get_indels(chrom, site, bamfile_n, ref_file)

        snp_AF, indel_AF = snp[-1], indel[-1]
        max_AF = max([snp_AF, indel_AF])

        GERMLINE_FILTER = 0.1

        if max_AF > GERMLINE_FILTER:
            # active germline variant, now check if it overlaps the somatic site
            if abs(site-ref_pos) <= margin:
                return True # site is close or equal to ref_pos

            # if it is an indel on the left of ref_pos, check if it overlaps the somatic site (ref_pos)
            elif site<ref_pos and indel_AF>=snp_AF:
                indel_length = abs(len(indel[0])-len(indel[1])) # length of insertion or deletion. diff between ref and alt sequence lengths
                right_end = site + indel_length # right end of indel

                # if the indel overlaps the somatic site within margin, reject somatic site
                if (right_end + margin >= ref_pos): 
                    return True

        else:
            # no active germline variant (AF>0.1) at this site, move on to next site
            continue

    return False

def parse_predictions_batch(batch, args, mode):    
    results = []
    ref_file = pysam.FastaFile(args.reference)

    bamfile_n = None
    if args.normal_bam and not args.ffpe:
        bamfile_n = pysam.AlignmentFile(args.normal_bam, "rb", check_sq=False)

    for _, row in batch.iterrows():
        # row: [CHROM, POS, REFERENCE_ALLELE, ALT_ALLELE, TUMOR_DEPTH, REFERENCE_ALLELE_COUNT_IN_TUMOR, ALT_ALLELE_READ_COUNT_IN_TUMOR, ALT_ALLELE_FRACTION_IN_TUMOR, pred_true]
        CHROM, POS, REFERENCE_ALLELE, ALT_ALLELE, TUMOR_DEPTH, REFERENCE_ALLELE_COUNT_IN_TUMOR, ALT_ALLELE_READ_COUNT_IN_TUMOR, ALT_ALLELE_FRACTION_IN_TUMOR, pred_true = \
            row[0], int(row[1]), row[2], row[3], int(row[4]), int(row[5]), int(row[6]), float(row[7]), float(row[8])

        if c.GERMLINE_FILTER and bamfile_n and germline_filter(CHROM, POS, bamfile_n, ref_file):
            continue
        
        ALT_ALLELE_FRACTION_IN_TUMOR = round(ALT_ALLELE_FRACTION_IN_TUMOR, 3)
        POSITION_1_INDEXED = POS + 1
        FILTER = 'PASS' if pred_true >= 0.5 else 'REJECT'
        INFO = 'TYPE=%s;SCORE=%s;DP=%d;RO=%d;AO=%d;AF=%s' % (mode, str(round(pred_true,4)), TUMOR_DEPTH, REFERENCE_ALLELE_COUNT_IN_TUMOR, ALT_ALLELE_READ_COUNT_IN_TUMOR, str(ALT_ALLELE_FRACTION_IN_TUMOR))
        FORMAT = 'GT:DP:RO:AO:AF'
        SAMPLE = '0/1:%d:%d:%d:%s' % (TUMOR_DEPTH, REFERENCE_ALLELE_COUNT_IN_TUMOR, ALT_ALLELE_READ_COUNT_IN_TUMOR, str(ALT_ALLELE_FRACTION_IN_TUMOR))

        OUT = (CHROM, POSITION_1_INDEXED, '.', REFERENCE_ALLELE, ALT_ALLELE, '.', FILTER, INFO, FORMAT, SAMPLE)
        results.append('\t'.join(map(str, OUT)) + '\n')

    if bamfile_n:
        bamfile_n.close()
    ref_file.close()
    return results

def split_candidates_into_batches(candidates, max_dist=1000000):
    """
    Group candidates such that each batch belongs to the same chromosome
    and the difference between the earliest and latest position is no more than max_dist (default: 2Mbp).
    
    Args:
        candidates: DataFrame with 'chrom' and 'pos' columns
        max_dist: Maximum distance between the earliest and latest position in a batch (default: 2Mbp)
    Returns:
        List of DataFrame batches (empty batches removed)
    """
    if len(candidates) == 0:
        return []
    
    cand_chroms = candidates['chrom'].values
    cand_pos = candidates['pos'].values
    split_indices = []

    batch_chrom = cand_chroms[0]
    batch_start_pos = cand_pos[0]

    for i in range(1, len(candidates)):
        curr_chrom = cand_chroms[i]
        curr_pos = cand_pos[i]

        if curr_chrom != batch_chrom or (curr_pos - batch_start_pos) > max_dist:
            split_indices.append(i)
            batch_chrom = curr_chrom
            batch_start_pos = curr_pos

    candidate_batches = np.split(candidates, split_indices)
    # remove empty batches
    candidate_batches = [batch for batch in candidate_batches if len(batch)]
    
    return candidate_batches

def concatenate_batch_prediction_results(predictions_folder):
    prediction_results_file = os.path.join(predictions_folder, c.combined_predictions_file)

    if os.path.exists(prediction_results_file):
        print(("Predictions file already exists: %s" % prediction_results_file))
        return

    from os import listdir
    from os.path import isfile, join

    batch_prediction_files = [join(predictions_folder, f) for f in listdir(predictions_folder) if isfile(join(predictions_folder, f))]
    
    positions_predicted = {} # for de-duplicating prediction sites

    with open(prediction_results_file, 'w') as f:
        for batch_file in batch_prediction_files:
            with open(batch_file) as r:
                for line in r:
                    CHROM, POS = line.strip().split()[0], line.strip().split()[1]
                    key = f'CHROM{CHROM}POS{POS}'
                    
                    if key not in positions_predicted:
                        positions_predicted[key]=True
                        f.write(line)

    for f in batch_prediction_files:
       os.remove(f)

PON_FILE = '1000g_pon.hg38.vcf.gz'

def make_vcf(sample_folder, snv_predictions_file, indel_predictions_file, args):
    from datetime import datetime

    # reference file handle is now opened within parallel parsing workers

    output_vcf = os.path.join(sample_folder, args.sample_name + '.vcf.gz')

    if os.path.exists(output_vcf):
        print("VCF file exists for sample. Delete the VCF to re-generate in current output dir.")
        print(("VCF:", output_vcf))
        return

    output_vcf = output_vcf.replace('.vcf.gz', '.temp.vcf.gz')

    if os.path.exists(output_vcf):
        # temp file exists, delete it
        os.remove(output_vcf)

    vcf_write = gzip.open(output_vcf, 'at')

    fileDate = datetime.now().strftime("%Y%B%d, %H:%M:%S")

    # declare contigs so tools like bcftools can write the records as BCF
    ref = pysam.FastaFile(args.reference)
    contig_lines = ''.join('##contig=<ID=%s,length=%d>\n' % (name, length) for name, length in zip(ref.references, ref.lengths))
    ref.close()

    vcf_header =  "##fileformat=VCFv4.2\n" + \
"""##fileDate=%s
##source=VarNet v%s
##reference=%s
##normalBAM=%s
##tumorBAM=%s
%s##INFO=<ID=TYPE,Number=.,Type=String,Description="Type of Somatic Event INDEL or SNV">
##INFO=<ID=SCORE,Number=1,Type=Float,Description="Prediction probability score">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Read Depth in the tumor">
##INFO=<ID=RO,Number=1,Type=Integer,Description="Reference allele observation count in the tumor">
##INFO=<ID=AO,Number=A,Type=Integer,Description="Alternate allele observation count in the tumor">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele fractions of alternate alleles in the tumor">
##FILTER=<ID=PASS,Description="Accept as somatic mutation with probability score at least 0.5">
##FILTER=<ID=REJECT,Description="Reject somatic mutation with probability score value below 0.5">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth in the tumor">
##FORMAT=<ID=RO,Number=1,Type=Integer,Description="Reference allele observation count in the tumor">
##FORMAT=<ID=AO,Number=A,Type=Integer,Description="Alternate allele observation count in the tumor">
##FORMAT=<ID=AF,Number=1,Type=Float,Description="Allele fractions of alternate alleles in the tumor">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n""" % (fileDate, c.__VERSION__, args.reference, args.normal_bam, args.tumor_bam, contig_lines)

    vcf_write.write(vcf_header)

    ALLELES = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    ALLELE_INDICES = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    if args.ffpe:
        # retain all calls from original args.vcf
        CUT_OFF = 0
    elif (not args.normal_bam):
        # tumor-only 
        CUT_OFF = 0
    else:
        # tumor-normal frozen
        CUT_OFF = 0.1 # 0.3

    if args.threshold is not None:
        CUT_OFF = args.threshold
        print('>>> VCF minimum score threshold:', CUT_OFF)

    # BAM handles are now opened within parallel parsing workers

    if not args.indel: # if not indel only
        snv_df = pd.read_csv(snv_predictions_file, sep='\t', header=None, dtype={0: str}) # read chrom as str, numeric chrom names (e.g. GRCh37 '1') are otherwise read as int, which pysam rejects
        snv_df = snv_df[snv_df[8] >= CUT_OFF]
        if len(snv_df) > 0:
            batches = np.array_split(snv_df, min(len(snv_df), args.processes * 4))
            results = Parallel(n_jobs=args.processes)(delayed(parse_predictions_batch)(batch, args, 'SNV') for batch in batches)
            for res_list in results:
                for line in res_list:
                    vcf_write.write(line)

    if not args.snv: # if not snv only
        indel_df = pd.read_csv(indel_predictions_file, sep='\t', header=None, dtype={0: str}) # read chrom as str, numeric chrom names (e.g. GRCh37 '1') are otherwise read as int, which pysam rejects
        indel_df = indel_df[indel_df[8] >= CUT_OFF]
        if len(indel_df) > 0:
            batches = np.array_split(indel_df, min(len(indel_df), args.processes * 4))
            results = Parallel(n_jobs=args.processes)(delayed(parse_predictions_batch)(batch, args, 'INDEL') for batch in batches)
            for res_list in results:
                for line in res_list:
                    vcf_write.write(line)

    vcf_write.close()
    
    final_vcf = output_vcf.replace('.temp.vcf.gz', '.vcf.gz')
    os.rename(output_vcf, final_vcf)

    if not args.normal_bam:
        # tumor-only, do PoN subtraction
        pon_vcf = final_vcf.replace('.vcf.gz', '.pon.vcf')
        os.system(f'bedtools subtract -a {final_vcf} -b {resource_path(args, PON_FILE)} -header > {pon_vcf} && gzip {pon_vcf} && mv {pon_vcf}.gz {final_vcf}')
        print('>>> Panel of normal subtraction complete.')

    # sort vcf
    print('>>> Sorting VCF...')
    sorted_vcf = final_vcf.replace('.vcf.gz', '.sort.vcf')
    # sort in reference order so records follow the declared contig order
    os.system(f'bedtools sort -faidx {args.reference}.fai -i {final_vcf} -header > {sorted_vcf} && gzip {sorted_vcf} && mv {sorted_vcf}.gz {final_vcf}')
    print('>>> VCF sorted.')

    print("Output VCF:", final_vcf)

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

def parse_vcf(vcf_file, filename, snv=True, PASS_only=True):
    count = 0

    if vcf_file.endswith('.gz'):
        import gzip
        vcf = gzip.open(vcf_file, 'rt')
    else:
        vcf = open(vcf_file)

    is_varnet_vcf, is_mutect2_vcf, strelka = False, False, False # flag to check if parsing varnet's vcf
    
    with open(filename, 'a') as f:
        for line in vcf:
            if line.startswith('#'):
                # header
                
                if line.startswith('##source=VarNet'):
                    is_varnet_vcf = True
                    # is_varnet_vcf = False # use only PASS calls for varnet too
                    print('>>> Parsing VarNet VCF:', vcf_file)

                elif line.startswith('##source=Mutect2'):
                    is_mutect2_vcf = True
                    print('>>> Parsing Mutect2 VCF:', vcf_file)

                elif line.startswith('##source=strelka'):
                    is_strelka_vcf = True
                    print('>>> Parsing Strelka VCF:', vcf_file)

                continue
                
            CHROM, POS, REF, ALT, FILTER = line.split()[0], line.split()[1], line.split()[3], line.split()[4], line.split()[6]

            if is_varnet_vcf:
                # separate rule for varnet vcf
                SCORE = float(line.split('SCORE=')[1].split(';')[0])
                if SCORE > 0.50: # 0.95, 0.9, 0.7 and 0.8 about the same for snv
                    USE_CALL = True
                else:
                    USE_CALL = False

            elif is_mutect2_vcf:
                import re
                columns = line.strip().split('\t')
                info_column = columns[7]
                tlod_match = re.search(r'TLOD=([\d\.]+)', info_column)

                if tlod_match:
                    TLOD = float(tlod_match.group(1))
                    USE_CALL = FILTER == 'PASS' and TLOD > 20 # 23 # 19 # 15 # 10 # 7 mutect2's default --tumor-lod is 3.0 (calls below this are not included in vcf)
                else:
                    USE_CALL = FILTER == 'PASS' # use all PASS calls

                # USE_CALL = FILTER == 'PASS' # use all PASS calls
                
            elif is_strelka_vcf:
                import re
                columns = line.strip().split('\t')
                info_column = columns[7]
                somaticevs_match = re.search(r'SomaticEVS=([\d\.]+)', info_column)
                
                if somaticevs_match:
                    SomaticEVS = float(somaticevs_match.group(1))
                    USE_CALL = FILTER == 'PASS' and SomaticEVS > 10 # 12 # 14 # 10 # strelka2's default EVS cutoff appears to be around 6 although it varies with version: https://github.com/Illumina/strelka/issues/79
                else:
                    USE_CALL = FILTER == 'PASS' # use all PASS calls
            else:
                # other callers
                USE_CALL = FILTER == 'PASS'
                # USE_CALL = FILTER == 'PASS' or (not PASS_only)
                
            if USE_CALL:
                if (len(REF) == 1 and (len(ALT) == 1 or (len(ALT) == 3 and ALT[1] == ','))): # REF and ALT should be one base each (ALT can have two possible bases separated by comma e.g. 'A,G')
                    # SNV
                    if snv:
                        f.write('%s\t%d\n' % (CHROM, int(POS)-1)) # convert to 0-indexed
                        count+=1
                else:
                    #INDEL
                    if not snv:
                        f.write('%s\t%d\n' % (CHROM, int(POS)-1)) # convert to 0-indexed
                        count+=1

    vcf.close()

    print('Extracted', count, 'SNVs' if snv else 'indels', 'from', vcf_file)

""" SINGLE MODEL INFERENCE

Worker processes generate the input tensors (predict_snvs/predict_indels) and the main process holds the only
copy of the model. Each worker has its own shared-memory slot of args.batch_size tensors: it copies a normalized
batch into the slot, sends a request and waits for the model output (snvs.predict.MainProcessModel). Workers are
forked before tensorflow is imported, so they never load tensorflow or the model.
"""

def _init_worker(server, worker_ids):
    worker_id = worker_ids.get()
    main_process_model.connect(server.buffer, worker_id * server.slot_size * 4, server.slot_size, server.requests, server.replies[worker_id], worker_id)

class InferenceServer:
    def __init__(self, args):
        ctx = mp.get_context('fork')
        self.processes = args.processes

        # one slot per worker, large enough for a batch of the larger of the snv and indel input tensors
        self.slot_size = args.batch_size * max(int(np.prod(snv_c.INPUT_SHAPE)), int(np.prod(indel_c.INPUT_SHAPE))) # float32 values
        self.buffer = mmap.mmap(-1, self.processes * self.slot_size * 4) # anonymous shared memory, pages allocated when written

        self.requests = ctx.Queue()
        self.replies = [ctx.SimpleQueue() for _ in range(self.processes)]

        worker_ids = ctx.SimpleQueue()
        for i in range(self.processes):
            worker_ids.put(i)

        self.pool = ctx.Pool(self.processes, initializer=_init_worker, initargs=(self, worker_ids))
        self.worker_pids = {p.pid for p in self.pool._pool}
        self.device = None

    def slot(self, worker_id, shape):
        size = int(np.prod(shape))
        assert size <= self.slot_size, 'batch does not fit in shared memory slot'
        return np.frombuffer(self.buffer, dtype=np.float32, count=size, offset=worker_id * self.slot_size * 4).reshape(shape)

    def setup_device(self):
        """Use the first GPU if available (one model copy only needs one), else CPU."""
        if self.device is not None:
            return

        # Cap TensorFlow thread pools to prevent thread explosion (and out-of-memory issues): without caps, TF
        # defaults to using all available cores for intra-op parallelism. Let the user override via environment.
        os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
        os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            # allocate GPU memory as needed instead of reserving all of it
            tf.config.experimental.set_memory_growth(gpus[0], True)
            self.device = gpus[0].name
        else:
            self.device = 'CPU'

        print('>>> Model inference device:', self.device)

    def run(self, predict_fn, tasks, get_model):
        """Run predict_fn(*task) in the workers while serving their model requests."""
        self.setup_device()
        model = get_model(args)

        result = self.pool.starmap_async(predict_fn, tasks, chunksize=1)

        while not result.ready():
            try:
                worker_id, shape = self.requests.get(timeout=1)
            except queue.Empty:
                if {p.pid for p in self.pool._pool} != self.worker_pids:
                    raise RuntimeError('Prediction worker process terminated unexpectedly (e.g. out of memory).')
                continue

            y_pred = model(self.slot(worker_id, shape), training=False).numpy()
            self.replies[worker_id].put(y_pred)

        result.get() # raise worker exceptions

        del model
        from tensorflow.keras import backend as K
        K.clear_session()

    def close(self):
        self.pool.close()
        self.pool.join()

def main():
    if args.ffpe:
        # extract candidates from args.vcf
        candidates_folder = os.path.join(sample_folder, c.sample_candidates_folder)

        create_folder(candidates_folder)
        print(("Candidates Directory: %s\n" % candidates_folder))

        snv_candidates_folder = os.path.join(candidates_folder, c.snv_candidates_folder)
        indel_candidates_folder = os.path.join(candidates_folder, c.indel_candidates_folder)
        create_folder(snv_candidates_folder)
        create_folder(indel_candidates_folder)
        
        if not args.indel: # do snv
            snv_candidates_file = os.path.join(snv_candidates_folder, c.filtered_positions_file)
            if not os.path.exists(snv_candidates_file):
                parse_vcf(args.vcf, snv_candidates_file, snv=True)
            else:
                print('>>> SNV candidates files exists.')

        if not args.snv: # do indel
            indel_candidates_file = os.path.join(indel_candidates_folder, c.filtered_positions_file)
            if not os.path.exists(indel_candidates_file):
                parse_vcf(args.vcf, indel_candidates_file, snv=False)
            else:
                print('>>> INDEL candidates files exists.')
        
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

            snv_candidates = pd.read_csv(snv_candidates_path, sep='\t', header=None, names=['chrom', 'pos', 'REF', 'ALT', 'DP', 'RO', 'AO', 'AF'], dtype={'chrom': str, 'pos': int, 'REF': str, 'ALT': str, 'DP': int, 'RO': int, 'AO': int, 'AF': float})

            # Sort the candidates file by chromosome and position and then reindex
            snv_candidates = snv_candidates.sort_values(['chrom', 'pos'], ascending=[True, True]).reset_index(drop=True)

            print(("Number of SNV candidates: ", len(snv_candidates)))

            snv_candidate_batches = split_candidates_into_batches(snv_candidates)

            try:
                inference_server.run(predict_snvs, [(batch, idx, args, snv_predictions_folder) for idx, batch in enumerate(snv_candidate_batches)], get_snv_model)
            except KeyboardInterrupt as e:
                print(('workerinterrupt', e))

            concatenate_batch_prediction_results(snv_predictions_folder)
            gc.collect()

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

            indel_candidates = pd.read_csv(indel_candidates_path, sep='\t', header=None, names=['chrom', 'pos', 'REF', 'ALT', 'DP', 'RO', 'AO', 'AF'], dtype={'chrom': str, 'pos': int, 'REF': str, 'ALT': str, 'DP': int, 'RO': int, 'AO': int, 'AF': float})

            # Sort the candidates file by chromosome and position and then reindex
            indel_candidates = indel_candidates.sort_values(['chrom', 'pos'], ascending=[True, True]).reset_index(drop=True)

            print(("Number of INDEL candidates: ", len(indel_candidates)))
            
            # indel model and input tensor is larger so reduce max_dist to 250kbp
            indel_candidate_batches = split_candidates_into_batches(indel_candidates, max_dist=250000)

            try:
                inference_server.run(predict_indels, [(batch, idx, args, indel_predictions_folder) for idx, batch in enumerate(indel_candidate_batches)], get_indel_model)
            except KeyboardInterrupt as e:
                print(('workerinterrupt', e))

            concatenate_batch_prediction_results(indel_predictions_folder)
            gc.collect()

    if args.ffpe:
        # filter args.vcf
        filter_vcf(sample_folder, snv_predictions_file, indel_predictions_file, args)
    else:
        """ MAKE VCF FILE """
        make_vcf(sample_folder, snv_predictions_file, indel_predictions_file, args)

    return snv_predictions_file, indel_predictions_file

def filter_vcf(sample_folder, snv_predictions_file, indel_predictions_file, args):
    output_vcf_filename = os.path.join(sample_folder, args.sample_name + '.vcf')
    
    if os.path.exists(output_vcf_filename):
        print(f'>>> Filtered VCF file exists. Delete it to re-generate: {output_vcf_filename}')
        return
        
    output_vcf = open(output_vcf_filename, 'w')

    varnet_ffpe_calls = {}
    filtered_artifact_count = 0

    def parse_preds(predictions_file, varnet_ffpe_calls):        
        with open(predictions_file) as r:
            for line in r:
                line = line.strip()
                s=line.split('\t')
                chrom, pos, pred_true = s[0], int(s[1]), round(float(s[2]), 4)

                key = f'chrom{chrom}pos{pos+1}' # convert pos to 1-indexed
                varnet_ffpe_calls[key] = pred_true
    
    parse_preds(snv_predictions_file, varnet_ffpe_calls)
    parse_preds(indel_predictions_file, varnet_ffpe_calls)
        
    if args.vcf.endswith('.gz'):
        import gzip
        vcf = gzip.open(args.vcf, 'rt')
    else:
        vcf = open(args.vcf)

    is_varnet_vcf, is_mutect2_vcf, strelka = False, False, False
    ARTIFACT_CUT_OFF = 0.5 # DEFAULT for other callers

    for line in vcf:
        if line.startswith('#'):
            # write header
            output_vcf.write(line)

            if line.startswith('##source=VarNet'):
                is_varnet_vcf = True
                ARTIFACT_CUT_OFF = 0.7 # 0.7
            elif line.startswith('##source=Mutect2'):
                is_mutect2_vcf = True
                ARTIFACT_CUT_OFF = 0.05
            elif line.startswith('##source=strelka'):
                is_strelka_vcf = True
                ARTIFACT_CUT_OFF = 0.85

            continue

        CHROM, POS, REF, ALT, FILTER = line.split()[0], line.split()[1], line.split()[3], line.split()[4], line.split()[6]

        if FILTER != 'PASS' and not is_varnet_vcf:
            continue # skip non-PASS calls from other callers
        else:
            # PASS call or varnet_vcf (output all calls for varnet)
            key = f'chrom{CHROM}pos{POS}'
            if key in varnet_ffpe_calls and varnet_ffpe_calls[key] < ARTIFACT_CUT_OFF:
                # skip artifact
                filtered_artifact_count += 1
                continue
            else:
                output_vcf.write(line)

    print('>>> Filtered artifacts:', filtered_artifact_count)
    print(f'>>> Saved filtered VCF: {output_vcf_filename}')
    
def parse_args():
    parser = argparse.ArgumentParser(description="Model Predictions")
    parser.add_argument('--version', action='version', version=f'VarNet {c.__VERSION__}')
    parser.add_argument('--path_to_positions_to_predict')
    parser.add_argument('--num_nodes')
    parser.add_argument('--node_no')
    parser.add_argument('--environment', default='aquila') # nscc/aquila cluster/workstation, used to set appropriate file paths
    parser.add_argument('--experiment_id', default=None)
    parser.add_argument('--include_allele_frequency', required=False)

    parser.add_argument('--sample_name', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--normal_bam', required=False, default=None)
    parser.add_argument('--tumor_bam', required=True)
    parser.add_argument('--processes', default=1, type=int)

    parser.add_argument('-snv', action='store_true') # read as snv_only
    parser.add_argument('-indel', action='store_true') # read as indel_only

    parser.add_argument('--ffpe', action='store_true') # for FFPE samples, must provide VCF
    parser.add_argument('--vcf', default=False, type=str, help='VCF file for FFPE prediction') # for FFPE samples, must provide VCF

    parser.add_argument('--threshold', default=None, type=float)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for model predictions (default: 64)')
    parser.add_argument('--resource_dir', required=False, type=str, default='.', help='folder with the panel of normals resource (for tumor-only mode, default: current folder)')

    return parser.parse_args()

            
if __name__ == '__main__':
    args = parse_args()

    if not args.normal_bam:
        # tumor-only: check the panel of normals exists before running predictions
        resource_path(args, PON_FILE)

    if args.ffpe and not args.vcf:
        parser.error("--vcf is required if --ffpe is provided.")

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

    sample_folder = os.path.join(args.output_dir, args.sample_name)
    args.sample_folder = sample_folder
    create_folder(sample_folder)

    output_vcf = os.path.join(sample_folder, args.sample_name + '.vcf')
    if (os.path.exists(output_vcf) or os.path.exists(output_vcf + '.gz')):
        print("VCF file exists for sample. Use new output_dir to re-run sample or delete the VCF to re-generate in current output dir.")
        print(("VCF:", output_vcf))
    else:
        # single model in the main process. created before tensorflow is imported so the forked workers do not carry it
        inference_server = InferenceServer(args)

        # call main() function
        main()

        inference_server.close()

else:
    ref_path = c.ref_path_on_aquila
    ref_path = c.ref_path_on_nscc
    ref_file = pysam.FastaFile(ref_path)



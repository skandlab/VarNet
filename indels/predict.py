# from check_memory_consumption import memory_consumed

# The main prediction pipeline script that loads the trained model and evaluates positions one-by-one on
# CPUs in parallel, given positions file to evaluate on. Generates y-file and
# computes resulting metrics

# from memory_profiler import profile

# === LIBRARIES === #
import numpy as np
import pandas as pd
import pysam

import os
import sys
import argparse
from time import time
from joblib import Parallel, delayed, __version__

# Disable tf logging. 1 to filter out INFO logs, 2 to additionally filter out WARNING logs,
# and 3 to additionally filter out ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import indels.constants as c
from indels.generate_training_data_specialized import create_input_tensor_for_position
from snvs.generate_training_data import create_tumor_only_input_tensor_for_position

from utils import get_ref_file, update_batch_norm_fn

CURRENT_DIR = os.path.dirname(__file__)

def parse_args():
    parser = argparse.ArgumentParser(description="Model Predictions")
    parser.add_argument('--path_to_normal_bam', default='')
    parser.add_argument('--path_to_tumor_bam', default='')
    parser.add_argument('--path_to_positions_to_predict', default='')
    parser.add_argument('--sample_name')
    parser.add_argument('--number_of_cores_used_per_node')
    parser.add_argument('--number_of_nodes_being_used')
    parser.add_argument('--node_no')
    parser.add_argument('--environment', default='aquila') # nscc/aquila cluster/workstation, used to set appropriate file paths
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    predictions_folder = c.predictions_folder

def get_model(args):
    from tensorflow.keras.models import model_from_json

    if args.normal_bam and not args.ffpe:
        # frozen tumor-normal convnet2 and not ffpe
        BEST_MODEL_ARCHITECTURE_PATH = c.BEST_MODEL_ARCHITECTURE_PATH
        BEST_MODEL_WEIGHTS_PATH = c.BEST_MODEL_WEIGHTS_PATH

        with open(os.path.join(CURRENT_DIR, BEST_MODEL_ARCHITECTURE_PATH)) as f:
            model_architecture = f.read()
        model = model_from_json(model_architecture)
        model.load_weights(os.path.join(CURRENT_DIR, BEST_MODEL_WEIGHTS_PATH))

    else:
        # frozen tumor-only or ffpe

        # <start> tumor-only convnet
        if args.ffpe:
            raise Exception('which model to load?')
        else:
            BEST_MODEL_ARCHITECTURE_PATH = c.TUMOR_ONLY_BEST_MODEL_ARCHITECTURE_PATH
            BEST_MODEL_WEIGHTS_PATH = c.TUMOR_ONLY_BEST_MODEL_WEIGHTS_PATH

            with open(os.path.join(CURRENT_DIR, BEST_MODEL_ARCHITECTURE_PATH)) as f:
                model_architecture = f.read()
            model = model_from_json(model_architecture)
            model.load_weights(os.path.join(CURRENT_DIR, BEST_MODEL_WEIGHTS_PATH))
        # </start> tumor-only convnet

        # <start> transformer 
        # from tensorflow.keras.models import load_model
        # if args.ffpe:
        #     # load ffpe model transformer
        #     BEST_MODEL_PATH = c.BEST_FFPE_TUMOR_ONLY_MODEL_PATH
        # else:
        #     # frozen tumor-only transformer
        #     BEST_MODEL_PATH = c.BEST_TUMOR_ONLY_MODEL_PATH

        # from pathlib import Path
        # transformer_root = Path(__file__).parents[1] # ../
        # path = os.path.join(transformer_root, BEST_MODEL_PATH)
        # print('Loading checkpoint:', path)
        # model = load_model(path)
        # <start> transformer 

    return model

def predict_position(input_tensor, model, channel_means, channel_stds, training=False, args=None):
    if input_tensor.dtype != np.float32:
        input_tensor = input_tensor.astype(np.float32)

    if channel_means is not None and channel_stds is not None:
        # normalization not required for tumor-only/FFPE transformers
        input_tensor -= channel_means
        input_tensor /= channel_stds

    if not training:
        y_pred_test = model.predict(input_tensor)
        return np.mean(y_pred_test)
    else:
        # to update batch norm statistics
        y_pred_test = model(input_tensor, training=True)
        return y_pred_test

def predict_indels(positions_to_predict, batch_num, args, indel_predictions_folder, output_path=None, update_batch_norm=False):
    print(("INDEL PREDICTION BATCH:", batch_num))

    if not output_path:
        csv_output_filename = "batch_%d.csv" % ( batch_num )
        output_path = os.path.join(indel_predictions_folder, csv_output_filename)
    
    if os.path.isfile(output_path):
        # don't delete batch since it is saved in one shot
        print(("BATCH COMPLETE:", output_path))
        return

    output_path = output_path.replace('.csv', '.temp.csv')      

    positions_completed = {}
    
    if os.path.exists(output_path):
        print(("FETCHING PREDICTIONS FROM PREVIOUS RUN: %s" % output_path))
        # temp file exists
        with open(output_path) as pfile:
            for idx, pline in enumerate(pfile):
                s = pline.strip().split()
                chrom, pos = s[0], s[1]
                pos_key = 'chrom%spos%s' % (chrom, pos)
                positions_completed[pos_key] = True

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.normal_bam:
        bamfile_n = pysam.AlignmentFile(args.normal_bam, "rb") # normal bamfile
    else:
        # tumor only mode
        bamfile_n = None
        
    bamfile_t = pysam.AlignmentFile(args.tumor_bam, "rb") # tumor bamfile
    ref_file = get_ref_file(args.reference) #Use one ref file per process due to parallelization issues

    model = get_model(args)
    assert model is not None

    columns = ['chrom', 'pos', 'REF', 'ALT', 'DP', 'RO', 'AO', 'AF', 'pred_true']
    results = pd.DataFrame(columns=columns)
    results['chrom'] = results['chrom'].astype(str)
    results['pos'] = results['pos'].astype(int)
    results['REF'] = results['REF'].astype(str)
    results['ALT'] = results['ALT'].astype(str)
    results['DP'] = results['DP'].astype(int)
    results['RO'] = results['RO'].astype(int)
    results['AO'] = results['AO'].astype(int)
    results['AF'] = results['AF'].astype(np.float64)
    results['pred_true'] = results['pred_true'].astype(np.float64)

    positions_iterator = positions_to_predict.iterrows()
    positions = []

    for i, row in positions_iterator:
        positions.append((row['pos'], row['chrom'], row['REF'], row['ALT'], row['DP'], row['RO'], row['AO'], row['AF']))

    if args.normal_bam and not args.ffpe:
        # tumor-normal frozen
        channel_means = np.load(os.path.join(CURRENT_DIR, c.NORMALIZATION_MEANS_PATH))
        channel_stds = np.load(os.path.join(CURRENT_DIR, c.NORMALIZATION_STD_DEVS_PATH))
    else:
        # tumor only frozen or ffpe

        # <load mean/std for tumor-only convnet encoding using varnet 1.1.0 train set>
        channel_means = np.load(os.path.join(CURRENT_DIR, c.NORMALIZATION_MEANS_PATH))
        channel_stds = np.load(os.path.join(CURRENT_DIR, c.NORMALIZATION_STD_DEVS_PATH))
        # </load mean/std for tumor-only convnet encoding using varnet 1.1.0 train set>

        # <load mean/std for tumor-only convnet encoding>
        # channel_means = np.load(os.path.join(CURRENT_DIR, c.TUMOR_ONLY_NORMALIZATION_MEANS_PATH))
        # channel_stds = np.load(os.path.join(CURRENT_DIR, c.TUMOR_ONLY_NORMALIZATION_STD_DEVS_PATH))
        # <load mean/std for tumor-only convnet encoding>
        
        # < set to None for transformer, no need normalization>
        # channel_means, channel_stds = None, None 
        # </ set to None for transformer, no need normalization>        

    if args.update_batch_norm:
        update_batch_norm_fn(model, positions, bamfile_n, bamfile_t, channel_means, channel_stds, ref_file=ref_file, create_input_fn=create_input_tensor_for_position, predict_fn=predict_position)

    for i, row in enumerate(positions):
        pos, chrom, REF, ALT, DP, RO, AO, AF = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]

        pos_key = 'chrom%spos%s' % (chrom, pos)

        if pos_key in positions_completed:
            continue

        if args.normal_bam and not args.ffpe:
            # inceptionv3 tumor-normal
            input_tensor = create_input_tensor_for_position(chrom, pos, bamfile_n, bamfile_t, ref_file)
        else:
            # inceptionv3 tumor-normal
            input_tensor = create_input_tensor_for_position(chrom, pos, bamfile_n, bamfile_t, ref_file)

            # tumor-only transformer encoding
            # input_tensor = create_tumor_only_input_tensor_for_position(chrom, pos, bamfile_t, ref_file)

        pred_true = predict_position(input_tensor, model, channel_means, channel_stds, args=args)

        results_dict = {'chrom': chrom, 'pos': pos, 'REF': REF, 'ALT': ALT, 'DP': DP, 'RO': RO, 'AO': AO, 'AF': AF, 'pred_true': pred_true}

        results = results.append(results_dict, ignore_index = True)

        if len(results) > 100:
            # append predictions to the file every 1000 predictions
            results.to_csv(output_path, sep='\t', index=False, encoding='utf-8', mode='a', header=False)
            results.drop(results.index, inplace=True)

    # write remaining results to file
    if len(results):
        results.to_csv(output_path, sep='\t', index=False, encoding='utf-8', mode='a', header=False)

    os.rename(output_path, output_path.replace('.temp.csv', '.csv'))

def concatenate_batch_prediction_results(predictions_folder):
    prediction_results_file = os.path.join(predictions_folder, c.combined_predictions_file)

    from os import listdir
    from os.path import isfile, join

    # get all the files in the predictions folder
    batch_prediction_files = [join(predictions_folder, f) for f in listdir(predictions_folder) if isfile(join(predictions_folder, f))]

    # lets get the header row from one of the batches
    some_batch = pd.read_csv(batch_prediction_files[0], sep='\t', header=0)
    # next, drop the data
    some_batch.drop(some_batch.index, inplace=True)
    # let's write just the header row to the csv file
    some_batch.to_csv(prediction_results_file, sep='\t', index=False, encoding='utf-8')

    for batch in batch_prediction_files:
        p = pd.read_csv(batch, sep='\t', header=None)
        p.to_csv(prediction_results_file, sep='\t', index=False, encoding='utf-8', mode='a', header=False)

    print(("Prediction output file: %s" % prediction_results_file))


if __name__ == '__main__':

    positions_to_predict = pd.read_csv(args.path_to_positions_to_predict, sep='\t', header=None, names=['chrom', 'pos', 'is_real'], dtype={'chrom': str, 'pos': int}, skiprows=1)
    #positions_to_predict = pd.read_csv(args.path_to_positions_to_predict, sep='\t', header=None, names=['X', 'chrom', 'pos','start', 'end', 'ref_mfv', 'alt_mfv', 'predict', 'fa', 'tr'], dtype={'chrom': str, 'pos': int}, skiprows=1)
    # Sort the labels file by position and chromosome and then reindex
    time_to_sort = time()
    positions_to_predict = positions_to_predict.sort_values(['pos'], ascending=[True]).reset_index(drop=True)
    print(("Time to sort %.5f seconds" % ( time() - time_to_sort )))

    print(("Number of positions for prediction: ", len(positions_to_predict)))

    # create the predictions output folder if it does not exist
    if not os.path.exists(predictions_folder):
        os.makedirs(predictions_folder)

    # create folder for this current run
    predictions_folder = os.path.join(predictions_folder, args.sample_name)
    if not os.path.exists(predictions_folder):
       os.makedirs(predictions_folder)

    global_start_time = time()

    # split positions_to_predict list into equal batches to run on multiple nodes
    positions_to_predict_split_into_batches = np.array_split(positions_to_predict, int(args.number_of_nodes_being_used))

    # get positions for this node
    positions_to_predict = positions_to_predict_split_into_batches[int(args.node_no)]

    # split positions_to_predict list into equal batches to run on multiple cores
    positions_to_predict_split_into_batches = np.array_split(positions_to_predict, int(args.number_of_cores_used_per_node))

    print(("num of batches " + str(len(positions_to_predict_split_into_batches))))
    print(("cores " + str(args.number_of_cores_used_per_node)))

    #for idx, batch in enumerate(positions_to_predict_split_into_batches):
    #    predict_indels(batch, idx, args)

    # process all the batches
    Parallel(n_jobs=int(args.number_of_cores_used_per_node))( delayed(predict_indels)(batch, idx, args) for idx, batch in enumerate(positions_to_predict_split_into_batches) )

    #concatenate_batch_prediction_results()

    print(("|FINAL TIME %.1f\n" % (time() - global_start_time)))

# from check_memory_consumption import memory_consumed

# The main prediction pipeline script that loads the trained model and evaluates positions one-by-one on
# CPUs in parallel, given positions file to evaluate on. Generates y-file and
# computes resulting metrics

# from memory_profiler import profile

import os
from time import time

# === LIBRARIES === #
import pysam
import numpy as np
import pandas as pd
import pysam

import joblib
from joblib import Parallel, delayed, __version__
from snvs.filter import check_read
from utils import get_ref_file, update_batch_norm_fn, ttt_get_models, test_time_train

# Disable tf logging. 1 to filter out INFO logs, 2 to additionally filter out WARNING logs,
# and 3 to additionally filter out ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import snvs.constants as c

from snvs.generate_training_data import get_reference, generate_image, populate_array, create_input_tensor_for_position, get_ref_base

CURRENT_DIR = os.path.dirname(__file__) 

def get_model():
    from tensorflow.keras.models import model_from_json
    with open(os.path.join(CURRENT_DIR, c.BEST_MODEL_ARCHITECTURE_PATH)) as f:
        model_architecture = f.read()
    
    if c.TEST_TIME_TRAINING:
        from tensorflow.keras.models import load_model
        import tensorflow_addons as tfa
        # for group normalization
        #model = model_from_json(model_architecture, custom_objects={'Addons>GroupNormalization': tfa.layers.GroupNormalization})
        model = load_model(os.path.join(CURRENT_DIR, c.BEST_MODEL_PATH), custom_objects={'Addons>GroupNormalization': tfa.layers.GroupNormalization})
        assert model.optimizer is not None
    else:
        model = model_from_json(model_architecture)
        model.load_weights(os.path.join(CURRENT_DIR, c.BEST_MODEL_WEIGHTS_PATH))

    return model

def predict_position(input_tensor, model, channel_means, channel_stds, training=False):
    input_tensor -= channel_means
    input_tensor /= channel_stds

    if not training:
        y_pred_test = model.predict(input_tensor)
        return np.mean(y_pred_test) # return mean in case of c.SAMPLE_READS_COUNT > 1, batch prediction by sampling multiple batches of reads for the SAME position (NOT for a batch of different positions). no difference if c.SAMPLE_READS_COUNT=1
    else:
        # to update batch norm statistics
        y_pred_test = model(input_tensor, training=True)
        return y_pred_test

def predict_snvs(positions_to_predict, batch_num, args, snv_predictions_folder, output_path=None, deep_sequencing=False, update_batch_norm=False):
    print(("SNV PREDICTION BATCH:", batch_num))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    bamfile_n = pysam.AlignmentFile(args.normal_bam, "rb", check_sq=False) # normal bamfile
    bamfile_t = pysam.AlignmentFile(args.tumor_bam, "rb", check_sq=False) # tumor bamfile
    ref_file = get_ref_file(args.reference)    

    columns = ['chrom', 'pos', 'pred_true']
    results = pd.DataFrame(columns=columns)
    results['chrom'] = results['chrom'].astype(str)
    results['pos'] = results['pos'].astype(int)
    results['pred_true'] = results['pred_true'].astype(np.float64)

    if not output_path:
        csv_output_filename = "batch_%d.csv" % ( batch_num )
        output_path = os.path.join(snv_predictions_folder, csv_output_filename)

    if os.path.exists(output_path):
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
                s = pline.strip().split('\t')
                chrom, pos, pred_true = s[0], s[1], float(s[2])
                pos_key = 'chrom%spos%s' % (chrom, pos)
                positions_completed[pos_key] = True

    model = get_model()
    
    assert model is not None
    
    channel_means = np.load(os.path.join(CURRENT_DIR, c.NORMALIZATION_MEANS_PATH))
    channel_stds = np.load(os.path.join(CURRENT_DIR, c.NORMALIZATION_STD_DEVS_PATH))

    positions_iterator = positions_to_predict.iterrows()
    positions = []

    for i, row in positions_iterator:
        positions.append((row['pos'], row['chrom']))

    if args.update_batch_norm:
        update_batch_norm_fn(model, positions, bamfile_n, bamfile_t, channel_means, channel_stds, ref_file=ref_file, create_input_fn=create_input_tensor_for_position, predict_fn=predict_position)

    if c.TEST_TIME_TRAINING:
        classifier_head, self_supervised_head = ttt_get_models(model, lr=0.0001)
        import random
        random.shuffle(positions) # shuffle positions for TTT

    for i, row in enumerate(positions):
        start_time = time()
        pos = row[0]
        chrom = row[1]
        
        pos_key = 'chrom%spos%s' % (chrom, pos)

        if pos_key in positions_completed:
            continue

        if deep_sequencing:
            '''
            Sampling reads
            '''
            n = bamfile_n.count_coverage(chrom, pos, pos + 1)
            normal_depth = int(n[0][0] + n[1][0] + n[2][0] + n[3][0])
            n = bamfile_t.count_coverage(chrom, pos, pos+1)
            tumor_depth = int(n[0][0] + n[1][0] + n[2][0] + n[3][0])
       
            max_depth = max(tumor_depth, normal_depth)

            if max_depth <= 100:
                # Not more than 100 reads here
                input_tensor = create_input_tensor_for_position(chrom, pos, bamfile_n, bamfile_t, ref_file)
                pred_true = predict_position(input_tensor, model, channel_means, channel_stds)

            else:        
                num_read_samples = max_depth/c.NUM_READS
                assert num_read_samples > 0
    
                pred_trues = []

                print((normal_depth, tumor_depth, max_depth))
                print(num_read_samples)

                print('...samping reads...')
                print(('chrom %s pos %s' % (chrom, str(pos))))

                for s in range(num_read_samples):
                    input_tensor = create_input_tensor_for_position(chrom, pos, bamfile_n, bamfile_t, sample_reads=True)
                    pred_true = predict_position(input_tensor, model, channel_means, channel_stds)
                    pred_trues.append(pred_true)
            
                from numpy import median
                pred_true = median(pred_trues)
   
                print(pred_trues)
                print((median(pred_trues)))

        else:
#            try:
#                input_tensor = create_input_tensor_for_position(chrom, pos, bamfile_n, bamfile_t, ref_file)
#            except Exception as e:
#                print(('Exception in creating input tensor;', chrom, pos))
#                print(e)
#                continue
        
            if c.TEST_TIME_TRAINING:
                input_tensor = create_input_tensor_for_position(chrom, pos, bamfile_n, bamfile_t, ref_file)
                test_time_train(self_supervised_head, input_tensor, channel_means, channel_stds)
                pred_true = predict_position(input_tensor, classifier_head, channel_means, channel_stds)

            else:
                input_tensor = create_input_tensor_for_position(chrom, pos, bamfile_n, bamfile_t, ref_file)
                pred_true = predict_position(input_tensor, model, channel_means, channel_stds)

        results_dict = {'chrom': chrom, 'pos': pos, 'pred_true': pred_true}
        
        results = results.append(results_dict, ignore_index = True)

        if len(results) > 100:
            # append predictions to the file every 1000 predictions
            results.to_csv(output_path, sep='\t', index=False, encoding='utf-8', mode='a', header=False)
            results.drop(results.index, inplace=True)

    # write remaining results to file
    if len(results):
        results.to_csv(output_path, sep='\t', index=False, encoding='utf-8', mode='a', header=False)

    os.rename(output_path, output_path.replace('.temp.csv', '.csv'))
    print(("SNV Batch Complete:", batch_num))

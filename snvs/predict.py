import os
from time import time

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

from snvs.generate_training_data import get_reference, generate_image, populate_array, create_input_tensor_for_position, create_tumor_only_input_tensor_for_position, get_ref_base

CURRENT_DIR = os.path.dirname(__file__) 

def get_model(args, adapted=False):
    from tensorflow.keras.models import model_from_json

    if args.normal_bam and not args.ffpe:
        # frozen tumor-normal convnet2 and not ffpe
        BEST_MODEL_ARCHITECTURE_PATH = c.BEST_MODEL_ARCHITECTURE_PATH
        BEST_MODEL_PATH = c.BEST_MODEL_PATH
        BEST_MODEL_WEIGHTS_PATH = c.BEST_MODEL_WEIGHTS_PATH

        if c.TEST_TIME_TRAINING:
            from tensorflow.keras.models import load_model
            # for group normalization
            # import tensorflow_addons as tfa
            #model = model_from_json(model_architecture, custom_objects={'Addons>GroupNormalization': tfa.layers.GroupNormalization})
            model = load_model(os.path.join(CURRENT_DIR, BEST_MODEL_PATH), custom_objects={'Addons>GroupNormalization': tfa.layers.GroupNormalization})
            assert model.optimizer is not None

        elif adapted:
            # load adapted model
            from tensorflow.keras.models import load_model
            model = load_model(os.path.join(args.sample_folder, c.SNV_ADAPTED_TUMOR_NORMAL_MODEL))

        else:
            with open(os.path.join(CURRENT_DIR, BEST_MODEL_ARCHITECTURE_PATH)) as f:
                model_architecture = f.read()
            model = model_from_json(model_architecture)
            model.load_weights(os.path.join(CURRENT_DIR, BEST_MODEL_WEIGHTS_PATH))

    else:
        # frozen tumor-only or ffpe

        # <start> tumor-only convnet2
        if args.ffpe:
            raise Exception('which model to load?')
        elif adapted:
            raise Exception('which model to load?')
        else:
            BEST_MODEL_ARCHITECTURE_PATH = c.TUMOR_ONLY_BEST_MODEL_ARCHITECTURE_PATH
            BEST_MODEL_WEIGHTS_PATH = c.TUMOR_ONLY_BEST_MODEL_WEIGHTS_PATH

            with open(os.path.join(CURRENT_DIR, BEST_MODEL_ARCHITECTURE_PATH)) as f:
                model_architecture = f.read()
            model = model_from_json(model_architecture)
            model.load_weights(os.path.join(CURRENT_DIR, BEST_MODEL_WEIGHTS_PATH))
        # </start> tumor-only convnet2

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
        # <start/> transformer 

    return model

def predict_position(input_tensor, model, channel_means, channel_stds, training=False, args=None):
    if c.MULTIPLE_READ_SAMPLES:
        raise Exception("MULTIPLE_READ_SAMPLES is True. Batch prediction requires it to be False.")

    if channel_means is not None and channel_stds is not None:
        # normalization not required for tumor-only/FFPE transformers
        input_tensor -= channel_means
        input_tensor /= channel_stds

    if not training:
        # full batch prediction
        y_pred_test = model.predict(input_tensor, batch_size=len(input_tensor))
        
        if y_pred_test.shape[1] != 1:
            # multi-class output, use somatic proba only
            y_pred_test = y_pred_test[:,1]
            
        return y_pred_test
    else:
        # to update batch norm statistics
        y_pred_test = model(input_tensor, training=True)
        return y_pred_test

def predict_snvs(positions_to_predict, batch_num, args, snv_predictions_folder, output_path=None, update_batch_norm=False, adapted=False):
    print(("SNV PREDICTION BATCH:", batch_num))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.normal_bam:
        bamfile_n = pysam.AlignmentFile(args.normal_bam, "rb", check_sq=False) # normal bamfile
    else:
        # tumor-only mode
        bamfile_n = None
    
    bamfile_t = pysam.AlignmentFile(args.tumor_bam, "rb", check_sq=False) # tumor bamfile
    ref_file = get_ref_file(args.reference)    

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
                s = pline.strip().split()
                chrom, pos = s[0], s[1]
                pos_key = 'chrom%spos%s' % (chrom, pos)
                positions_completed[pos_key] = True

    model = get_model(args, adapted=adapted)
    
    assert model is not None
    
    if args.normal_bam and not args.ffpe:
        # tumor-normal frozen
        channel_means = np.load(os.path.join(CURRENT_DIR, c.NORMALIZATION_MEANS_PATH))
        channel_stds = np.load(os.path.join(CURRENT_DIR, c.NORMALIZATION_STD_DEVS_PATH))
    else:
        # tumor only frozen or ffpe

        # <load mean/std for tumor-only convnet trained on experiment 9>
        channel_means = np.load(os.path.join(CURRENT_DIR, c.NORMALIZATION_MEANS_PATH))
        channel_stds = np.load(os.path.join(CURRENT_DIR, c.NORMALIZATION_STD_DEVS_PATH))
        # </load mean/std for tumor-only convnet trained on experiment 9>

        # <load mean/std for tumor-only convnet encoding>
        # channel_means = np.load(os.path.join(CURRENT_DIR, c.TUMOR_ONLY_NORMALIZATION_MEANS_PATH))
        # channel_stds = np.load(os.path.join(CURRENT_DIR, c.TUMOR_ONLY_NORMALIZATION_STD_DEVS_PATH))
        # <load mean/std for tumor-only convnet encoding>
        
        # < set to None for transformer, no need normalization>
        # channel_means, channel_stds = None, None 
        # </ set to None for transformer, no need normalization>        

    positions_iterator = positions_to_predict.iterrows()
    positions = []

    for i, row in positions_iterator:
        positions.append((row['pos'], row['chrom'], row['REF'], row['ALT'], row['DP'], row['RO'], row['AO'], row['AF']))

    if args.update_batch_norm:
        update_batch_norm_fn(model, positions, bamfile_n, bamfile_t, channel_means, channel_stds, ref_file=ref_file, create_input_fn=create_input_tensor_for_position, predict_fn=predict_position)

    if c.TEST_TIME_TRAINING:
        classifier_head, self_supervised_head = ttt_get_models(model, lr=0.0001)
        import random
        random.shuffle(positions) # shuffle positions for TTT

    batch_size = args.batch_size
    for i in range(0, len(positions), batch_size):
        batch = positions[i:i + batch_size]
        
        batch_input_tensors = []
        batch_metadata = []

        for row in batch:
            pos, chrom, REF, ALT, DP, RO, AO, AF = row
            pos_key = 'chrom%spos%s' % (chrom, pos)

            if pos_key in positions_completed:
                continue

            if c.TEST_TIME_TRAINING:
                input_tensor = create_input_tensor_for_position(chrom, pos, bamfile_n, bamfile_t, ref_file)
                test_time_train(self_supervised_head, input_tensor, channel_means, channel_stds)
                pred_true = predict_position(input_tensor, classifier_head, channel_means, channel_stds, args=args)
                pred_true = float(pred_true[0])
                
                results_dict = {'chrom': chrom, 'pos': pos, 'REF': REF, 'ALT': ALT, 'DP': DP, 'RO': RO, 'AO': AO, 'AF': AF, 'pred_true': pred_true}
                results = results.append(results_dict, ignore_index=True)
            else:
                input_tensor = create_input_tensor_for_position(chrom, pos, bamfile_n, bamfile_t, ref_file)
                batch_input_tensors.append(input_tensor)
                batch_metadata.append(row)

        if not c.TEST_TIME_TRAINING and batch_input_tensors:
            input_tensor_batch = np.concatenate(batch_input_tensors, axis=0)
            preds = predict_position(input_tensor_batch, model, channel_means, channel_stds, args=args)
            
            for idx, row in enumerate(batch_metadata):
                pos, chrom, REF, ALT, DP, RO, AO, AF = row
                pred_true = float(preds[idx])
                results_dict = {'chrom': chrom, 'pos': pos, 'REF': REF, 'ALT': ALT, 'DP': DP, 'RO': RO, 'AO': AO, 'AF': AF, 'pred_true': pred_true}
                results = results.append(results_dict, ignore_index=True)

        if len(results):
            # append predictions to the file every batch
            results.to_csv(output_path, sep='\t', index=False, encoding='utf-8', mode='a', header=False)
            results.drop(results.index, inplace=True)

    # write remaining results to file
    if len(results):
        results.to_csv(output_path, sep='\t', index=False, encoding='utf-8', mode='a', header=False)

    os.rename(output_path, output_path.replace('.temp.csv', '.csv'))
    print(("SNV Batch Complete:", batch_num))

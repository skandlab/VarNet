import os
import numpy as np
import random

def get_ref_file(path):
    import pysam
    return pysam.FastaFile(path)

def sample_reads_fn(reads, n, seed=1):
    num_reads = len(reads)
    # don't sample if there are less than or equal to n reads
    if num_reads <=n: return reads
    # sample n items from reads list while preserving relative order of sampled reads in original list
    read_indices = list(range(num_reads)) # [0,1,2,3...len(reads)-1]
    random.seed(seed) # use same random seed for every site. easier to debug
    sample_indices = random.sample(read_indices, n) # sample n reads
    sample_indices.sort() # sort reads by order in original list
    output = [reads[_] for _ in sample_indices]
    return output

def update_batch_norm_fn(model, positions, bamfile_n, bamfile_t, channel_means, channel_stds, num_positions=1000, ref_file=None, create_input_fn=None, predict_fn=None):
    """
    1. Updates batch norm momentum to 0.5 for all batch norm layers in model.
    2. Forward pass 10k test points to update batch norm stats using training=True
    3. Assumes there are no dropout layers in the model as training=True will affect dropout in forward pass
    """
    # default momentum for updating running mean and variance is 0.99. changing it to 0.5 so they get updated faster in the test batch
    for layer in model.layers:
        if 'batch_normalization' in layer.name:
            layer.momentum = 0.95

    positions_copy = positions.copy()
    import random
    random.shuffle(positions_copy)
    positions_copy = positions_copy[:num_positions]


    batch_size = 128
    batches = np.array_split(positions_copy, int(len(positions_copy)/batch_size))

    for batch in batches:
        input_batch = None

        for i, row in enumerate(batch):
            pos = int(row[0])
            chrom = row[1]

            try:
                input_tensor = create_input_fn(chrom, pos, bamfile_n, bamfile_t, ref_file=ref_file)

                if input_batch is None:
                    input_batch = input_tensor
                else:
                    input_batch = np.concatenate((input_batch, input_tensor), axis=0)

            except Exception as e:
                print(('Exception in creating input tensor;', chrom, pos))
                print(e)
                continue

        predict_fn(input_batch, model, channel_means, channel_stds, training=True)

    print('Updated BatchNorm statistics...')

def tf_log2(x):
    import tensorflow as tf
    return tf.math.log(x) / tf.cast(tf.math.log(2.), dtype=tf.float64)

def entropy_loss_fn(y_true, y_pred):
    # shannon entropy of predictions. ignores y_true
    import tensorflow as tf
    entropy =  -(tf_log2(y_pred)*y_pred + (1-y_pred) * tf_log2(1-y_pred))
    return tf.reduce_mean(entropy)  # Note the `axis=-1`

def get_tent_model(model):
    """
    Based on TENT (test time entropy): https://openreview.net/forum?id=uXl3bZLkr3c
    """
    # freeze all layers except batch norm. important to call model.compile after freezing and updating layers
    for layer in model.layers:
        if 'batch_normalization' not in layer.name:
            layer.trainable = False
        else:
            bn_weights = layer.get_weights()
            assert len(bn_weights) == 4 # [ gamma, beta, moving_mean, moving_variance]

            # reset mean and variance to 0 and 1
            current_mean, current_variance = bn_weights[2], bn_weights[3]
            bn_weights[2] = np.zeros(current_mean.shape)
            bn_weights[3] = np.ones(current_variance.shape)

    # recompile the model with entropy loss function
    model.compile(loss= entropy_loss_fn, optimizer = model.optimizer)
    return model

def train_tent_model(model, input_tensor, channel_means, channel_stds):
    """
    input expected to be (1, H, W, C)
    """
    input_tensor = input_tensor.copy() # don't modify the input arg as it is used downstream

    input_tensor -= channel_means
    input_tensor /= channel_stds

    batch = np.append(input_tensor, input_tensor.copy(), axis=0)
    assert batch.shape == (2, input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3])

    batch, labels = rotate_batch(batch, testing=True) # returns batch of one unrotated and one rotated sample
    assert np.isclose(model.optimizer.get_config()['learning_rate'], 0.0001)
    # assert np.isclose(model.optimizer.get_config()['learning_rate'], 0.0001) # for snv model
    model.train_on_batch(batch, labels) # learning rate set in ttt_get_models()

def ttt_get_models(model, lr=None):
    """
    lr used to compile self-supervised head for test-time training
    This must be equal to the last lr used in training (Test Time Training by Sun et al, ICML 2020)
    0.0001 is for snv model. for indel model it is different
    """
    # Define the classifier sub model
    from tensorflow.keras.models import Model
    assert model.layers[0].name == 'classification_input'
    assert model.layers[-2].name == 'classification_output'

    classifier_head = Model(inputs=model.layers[0].output, outputs=model.layers[-2].output)

    # Define the self-sup sub model
    assert model.layers[1].name == 'self_supervised_input'
    assert model.layers[-1].name == 'self_supervised_output'

    self_supervised_head = Model(inputs=model.layers[1].output, outputs=model.layers[-1].output)

    # don't compile again. the model weights should have optimizer state already
    #from tensorflow.keras import optimizers
    #adam = optimizers.Adam(lr=lr)

    # compile for ttt
    # compile with overall model's optimizer - only applicable if you loaded model with load_model
    self_supervised_head.compile(loss= 'binary_crossentropy', optimizer = model.optimizer)

    return classifier_head, self_supervised_head

def test_time_train(model, input_tensor, channel_means, channel_stds):
    """
    model must be self-supervised head
    input expected to be (1, H, W, C)
    """
    from snvs.augment_training_data import rotate_batch

    input_tensor = input_tensor.copy() # don't modify the input arg as it is used downstream

    input_tensor -= channel_means
    input_tensor /= channel_stds

    batch = np.append(input_tensor, input_tensor.copy(), axis=0)
    assert batch.shape == (2, input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3])

    batch, labels = rotate_batch(batch, testing=True) # returns batch of one unrotated and one rotated sample
    assert np.isclose(model.optimizer.get_config()['learning_rate'], 0.0001)
    # assert np.isclose(model.optimizer.get_config()['learning_rate'], 0.0001) # for snv model
    model.train_on_batch(batch, labels) # learning rate set in ttt_get_models()

###### TESTING ######

def run_benchmark(test_set, benchmark_name, mutation='', model_name='', exp_id=None):
    from tensorflow.keras.models import load_model
    from sklearn.metrics import precision_recall_curve
    from snvs.train import normalize # can use for both snv and indel
    import gc

    if mutation == 'snvs':
        from snvs import constants as c
    elif mutation == 'indels':
        from indels import constants as c
    else:
        raise Exception('')

    if exp_id is None:
        exp_id = c.DEFAULT_EXPERIMENT_ID

    c.set_experiment_paths(exp_id)

    channel_means, channel_std_deviations = np.load(os.path.join(c.CURRENT_EXPERIMENT_FOLDER, c.channels_means_file)), np.load(os.path.join(c.CURRENT_EXPERIMENT_FOLDER, c.channels_std_devs_file))
    X, Y = normalize(test_set, channel_means, channel_std_deviations)

    ###
    from snvs.compress_npy_helper import save_compressed_npy
    o=os.path.join(c.CURRENT_EXPERIMENT_FOLDER, os.path.basename(test_set))
    print(o)
    save_compressed_npy(o, X, Y)
    return
    ###


    if c.CURRENT_EXPERIMENT_ID == 23:# or c.CURRENT_EXPERIMENT_ID == 9:
        # add custom coordconv layer when loading model
        import sys
        sys.path.append('/home/kiran/keras-coordconv')
        from coord import CoordinateChannel2D
        model = load_model(c.CURRENT_BEST_MODEL_PATH, custom_objects= { 'CoordinateChannel2D': CoordinateChannel2D })
    else:
#        model = load_model(os.path.join(c.CURRENT_EXPERIMENT_FOLDER, 'trained_models/model.SWA.hdf5'))
        print("Loading model", c.CURRENT_BEST_MODEL_PATH)
        model = load_model(c.CURRENT_BEST_MODEL_PATH)
        #model = load_model(os.path.join(c.CURRENT_FINETUNED_MODELS_DIR, benchmark_name, 'model.finetuned.%s.hdf5' % model_name))

    # Split into batches for memory
    pred = np.ones((0,1)) # just empty inital value
    num_splits = 10
    x_splits = np.array_split(X, num_splits)

    del X
    gc.collect()

    for x in x_splits:
        pred = np.append(pred, model.predict(x), axis=0)

    print(("Performance on %s" % test_set))
    print(("Mutations: %d, Overall Positions: %d" % (np.count_nonzero(Y), Y.shape[0])))

    precisions, recalls, thresholds = precision_recall_curve(np.round(Y), pred)
    f1s = [(2*precisions[i]*recalls[i])/(precisions[i] + recalls[i]) for i, val in enumerate(precisions)]
    optimal_f1, optimal_precision, optimal_recall, optimal_threshold = max(f1s), precisions[np.argmax(f1s)], recalls[np.argmax(f1s)], thresholds[np.argmax(f1s)]

    optimum = [optimal_f1, optimal_precision, optimal_recall, optimal_threshold]

    benchmark_file = os.path.join(c.CURRENT_EXPERIMENT_FOLDER, 'benchmark_%s' % (benchmark_name))
    np.save(benchmark_file, [ optimum, precisions, recalls, thresholds ])
    print(('Saved ', benchmark_file))

def run_tests(mutation_type, exp_id=None):
    test_sets_path = '/media/nvme/kiran/smudl_training_data/test_sets'
    benchmarks = ['icgc_cll', 'icgc_mbl', 'tgen'] + [ 'dream%d' % _ for _ in range(1,6) ]

    assert mutation_type in ['snvs', 'indels']

    for benchmark in benchmarks:
        benchmark_name = benchmark + '_%s_test_set' % mutation_type
        test_file = os.path.join(test_sets_path, benchmark_name + '.npz')

        if not os.path.exists(test_file):
            print(test_file, 'not found')
            continue

        print(test_file, benchmark_name)
        run_benchmark(test_file, benchmark_name, mutation=mutation_type, exp_id=exp_id)

def get_constants_as_dict(constants):
    d = vars(constants) # convert constants module to dict
    o={}
    import json

    # excludes non-serializable objects
    for _ in d.keys():
        try:
            json.dumps(d[_])
            o[_]=d[_]
        except TypeError:
            pass

    return o

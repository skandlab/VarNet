import os
from os.path import expanduser
from enum import Enum, unique

@unique
class User(Enum):
    DYLAN_AQUILA = 0
    DYLAN_NSCC = 1
    DYLAN_WORKSTATION = 2
    ALEX_AQUILA = 3
    ALEX_NSCC = 4
    ALEX_WORKSTATION = 5
    KIRAN_NSCC = 6
    KIRAN_WORKSTATION = 7
    KIRAN_AQUILA = 8

@unique
class Cluster(Enum):
    AQUILA = 0
    NSCC = 1
    WORKSTATION = 2

HOME_DIRECTORY = expanduser('~')

cluster, user = None, None

if HOME_DIRECTORY == '/home/delu':
    path_to_smudl = '/home/delu/smudl/'
    user = User.DYLAN_AQUILA
    cluster = Cluster.AQUILA

elif HOME_DIRECTORY == '/home/users/astar/gis/delu':
    path_to_smudl = '/home/users/astar/gis/delu/scratch/smudl/' # '/home/users/astar/gis/krishnak/scratch/smudl/'
    user = User.DYLAN_NSCC
    cluster = Cluster.NSCC

elif HOME_DIRECTORY == '/home/aswift-scott':
    path_to_smudl = '/home/aswift-scott/repos/smudl/'
    user = User.ALEX_AQUILA
    cluster = Cluster.AQUILA

elif HOME_DIRECTORY == '/home/users/astar/gis/aswiftsc':
    path_to_smudl = '/home/users/astar/gis/aswiftsc/repos/smudl/'
    user = User.ALEX_NSCC
    cluster = Cluster.NSCC

elif HOME_DIRECTORY == '/home/dylan':
    path_to_smudl = '/home/dylan/smudl/'
    user = User.DYLAN_WORKSTATION
    cluster = Cluster.WORKSTATION

elif HOME_DIRECTORY == '/home/alex':
    path_to_smudl = '/home/alex/smudl/'
    user = User.ALEX_WORKSTATION
    cluster = Cluster.WORKSTATION

elif HOME_DIRECTORY == '/home/kiran':
    path_to_smudl = '/home/kiran/smudl'
    user = User.KIRAN_WORKSTATION
    cluster = Cluster.WORKSTATION

elif HOME_DIRECTORY == '/home/users/astar/gis/krishnak':
    path_to_smudl = '/home/users/astar/gis/krishnak/scratch/smudl/'
    user = User.KIRAN_NSCC
    cluster = Cluster.NSCC

elif HOME_DIRECTORY == '/home/krishnak':
    path_to_smudl = '/home/krishnak/smudl/'
    user = User.KIRAN_AQUILA
    cluster = Cluster.AQUILA


else:
    import sys
    # print 'User is not in constants file, exiting'
    # sys.exit(0)

# === Data Locations === #
if cluster == Cluster.AQUILA:
    ref_path = '/mnt/projects/huangwt/wgs/genomes/seq/GRCh37.fa'
    liver_patients_smurf_predictions = '/home/delu/SMURF_INDEL_PREDICTIONS/LIVER/'
    liver_patients_root_folder = '/mnt/projects/skanderupamj/wgs/data/training/ready.bams/liver/'
    liver_patients_normal_bam_file_path = liver_patients_root_folder + '%s-N-ready.bam'
    liver_patients_tumor_bam_file_path = liver_patients_root_folder + '%s-T-ready.bam'
    lung_patients_smurf_predictions = '/home/delu/SMURF_INDEL_PREDICTIONS/LUNG/'
    lung_bam_files_root_folder = '/mnt/projects/skanderupamj/wgs/data/training/ready.bams/lung/'
    lung_patients_normal_bam_file_path = lung_bam_files_root_folder + '%s-N-ready.bam'
    lung_patients_tumor_bam_file_path = lung_bam_files_root_folder + '%s-T-ready.bam'
    sarcoma_patients_smurf_predictions = '/home/delu/SMURF_INDEL_PREDICTIONS/SARCOMA/'
    sarcoma_bam_files_root_folder = '/mnt/projects/skanderupamj/wgs/data/training/ready.bams/cfdna/'
    sarcoma_patients_normal_bam_file_path = sarcoma_bam_files_root_folder + '%s-N-ready.bam'
    sarcoma_patients_tumor_bam_file_path = sarcoma_bam_files_root_folder + '%s-T-ready.bam'
    thyroid_patients_smurf_predictions = '/home/delu/SMURF_INDEL_PREDICTIONS/THYROID/'
    thyroid_bam_files_root_folder = '/mnt/projects/skanderupamj/wgs/data/training/ready.bams/thyroid/'
    thyroid_patients_normal_bam_file_path = thyroid_bam_files_root_folder + '%s-N-ready.bam'
    thyroid_patients_tumor_bam_file_path = thyroid_bam_files_root_folder + '%s-T-ready.bam'

    goldset_files = [

        ('icgc_cll',
            '/mnt/projects/delu/DeepLearning/candidate_positions_2/g/icgc_cll.csv',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-N.bam',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-T.bam'),
        ('icgc_mbl',
            '/mnt/projects/delu/DeepLearning/candidate_positions_2/g/icgc_mbl.csv',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-N.bam',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-T.bam'),
        ('icgc_cll-T20',
            '/mnt/projects/delu/DeepLearning/candidate_positions_2/g-T20/icgc_cll.csv',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-N.bam',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-T20.bam'),
        ('icgc_cll-T40',
            '/mnt/projects/delu/DeepLearning/candidate_positions_2/g-T40/icgc_cll.csv',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-N.bam',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-T40.bam'),
        ('icgc_cll-N30',
            '/mnt/projects/delu/DeepLearning/candidate_positions_2/g-N30/icgc_cll.csv',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-N30.bam',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-T.bam'),
        ('icgc_mbl-T20',
            '/mnt/projects/delu/DeepLearning/candidate_positions_2/g-T20/icgc_mbl.csv',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-N.bam',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-T20.bam'),
        ('icgc_mbl-T40',
            '/mnt/projects/delu/DeepLearning/candidate_positions_2/g-T40/icgc_mbl.csv',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-N.bam',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-T40.bam'),
        ('icgc_mbl-N30',
            '/mnt/projects/delu/DeepLearning/candidate_positions_2/g-N30/icgc_mbl.csv',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-N30.bam',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-T.bam'),
        ('icgc_cll-smurf',
            '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_cll/indel-allpredictions.txt',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-N.bam',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-T.bam',
            '/mnt/projects/aswift-scott/indels/patients_for_prefiltering/icgc_cll/icgc_cll_ncomms.txt'),
        ('icgc_mbl-smurf',
            '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_mbl/indel-allpredictions.txt',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-N.bam',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-T.bam',
            '/mnt/projects/aswift-scott/indels/patients_for_prefiltering/icgc_mbl/icgc_mbl_ncomms.txt'),
        ('icgc_cll-T20-smurf',
            '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_cll_T20/indel-allpredictions.txt',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-N.bam',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-T20.bam',
            '/mnt/projects/aswift-scott/indels/patients_for_prefiltering/icgc_cll/icgc_cll_ncomms.txt'),
        ('icgc_cll-T40-smurf',
            '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_cll_T40/indel-allpredictions.txt',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-N.bam',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-T40.bam',
            '/mnt/projects/aswift-scott/indels/patients_for_prefiltering/icgc_cll/icgc_cll_ncomms.txt'),
        ('icgc_cll-N30-smurf',
            '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_cll_N30/indel-allpredictions.txt',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-N30.bam',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-T.bam',
            '/mnt/projects/aswift-scott/indels/patients_for_prefiltering/icgc_cll/icgc_cll_ncomms.txt'),
        ('icgc_mbl-T20-smurf',
            '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_mbl_T20/indel-allpredictions.txt',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-N.bam',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-T20.bam',
            '/mnt/projects/aswift-scott/indels/patients_for_prefiltering/icgc_mbl/icgc_mbl_ncomms.txt'),
        ('icgc_mbl-T40-smurf',
            '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_mbl_T40/indel-allpredictions.txt',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-N.bam',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-T40.bam',
            '/mnt/projects/aswift-scott/indels/patients_for_prefiltering/icgc_mbl/icgc_mbl_ncomms.txt'),
        ('icgc_mbl-N30-smurf',
            '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_mbl_N30/indel-allpredictions.txt',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-N30.bam',
            '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-T.bam',
            '/mnt/projects/aswift-scott/indels/patients_for_prefiltering/icgc_mbl/icgc_mbl_ncomms.txt')

    ]


elif cluster == Cluster.NSCC:
    ref_path = '/seq/astar/gis/projects/skandera/SMUDL/GRCh37.fa'
    crc_patients_smurf_predictions = '/seq/astar/gis/projects/skandera/SMUDL/SMURF_CRC_INDEL_PREDICTIONS/'
    crc_patients_normal_bam_file_path = '/seq/astar/gis/projects/skandera/training/wgs/colorectal/%s-N-ready.bam'
    crc_patients_tumor_bam_file_path = '/seq/astar/gis/projects/skandera/training/wgs/colorectal/%s-T-ready.bam'
    gastric_patients_smurf_predictions = '/seq/astar/gis/projects/skandera/SMUDL/SMURF_GASTRIC_INDEL_PREDICTIONS/'
    gastric_patients_normal_bam_file_path = '/seq/astar/gis/projects/skandera/training/wgs/gastric/%s-N-ready.bam'
    gastric_patients_tumor_bam_file_path = '/seq/astar/gis/projects/skandera/training/wgs/gastric/%s-T-ready.bam'
    lymphoma_patients_smurf_predictions = '/seq/astar/gis/projects/skandera/SMUDL/SMURF_LYMPHOMA_INDEL_PREDICTIONS/'
    lymphoma_patients_normal_bam_file_path = '/seq/astar/gis/projects/skandera/training/wgs/lymphoma/%s-N-ready.bam'
    lymphoma_patients_tumor_bam_file_path = '/seq/astar/gis/projects/skandera/training/wgs/lymphoma/%s-T-ready.bam'

# training data paths
if user == User.ALEX_AQUILA:
    training_data_folder = '/mnt/projects/aswift-scott/indels/training_data/'

elif user == User.ALEX_NSCC:
    training_data_folder = '/home/users/astar/gis/aswiftsc/scratch/training_data/'

elif user == User.DYLAN_AQUILA:
    training_data_folder = '/mnt/projects/delu/DeepLearning/SMURF_TRAINING_SET/'
    predictions_folder = '/mnt/projects/delu/DeepLearning/smudl_predictions/'
    experiments_folder = '/mnt/projects/delu/DeepLearning/experiments/'

elif user == User.DYLAN_NSCC:
    training_data_folder = '/home/users/astar/gis/krishnak/scratch/smudl_training_data/indels' #'/home/users/astar/gis/delu/scratch/training_data_folder/'
    predictions_folder = '/scratch/users/astar/gis/delu/smudl_predictions/'
    experiments_folder = '/scratch/users/astar/gis/delu/experiments/'

elif user == User.ALEX_AQUILA:
    training_data_folder = '/mnt/projects/aswift-scott/indels/training_data/'
    experiments_folder = '/media/sda4/dylan/experiments/'

elif user == User.DYLAN_WORKSTATION:
    training_data_folder = '/media/sda4/dylan/training_data_folder/'
    experiments_folder = '/media/nvme/smudl_trained_models/indels'

elif user == User.ALEX_WORKSTATION:
    experiments_folder = '/media/sda4/dylan/experiments/'

elif user == User.KIRAN_WORKSTATION:
    experiments_folder = '/media/nvme/kiran/smudl_trained_models/indels'

elif user == User.KIRAN_NSCC:
    training_data_folder = '/home/users/astar/gis/krishnak/scratch/smudl_training_data/indels'
    predictions_folder = '/scratch/users/astar/gis/krishnak/smudl_predictions/indels'
    experiments_folder = '/scratch/users/astar/gis/krishnak/smudl_trained_models/indels'

elif user == User.KIRAN_AQUILA:
    training_data_folder = '/mnt/projects/krishnak/kiran/smudl_training_data/indels/'
    predictions_folder = '/mnt/projects/krishnak/kiran/smudl_predictions/indels'
    experiments_folder = '/mnt/projects/krishnak/kiran/smudl_trained_models/indels/'

######### PREDICTION PARAMS #########
SAMPLE_READS = True # sample reads and sort by CO
VARIABLE_INPUT_SIZE = False

###### TRAINING PARAMS #########
SPECTRAL_DECOUPLING = False # default l2 reg with weight 2e-05
SD_COEFF = 2e-06 # SPECTRAL_DECOUPLING lambda
SAMPLE_WEIGHTING = False # True
ADVERSARIAL_TRAINING = False
OPTIMIZER = 'adam' # 'adam', 'sgd'
LR_SCHEDULE = True # False
LOSS = 'binary_crossentropy' # 'mean_squared_error' # 

SAMPLE_READS = True # sample reads and sort by CO. don't use this when generating training data if you don't want sampling, or set SAMPLE_READS_COUNT to 1.
SAMPLE_READS_COUNT = 1 # default 1. for use with SAMPLE_READS

VARIABLE_INPUT_SIZE = False
SAVE_LATEST_CHECKPOINT = False

INPUT_WHITENING = False # decorrelate input pixels per channel
WHITENING_MATRIX = 'training_data_whitening_matrix.npy' # computed using 10k random training samples. see smudl/utils.py -> compute_whitening_matrix_torch

mixup_alpha = 0.5 # --mixup command line arg must be passed to train.py. alpha < 1 is not much mixing, concentrated close to 0 and 1 (alpha=0 is no mixing); alpha=1 is uniform in [0,1]; alpha -> infinity is concentrated at 0.5 (equal mixing)

architecture = 'convnet2_retrain_subset' # 'inceptionv3_retrain_subset'#'inceptionv3_input_whitening'# 'mixup_%s' % mixup_alpha # 'inceptionv3_train_on_val.ood_val_set'# 'inceptionv3_train_on_val' # 'inceptionv3_mean_squared_error' # 'EMA.inceptionv3_spectral_decoupling_sd_coeff_%s' % SD_COEFF # 'inceptionv3' # 'inceptionv3_fgsm_rand_eps_0.25' #'convnet2_unaugmented_batch_size_32_sample_weights' # 'inceptionv3_unaugmented_sample_weight_crc_0_1_batch_size_32' # 'inceptionv3_unaugmented_batch_size_256' #'inceptionv3_unaugmented_sample_weight_crc_0.5_batch_size_32' # 'inceptionv3_data_augmented_batch_size_256' # 'convnet2_dropout_memory' # 'EfficientNetB0'

all_data_folder_name = 'all_data'
normalized_training_data_folder_name = 'normalized_training_data'
save_models_to_folder = 'trained_models'
tensorboard_log_directory = 'Tensorboard_logs'
validation_f1_score_history_file = 'validation_f1_score_history.%s.npy' % architecture
validation_precision_history_file = 'validation_precision_history.%s.npy' % architecture
validation_recall_history_file = 'validation_recall_history.%s.npy' % architecture
experiment_details_file = 'experiment_details.txt'
training_filenames = 'list_of_files_used_for_training.npy'
validation_filenames = 'list_of_files_used_for_validation.npy'
normalized_filenames = 'list_of_normalized_files.npy'

# save latest model
model_latest = 'model.latest.hdf5'
model_latest_arch = 'model.latest.architecture.json'
model_latest_weights = 'model.latest.weights.hdf5'

best_model_name = 'model.best.%s.hdf5' % architecture
best_model_architecture = 'model.best.architecture.%s.json' % architecture # architecture only of the best model as a json file
best_model_weights = 'model.best.weights.%s.hdf5' % architecture # weights only
initial_model_name = 'model.initial.%s.hdf5' % architecture # initial model to continue training
channels_means_file = 'channel_means_for_normalization.npy'
channels_std_devs_file = 'channel_standard_deviations_for_normalization.npy'
shuffled_training_data_folder = 'shuffled_training_data'
shuffled_data_indices = 'shuffled_data_indices.npy'
combined_predictions_file = 'Predictions.csv'

config_file = 'config_%s.npy' % architecture # save constants file as dictionary

BALANCE_CANCER_TYPES_IN_TRAINING = False
shuffled_batch_size = 10000
EPOCHS_TO_TRAIN = 200
reg_batch_size = 30000
TRAINING_BATCH_SIZE = 32 # 32, 64, 128, 256
FULLY_VERBOSE = False

STOCHASTIC_WEIGHT_AVERAGING = False # keep an swa copy of the model
SWA_WEIGHTS_MODEL = 'model.SWA.%s.hdf5' % architecture

EXPONENTIAL_MOVING_AVERAGE = False # keep an ema copy of the model
EMA_WEIGHTS_MODEL = 'model.EMA.%s.hdf5' % architecture

""" INDEL MODEL """
indel_model_folder = 'indel_model'
BEST_MODEL_ARCHITECTURE_PATH = os.path.join(indel_model_folder, 'model.best.architecture.json')
BEST_MODEL_WEIGHTS_PATH = os.path.join(indel_model_folder, 'model.best.weights.hdf5')
NORMALIZATION_MEANS_PATH = os.path.join(indel_model_folder, channels_means_file)
NORMALIZATION_STD_DEVS_PATH = os.path.join(indel_model_folder, channels_std_devs_file)

## CURRENT EXPERIMENT
def set_experiment_paths(experiment_id):
    global CURRENT_EXPERIMENT_ID
    CURRENT_EXPERIMENT_ID = experiment_id

    global experiment_name
    experiment_name = 'experiment_%s' % str(CURRENT_EXPERIMENT_ID)

    global CURRENT_EXPERIMENT_FOLDER
    CURRENT_EXPERIMENT_FOLDER = os.path.join(experiments_folder, experiment_name)

    global CURRENT_BEST_MODEL_PATH
    CURRENT_BEST_MODEL_PATH = os.path.join(CURRENT_EXPERIMENT_FOLDER, save_models_to_folder, best_model_name)

    global CURRENT_BEST_MODEL_ARCHITECTURE_PATH
    CURRENT_BEST_MODEL_ARCHITECTURE_PATH = os.path.join(CURRENT_EXPERIMENT_FOLDER, save_models_to_folder, best_model_architecture)

    global CURRENT_BEST_MODEL_WEIGHTS_PATH
    CURRENT_BEST_MODEL_WEIGHTS_PATH = os.path.join(CURRENT_EXPERIMENT_FOLDER, save_models_to_folder, best_model_weights)

    global CURRENT_NORMALIZATION_MEANS_PATH
    CURRENT_NORMALIZATION_MEANS_PATH = os.path.join(CURRENT_EXPERIMENT_FOLDER, channels_means_file)

    global CURRENT_NORMALIZATION_STD_DEVS_PATH
    CURRENT_NORMALIZATION_STD_DEVS_PATH = os.path.join(CURRENT_EXPERIMENT_FOLDER, channels_std_devs_file)

DEFAULT_EXPERIMENT_ID = 2#1


TUMOR_NORMAL_ADJACENT = True
SEQ_LENGTH = 75 # length of sequence. must be odd
FLANK = int((SEQ_LENGTH-1)/2)
NUM_READS = 140
NUM_CHANNELS_PER_IMAGE = 4

MAX_READS = 500 # max reads to use if all reads are used during prediction (when VARIABLE_INPUT_SIZE=True)

if TUMOR_NORMAL_ADJACENT:
    NUM_CHANNELS = NUM_CHANNELS_PER_IMAGE + 1
    INPUT_SHAPE = [ NUM_READS, 2*(SEQ_LENGTH), NUM_CHANNELS]
else:
    NUM_CHANNELS = 2 * NUM_CHANNELS_PER_IMAGE + 1
    INPUT_SHAPE = [NUM_READS, SEQ_LENGTH, NUM_CHANNELS]

encoding_name = 'NUM_READS_%s_SEQ_LENGTH_%s_NUM_CHANNELS_%s_NUM_CHANNELS_PER_IMG_%s' % \
                (str(NUM_READS), str(SEQ_LENGTH), str(NUM_CHANNELS), str(NUM_CHANNELS_PER_IMAGE) )

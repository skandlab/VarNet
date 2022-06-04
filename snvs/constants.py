import os

from datetime import datetime

# TRAINING DATA

# 43 liver patients
liver_patients_smurf_predictions = '/mnt/projects/krishnak/kiran/SMURF_LIVER_PREDICTIONS'
liver_patients_root_folder = '/mnt/projects/skanderupamj/wgs/data/training/ready.bams/liver/'
liver_patients_normal_bam_file_path = os.path.join(liver_patients_root_folder, '%s-N-ready.bam')
liver_patients_tumor_bam_file_path = os.path.join(liver_patients_root_folder, '%s-T-ready.bam')

# 164 crc patients
crc_patients_smurf_predictions = '/seq/astar/gis/projects/skandera/SMUDL/SMURF_CRC_PREDICTIONS'
crc_patients_smurf_indel_predictions = '/seq/astar/gis/projects/skandera/SMUDL/SMURF_CRC_INDEL_PREDICTIONS'
crc_patients_normal_bam_file_path = '/seq/astar/gis/projects/skandera/training/wgs/colorectal/%s-N-ready.bam'
crc_patients_tumor_bam_file_path = '/seq/astar/gis/projects/skandera/training/wgs/colorectal/%s-T-ready.bam'

# 38 gastric patients
gastric_patients_smurf_predictions = '/seq/astar/gis/projects/skandera/SMUDL/SMURF_GASTRIC_PREDICTIONS/'
gastric_patients_smurf_indel_predictions = '/seq/astar/gis/projects/skandera/SMUDL/SMURF_GASTRIC_INDEL_PREDICTIONS/'
gastric_patients_normal_bam_file_path = '/seq/astar/gis/projects/skandera/training/wgs/gastric/%s-N-ready.bam'
gastric_patients_tumor_bam_file_path = '/seq/astar/gis/projects/skandera/training/wgs/gastric/%s-T-ready.bam'

# 22 lung patients
lung_patients_smurf_predictions = '/mnt/projects/krishnak/kiran/SMURF_LUNG_PREDICTIONS'
lung_bam_files_root_folder = '/home/skanderupamj/projects/wgs/data/luad/'
lung_bam_files_root_folder = '/mnt/projects/skanderupamj/wgs/data/training/ready.bams/lung/'
lung_patients_normal_bam_file_path = os.path.join(lung_bam_files_root_folder, '%s-N-ready.bam')
lung_patients_tumor_bam_file_path = os.path.join(lung_bam_files_root_folder, '%s-T-ready.bam')

# 23 sarcoma patients
sarcoma_patients_smurf_predictions = '/mnt/projects/krishnak/kiran/SMURF_SARCOMA_PREDICTIONS/'
sarcoma_bam_files_root_folder = '/mnt/projects/skanderupamj/wgs/data/training/ready.bams/cfdna/'
sarcoma_patients_normal_bam_file_path = os.path.join(sarcoma_bam_files_root_folder, '%s-N-ready.bam')
sarcoma_patients_tumor_bam_file_path = os.path.join(sarcoma_bam_files_root_folder, '%s-T-ready.bam')

# 6 thyroid patients
thyroid_patients_smurf_predictions = '/mnt/projects/krishnak/kiran/SMURF_THYROID_PREDICTIONS'
thyroid_bam_files_root_folder = '/mnt/projects/skanderupamj/wgs/data/training/ready.bams/thyroid/'
thyroid_patients_normal_bam_file_path = os.path.join(thyroid_bam_files_root_folder, '%s-N-ready.bam')
thyroid_patients_tumor_bam_file_path = os.path.join(thyroid_bam_files_root_folder, '%s-T-ready.bam')

# 60 lymphoma patients
lymphoma_patients_smurf_predictions = '/seq/astar/gis/projects/skandera/SMUDL/SMURF_LYMPHOMA_PREDICTIONS'
lymphoma_patients_smurf_indel_predictions = '/seq/astar/gis/projects/skandera/SMUDL/SMURF_LYMPHOMA_INDEL_PREDICTIONS'
lymphoma_patients_normal_bam_file_path = '/seq/astar/gis/projects/skandera/training/wgs/lymphoma/%s-N-ready.bam'
lymphoma_patients_tumor_bam_file_path = '/seq/astar/gis/projects/skandera/training/wgs/lymphoma/%s-T-ready.bam'

BALANCE_CANCER_TYPES_IN_TRAINING = False

def get_crc_patient_bam_file_paths(sample_name):
    with open(crc_patients_bam_info_file) as f:
        for line in f:
            if sample_name in line:
                split_line = line.split('\t')
                normal_bam_file, tumor_bam_file = split_line[4], split_line[2]
                return (os.path.join(crc_patients_bam_files_root_path, normal_bam_file), os.path.join(crc_patients_bam_files_root_path, tumor_bam_file))

def get_gastric_patient_bam_file_paths(sample_name):
    gastric_bams = [ x for x in os.listdir(gastric_bam_files_root_folder) if x.endswith('.bam') ]
    normal_bam_file, tumor_bam_file = None, None

    for bam in gastric_bams:
        if 'N' + sample_name in bam:
            normal_bam_file = os.path.join(gastric_bam_files_root_folder, bam)
        if 'T' + sample_name in bam:
            tumor_bam_file = os.path.join(gastric_bam_files_root_folder, bam)

    assert normal_bam_file != None and tumor_bam_file != None

    print(("Sample name %s" % sample_name))
    print(("Normal BAM %s" % normal_bam_file))
    print(("Tumor BAM %s" % tumor_bam_file))

    return (normal_bam_file, tumor_bam_file)

def check_if_crc_patient_has_indexed_bam(sample_name):
    normal_bam, tumor_bam = get_crc_patient_bam_file_paths(sample_name)
    return os.path.exists(normal_bam + '.bai') and os.path.exists(tumor_bam + '.bai')

goldset_files = [

('icgc_cll', '/home/krishnak/smudl/goldset_files/truth1_positions.txt', '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_cll/snv-allpredictions.txt', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-N.bam', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-T.bam'),

('icgc_mbl', '/home/krishnak/smudl/goldset_files/truth2_positions.txt', '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_mbl/snv-allpredictions.txt', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-N.bam', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-T.bam'),

('icgc_cll-T40', '/home/krishnak/smudl/goldset_files/truth1_positions.txt', '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_cll_T40/snv-allpredictions.txt', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-N.bam', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-T40.bam'),

('icgc_mbl-T40', '/home/krishnak/smudl/goldset_files/truth2_positions.txt', '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_mbl_T40/snv-allpredictions.txt', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-N.bam', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-T40.bam'),

('icgc_cll-T20', '/home/krishnak/smudl/goldset_files/truth1_positions.txt', '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_cll_T20/snv-allpredictions.txt', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-N.bam', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-T20.bam'),

('icgc_mbl-T20', '/home/krishnak/smudl/goldset_files/truth2_positions.txt', '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_mbl_T20/snv-allpredictions.txt', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-N.bam', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-T20.bam'),

('icgc_cll-N30X', '/home/krishnak/smudl/goldset_files/truth1_positions.txt', '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_cll_N30/snv-allpredictions.txt', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-N30.bam', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_cll-T.bam'),

('icgc_mbl-N30X', '/home/krishnak/smudl/goldset_files/truth2_positions.txt', '/mnt/projects/huangwt/wgs/Results-SMuRF/Real-bcbio103-samples/2015-07-31_icgc_mbl_N30/snv-allpredictions.txt', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-N30.bam', '/mnt/projects/huangwt/wgs/Real-Data-v1.0.3/bam/icgc_mbl-T.bam'),

]

goldset_files_on_nscc = [

('icgc_cll', '/home/users/astar/gis/krishnak/scratch/smudl/goldset_files/truth1_positions.txt', '/home/users/astar/gis/krishnak/scratch/SMURF_GOLDSET_PREDICTIONS/icgc_cll', '/home/users/astar/gis/krishnak/scratch/icgc_cll-N.bam', '/home/users/astar/gis/krishnak/scratch/icgc_cll-T.bam'),

('icgc_mbl', '/home/users/astar/gis/krishnak/scratch/smudl/goldset_files/truth2_positions.txt', '/home/users/astar/gis/krishnak/scratch/SMURF_GOLDSET_PREDICTIONS/icgc_mbl', '/home/users/astar/gis/krishnak/scratch/icgc_mbl-N.bam', '/home/users/astar/gis/krishnak/scratch/icgc_mbl-T.bam'),

('icgc_cll-T40', '/home/users/astar/gis/krishnak/scratch/smudl/goldset_files/truth1_positions.txt', '/home/users/astar/gis/krishnak/scratch/SMURF_GOLDSET_PREDICTIONS/icgc_cll-T40', '/home/users/astar/gis/krishnak/scratch/icgc_cll-N.bam', '/home/users/astar/gis/krishnak/scratch/icgc_cll-T40.bam'),

('icgc_mbl-T40', '/home/users/astar/gis/krishnak/scratch/smudl/goldset_files/truth2_positions.txt', '/home/users/astar/gis/krishnak/scratch/SMURF_GOLDSET_PREDICTIONS/icgc_mbl-T40', '/home/users/astar/gis/krishnak/scratch/icgc_mbl-N.bam', '/home/users/astar/gis/krishnak/scratch/icgc_mbl-T40.bam'),

('icgc_cll-T20', '/home/users/astar/gis/krishnak/scratch/smudl/goldset_files/truth1_positions.txt', '/home/users/astar/gis/krishnak/scratch/SMURF_GOLDSET_PREDICTIONS/icgc_cll-T20', '/home/users/astar/gis/krishnak/scratch/icgc_cll-N.bam', '/home/users/astar/gis/krishnak/scratch/icgc_cll-T20.bam'),

('icgc_mbl-T20', '/home/users/astar/gis/krishnak/scratch/smudl/goldset_files/truth2_positions.txt', '/home/users/astar/gis/krishnak/scratch/SMURF_GOLDSET_PREDICTIONS/icgc_mbl-T20', '/home/users/astar/gis/krishnak/scratch/icgc_mbl-N.bam', '/home/users/astar/gis/krishnak/scratch/icgc_mbl-T20.bam'),

('icgc_cll-N30X', '/home/users/astar/gis/krishnak/scratch/smudl/goldset_files/truth1_positions.txt', '/home/users/astar/gis/krishnak/scratch/SMURF_GOLDSET_PREDICTIONS/icgc_cll-N30X', '/home/users/astar/gis/krishnak/scratch/icgc_cll-N30.bam', '/home/users/astar/gis/krishnak/scratch/icgc_cll-T.bam'),

('icgc_mbl-N30X', '/home/users/astar/gis/krishnak/scratch/smudl/goldset_files/truth2_positions.txt', '/home/users/astar/gis/krishnak/scratch/SMURF_GOLDSET_PREDICTIONS/icgc_mbl-N30X', '/home/users/astar/gis/krishnak/scratch/icgc_mbl-N30.bam', '/home/users/astar/gis/krishnak/scratch/icgc_mbl-T.bam')

]

#dream_challenge_files_on_nscc = [ ('synthetic_sample_%d' % x, '/seq/astar/gis/projects/skandera/DREAM_SYNTHETIC_TUMORS/synthetic.challenge.set%d.trues.txt' % x, '/seq/astar/gis/projects/skandera/DREAM_SYNTHETIC_TUMORS/synthetic.challenge.set%d.normal.bam' % x, '/seq/astar/gis/projects/skandera/DREAM_SYNTHETIC_TUMORS/synthetic.challenge.set%d.tumor.bam' % x) for x in range(1, 6) ]

dream_challenge_files_on_nscc = [

('dream%d' % x, '/seq/astar/gis/projects/skandera/training/wgs/dream/dream%d.truth.txt' % x, '/seq/astar/gis/projects/skandera/training/wgs/dream/dream%d-N-ready.bam' % x, '/seq/astar/gis/projects/skandera/training/wgs/dream/dream%d-T-ready.bam' % x) for x in range(1,6)

]

giab_files_on_nscc = [

('giab', '/seq/astar/gis/projects/skandera/training/wgs/giab/giab_truth_snvs.txt', '/seq/astar/gis/projects/skandera/training/wgs/giab/giab-N-ready.bam', '/seq/astar/gis/projects/skandera/training/wgs/giab/giab-T-ready.bam')

]

TGEN_FILES_ON_NSCC = [

('tgen', '/home/users/astar/gis/krishnak/scratch/TGEN/tgen_snv_trues.txt', '/home/users/astar/gis/krishnak/scratch/TGEN/tgen_colo829-N-ready.bam', '/home/users/astar/gis/krishnak/scratch/TGEN/tgen_colo829-T-ready.bam')

]

subsampled_data_folder_name = 'subsampled'
all_data_folder_name = 'all_data'

#### GERMLINE VARIANT FILTER IN POST-PROCESSING VCF ###
GERMLINE_FILTER = True

############# TRAINING PARAMS #########

EPOCHS_TO_TRAIN = 200
TRAINING_BATCH_SIZE = 32

TEST_TIME_TRAINING = False
ADVERSARIAL_TRAINING = False
SPECTRAL_DECOUPLING = False # default l2 reg with weight 2e-05
SD_COEFF = 2e-05 # SPECTRAL_DECOUPLING lambda
LR_SCHEDULE = True
LOSS = 'binary_crossentropy' # 'mean_squared_error'

SAMPLE_READS = True # sample reads and sort by CO. don't use this when generating training data if you don't want sampling, or set SAMPLE_READS_COUNT to 1. 
SAMPLE_READS_COUNT = 1 # default 1. for use with SAMPLE_READS

INPUT_WHITENING = False # decorrelate input pixels per channel
WHITENING_MATRIX = 'training_data_whitening_matrix.npy' # computed using 10k random training samples. see smudl/utils.py -> compute_whitening_matrix_torch

VARIABLE_INPUT_SIZE = False # if True, all reads at candidate sites are used
INITIALIZE_EXISTING_WEIGHTS = False

STOCHASTIC_WEIGHT_AVERAGING = False # keep an swa copy of the model
EXPONENTIAL_MOVING_AVERAGE = False # keep an ema copy of the model

SPIKE_IN_GOLDSET = False # during training
DOMAIN_ADVERSARIAL_TRAINING = False # DANN
best_combined_dann_model = 'best.combined_model.dann.hdf5'
dann_target_dist_data_folder = 'target_distribution_data'

# EXPERIMENTS FOLDER
architecture = 'convnet2_retrain_subset' # 'convnet2_input_whitening' # 'inceptionv3_train_on_val.ood_val_set' # 'convnet2_train_on_val' # 'inceptionv3_train_on_val' # 'inceptionv3_variable_input_height' # 'snv.inceptionv3.new_architecture.old_weights' # 'convnet2_global_average_pooling' # 'inception' # 'convnet2_mean_squared_error' # 'EMA.convnet2_spectral_decoupling_sd_coeff_%s' % SD_COEFF # 'convnet2_fgsm_rand_l2_eps_0.25' # 'convnet2.lr_0.0001_dr_0.0' #  'convnet2_ttt' # 'convnet2_selu' # 'convnet2_group_normalization' # 'EfficientNetB0'
#print("architecture:", architecture)

#### TRAINING FOLDERS
training_data_folder_on_aquila = '/mnt/projects/krishnak/kiran/smudl_training_data/'
training_data_folder_on_nscc = '/home/users/astar/gis/krishnak/scratch/smudl_training_data'
training_data_folder_on_workstation = '/media/nvme/kiran/smudl_training_data/'
coverage_info_file = 'coverage_info.txt'
mutation_burden_info_file = os.path.join('mutation_burden_info.npy')
training_data_folder = training_data_folder_on_workstation

folder_to_save_trained_models = '/media/nvme/kiran/smudl_trained_models/'
normalized_training_data_folder_name = 'normalized_training_data'
save_models_to_folder = 'trained_models'
tensorboard_log_directory = 'Tensorboard_logs'
validation_f1_score_history_file = 'validation_f1_score_history.%s.npy' % architecture
validation_precision_history_file = 'validation_precision_history.%s.npy' % architecture
validation_recall_history_file = 'validation_recall_history.%s.npy' % architecture
validation_domain_classifier_accuracy_history_file = 'validation_domain_classifier_accuracy_history.npy'

config_file = 'config_%s.npy' % architecture # save constants file as dictionary

experiment_details_file = 'experiment_details.txt'
patient_names_file = 'list_of_patients_used.npy'
training_filenames = 'list_of_files_used_for_training.npy'
validation_filenames = 'list_files_used_for_validation.npy'
testing_filenames = 'list_of_files_used_for_testing.npy'
best_model_name = 'model.best.%s.hdf5' % architecture
initial_model_name = 'model.initial.%s.hdf5' % architecture # initial model to continue training
best_model_architecture = 'model.best.architecture.%s.json' % architecture # architecture only of the best model as a json file
best_model_weights = 'model.best.weights.%s.hdf5' % architecture # weights only
finetuned_models_dir = 'finetuned_models'
SWA_WEIGHTS_MODEL = 'model.SWA.%s.hdf5' % architecture
EMA_WEIGHTS_MODEL = 'model.EMA.%s.hdf5' % architecture
SAVE_EVERY_EPOCH = False # if True, save every epoch, not just the best val acc. model

channels_means_file = 'channel_means_for_normalization.npy'
channels_std_devs_file = 'channel_standard_deviations_for_normalization.npy'
shuffled_training_data_folder = 'shuffled_training_data'
shuffled_data_indices = 'shuffled_data_indices.npy'
shuffled_batch_size = 10000

experiments_folder_on_aquila = '/mnt/projects/krishnak/kiran/smudl_trained_models/'
experiments_folder_in_nscc = '/home/users/astar/gis/krishnak/scratch/smudl_trained_models'
#experiments_folder_in_nscc = '/home/users/astar/gis/krishnak/scratch/workstation_backup/smudl_trained_models'
experiments_folder_in_workstation = '/media/nvme/kiran/smudl_trained_models/'
experiments_folder = experiments_folder_in_workstation

# SAMPLE FOLDER NAMES
sample_candidates_folder = 'candidates'
sample_predictions_folder = 'predictions'

snv_candidates_folder = 'snvs'
indel_candidates_folder = 'indels'

""" SNV MODEL """
snv_model_folder = 'snv_model'
BEST_MODEL_ARCHITECTURE_PATH = os.path.join(snv_model_folder, 'model.best.architecture.json')
BEST_MODEL_WEIGHTS_PATH = os.path.join(snv_model_folder, 'model.best.weights.hdf5')
BEST_MODEL_PATH = os.path.join(snv_model_folder, 'model.best.hdf5')
NORMALIZATION_MEANS_PATH = os.path.join(snv_model_folder, channels_means_file)
NORMALIZATION_STD_DEVS_PATH = os.path.join(snv_model_folder, channels_std_devs_file)

""" CURRENT EXPERIMENT """
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

    global CURRENT_FINETUNED_MODELS_DIR
    CURRENT_FINETUNED_MODELS_DIR = os.path.join(CURRENT_EXPERIMENT_FOLDER, save_models_to_folder, finetuned_models_dir)

    global CURRENT_NORMALIZATION_MEANS_PATH
    CURRENT_NORMALIZATION_MEANS_PATH = os.path.join(CURRENT_EXPERIMENT_FOLDER, channels_means_file)

    global CURRENT_NORMALIZATION_STD_DEVS_PATH
    CURRENT_NORMALIZATION_STD_DEVS_PATH = os.path.join(CURRENT_EXPERIMENT_FOLDER, channels_std_devs_file)


DEFAULT_EXPERIMENT_ID = 144 # 9 # 85 # 43

### PRE-FILTER POSITIONS IN WHOLE GENOME
filtering_root_folder_on_nscc = '/home/users/astar/gis/krishnak/scratch/filtered_positions/'
filtering_root_folder_on_aquila = '/home/krishnak/filtered_positions/'
filtering_root_folder = filtering_root_folder_on_aquila
CURRENT_FILTERING_FOLDER = 'filtering_3'

filtering_folder = os.path.join(filtering_root_folder, CURRENT_FILTERING_FOLDER)
filtered_positions_file = 'Positions.csv'
filtering_details_file = 'filter_details.txt'
filtering_batches_folder = os.path.join(filtering_folder, 'output')

# CUSTOM PRE_FILTERS
MIN_BASE_QUALITY = 22
MIN_COVERAGE = 7
MIN_MUTANT_ALLELE_READS_IN_TUMOR = 2
MIN_READ_MAPPING_QUALITY = 10
MAX_ALTERNATIVE_ALLELE_FREQUENCY_IN_NORMAL = 0.05
MIN_MUTANT_ALLELE_FREQUENCY_IN_TUMOR = 0.035
MIN_MAPPING_QUALITY_FOR_MUTANT_ALLELE_READS = 30

# Reference: https://media.nature.com/original/nature-assets/ncomms/2015/151209/ncomms10001/extref/ncomms10001-s1.pdf
# Supplementary table 7
MIN_DISTANCE_FROM_VARIANT_TO_ALIGNMENT_END_MEDIAN = 10
MIN_DISTANCE_FROM_VARIANT_TO_ALIGNMENT_END_MAD = 3 # MAD - Median Absolute Deviation
MAX_PROPORTION_OF_LOW_MAP_QUAL_READS_AT_VARIANT = .10 # low map qual is if MAPQ < 1
MAX_MAP_QUAL_DIFF_MEDIAN = 5 # The difference in the median mapping quality of variant reads (in the tumor) and reference reads (in the normal) is greater than 5
MIN_VARIANT_MAP_QUAL_MEDIAN = 40 # The median mapping quality of variant reads is less than 40
MIN_VARIANT_BASE_QUAL_MEDIAN = 30 # The median base quality at the variant position of variant reads is less than 30
MIN_VARIANT_ALLELE_COUNT = 4 # The number of variant-supporting reads in the tumor is less than 4
MAX_VARIANT_ALLELE_COUNT_IN_CONTROL = 1 # The number of variant-supporting reads in the normal is greater than 1
MIN_STRAND_BIAS = 0.02 # The strand bias for variant reads covering the variant position, i.e. the fraction of reads in either direction, is less than 0.02, unless the strand bias for all reads is also less than 0.02
"""
The largest number of variant positions within any 50 base pair
window surrounding, but excluding, the variant position is greater
than 2; variant positions are those in which the number of
alternate allele is supported by at least 2 reads and at least 5% of
all reads covering that position.
"""
SNVCluster50 = 2
"""
The largest number of variant positions within any 100 base pair
window surrounding, but excluding, the variant position is greater
than 4; variant positions are those in which the number of
alternate allele is supported by at least 2 reads and at least 5% of
all reads covering that position
"""
SNVCluster100 = 2


ref_path_on_workstation = "/media/nvme/kiran/GRCh37/GRCh37.fa"
predictions_folder_on_workstation = '/home/kiran/smudl_predictions/'

ref_path_on_aquila = "/mnt/projects/huangwt/wgs/genomes/seq/GRCh37.fa"
predictions_folder_on_aquila = '/home/krishnak/smudl_predictions/'

ref_path_on_nscc = "/home/users/astar/gis/krishnak/scratch/GRCh37.fa"
predictions_folder_on_nscc = '/home/users/astar/gis/krishnak/scratch/smudl_predictions/'

from os.path import expanduser

HOME_DIRECTORY = expanduser('~')

if HOME_DIRECTORY == '/home/kiran':
    ref_path = ref_path_on_workstation

elif HOME_DIRECTORY == '/home/users/astar/gis/krishnak':
    ref_path = ref_path_on_nscc

elif HOME_DIRECTORY == '/home/krishnak':
    ref_path = ref_path_on_aquila

combined_predictions_file = 'Predictions.csv'

############## INPUT ENCODING SETTINGS #####################

REMOVE_REFERENCE_CHANNEL = False
ADD_NORMAL_TUMOR_FLAG_CHANNEL = False # channel to indicate if position is in normal or tumor

CTR_DUP = 5 # Duplicate the center column, which is to be predicted
SEQ_LENGTH = 31 # must be odd, length of sequence

# SEQ_LENGTH needs to be ODD
assert SEQ_LENGTH % 2

PER_IMAGE_WIDTH = SEQ_LENGTH + CTR_DUP - 1

FLANK = int((SEQ_LENGTH-1)/2)
NUM_READS = 100 # max number of reads to include / array height

MAX_READS = 500 # max reads to use if all reads are used during prediction (when VARIABLE_INPUT_SIZE=True)

# tumor and normal images adjacent in the encoding. If False, tumor and normal stacked one behind another
TUMOR_NORMAL_ADJACENT = True

TETRIS_MODE = False # if False, only one read is encoded in each row of the image. If true, reads fill up the image from the top down per position, even if reads have to be broken
ENCODE_INSERTIONS = False
SORT_BASES = False # sort each column by base A, T, G, C

SMALLER_INPUT = True

def set_input_encoding(af):
    global NUM_CHANNELS_PER_IMAGE
    NUM_CHANNELS_PER_IMAGE = 4

    if af:
        NUM_CHANNELS_PER_IMAGE += 4 # one channel for each base

    global INCLUDE_ALLELE_FREQUENCY
    INCLUDE_ALLELE_FREQUENCY = af

    global NUM_CHANNELS
    global INPUT_SHAPE

    if TUMOR_NORMAL_ADJACENT:
        if REMOVE_REFERENCE_CHANNEL:
            NUM_CHANNELS = NUM_CHANNELS_PER_IMAGE
        else:
            NUM_CHANNELS = NUM_CHANNELS_PER_IMAGE + 1 # add one for reference channel

        if ADD_NORMAL_TUMOR_FLAG_CHANNEL:
            NUM_CHANNELS += 1

        INPUT_SHAPE = [ NUM_READS, 2*(SEQ_LENGTH + CTR_DUP - 1), NUM_CHANNELS]
    else:
        NUM_CHANNELS = NUM_CHANNELS_PER_IMAGE + NUM_CHANNELS_PER_IMAGE + 1 # tumor + normal + 1 for ref channel
        INPUT_SHAPE = [ NUM_READS, SEQ_LENGTH + CTR_DUP - 1, NUM_CHANNELS]

DEFAULT_INCLUDE_ALLELE_FREQUENCY = False
set_input_encoding(DEFAULT_INCLUDE_ALLELE_FREQUENCY)

# if True, compares bases to the reference base. Encodes 0 if they are the same
COMPARE_REF_BASE = False

def set_encoding(tetris_mode, encode_insertions, sort_bases):
    global encoding_name
    encoding_name = 'NUM_READS_%s_SEQ_LENGTH_%s_CTR_DUP_%s_NUM_CHANNELS_%s_NUM_CHANNELS_PER_IMG_%s' % (str(NUM_READS), str(SEQ_LENGTH), str(CTR_DUP), str(NUM_CHANNELS), str(NUM_CHANNELS_PER_IMAGE) )

    if tetris_mode:
        encoding_name += '_TETRIS_MODE'

    if encode_insertions:
        encoding_name += '_ENCODE_INSERTIONS'

    if sort_bases:
        encoding_name += '_SORT_BASES'

set_encoding(TETRIS_MODE, ENCODE_INSERTIONS, SORT_BASES)


# NOTE: We set up an array with 7 channels where:
    #   0 => sequence comparision of reference to normal
    #   1 => base quality of normal
    #   2 => strand direction of normal
    #   3 => reference sequence
    #   4 => sequence comparision of reference to tumor
    #   5 => base quality of tumor
    #   6 => strand direction of tumor

__VERSION__ = '1.1.0'

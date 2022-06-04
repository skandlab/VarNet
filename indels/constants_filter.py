"""
Name:    constants.py

Description:
        Contains all filepaths, hyperparameters, and input encoding
        settings needed by other scripts in main_indels.
        (note that filepaths and functions for the patient BAM files themselves are
        stored in patient_data_locations.py, not constants.py)

        When constants.py is imported, all of the JSON serializable variables it contains are saved
        in a dictionary, called data.

        WARNING: do not put functions in constants.py, because they are not JSON serializable and will not
        be saved in the JSON string

How to Use:
        There are two ways to use values from constants.py:

        1) importing: at top of script, use
            >>> import constants as c

           This has the disadvantage that whenever a job is submitted and queued, if constants.py
           is changed before the job actually runs, it will then import the CHANGED version of constants,
           instead of the original one from when it was submitted.

           This can lead to a lot of errors when a constant is expected to have one value, but
           is run with another. It is also inconvenient when submitting jobs with different hyperparameters

           As a result, DO NOT import constants like this for scripts in the main workflow
           (importing is only acceptable for temporary scripts that will not be used again)

         2) loading from JSON string.

             This method allows you to avoid the problems with method 1), by doing the following:

             Save constants.data with a job-specific name (called json_name) as soon as a job
             is submitted. Then, when the submitted job runs, the JSON string containing constants.data
             is opened and converted into a stuct, so that calling:
               var = c.user
             gives exactly the same result as if you had imported constants.

             submit_job automatically saves the JSON string of constants.data when you submit a job, with:

            >>> with open(json_name, 'w') as jsonfile:
            >>>     json.dump(c.data, jsonfile)

            (submit_job also submits a job with concatenate_logfiles.py, which automatically deletes the JSON string)

            Then, add the following near the top of any script you want to use constants in:

            >>>    class Struct:
            >>>     def __init__(self, **entries):
            >>>         self.__dict__.update(entries)
            >>>
            >>> def load_constants(json_name):
            >>>     global c
            >>>     if json_name:
            >>>            with open(json_name, 'r') as f:
            >>>             c = Struct(**json.load(f))
            >>>     else:
            >>>         c = __import__("constants")

            When you call "load_constants(json_name)" anywhere in that script, it is the equivalent of "import constants as c"

"""
import json
from os.path import expanduser, join
from enum import Enum, unique

@unique
class User(Enum):
    DYLAN_AQUILA = 0
    DYLAN_NSCC = 1
    DYLAN_WORKSTATION = 2
    ALEX_AQUILA = 3
    ALEX_NSCC = 4
    ALEX_WORKSTATION = 5
    KIRAN_WORKSTATION = 6
    KIRAN_NSCC = 7

@unique
class Cluster(Enum):
    AQUILA = 0
    NSCC = 1
    WORKSTATION = 2

HOME_DIRECTORY = expanduser('~')

if HOME_DIRECTORY == '/home/delu':
    path_to_smudl = '/home/delu/smudl/'
    user = User.DYLAN_AQUILA
    cluster = Cluster.AQUILA
    
elif HOME_DIRECTORY == '/home/users/astar/gis/delu':
    path_to_smudl = '/home/users/astar/gis/delu/scratch/smudl/'
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
    path_to_smudl = '/home/kiran/smudl/'
    user = User.KIRAN_WORKSTATION
    cluster = Cluster.WORKSTATION

elif HOME_DIRECTORY == '/home/users/astar/gis/krishnak':
    path_to_smudl = '/home/users/astar/gis/krishnak/scratch/smudl/'
    user = User.KIRAN_NSCC
    cluster = Cluster.NSCC

else:
    # raise Exception("home directory {} does not contain any recognized user names".format(HOME_DIRECTORY))
    user = None
    cluster = None 

# === Job Submission Constants === #

if user == User.ALEX_AQUILA:
    path_to_smudl = "/home/aswift-scott/repos/smudl/"
    logfile_path = "/home/aswift-scott/logfiles/indels"
    notifications_address = "Swift-Scott_Alexander_from.tp@gis.a-star.edu.sg"

elif user == User.ALEX_NSCC:
    path_to_smudl = "/home/users/astar/gis/aswiftsc/repos/smudl/"    
    logfile_path = "/home/users/astar/gis/aswiftsc/logfiles/indels"    
    notifications_address = "Swift-Scott_Alexander_from.tp@gis.a-star.edu.sg"

elif user == User.ALEX_WORKSTATION:
    path_to_smudl = "TODO: PUT path_to_smudl INTO CONSTANTS"    
    logfile_path = "TODO: PUT logfile_path INTO CONSTANTS"  
    notifications_address = "Swift-Scott_Alexander_from.tp@gis.a-star.edu.sg"

elif user == User.DYLAN_AQUILA:
    path_to_smudl = "/home/delu/smudl/"
    logfile_path = "/home/delu/logfiles/indels"
    notifications_address = "TODO: PUT notifications_address INTO CONSTANTS"

elif user == User.DYLAN_NSCC:
    path_to_smudl = "/home/users/astar/gis/delu/smudl/"    
    logfile_path = "/scratch/users/astar/gis/delu/logfiles"    
    notifications_address = "TODO: PUT notifications_address INTO CONSTANTS"

elif user == User.DYLAN_WORKSTATION:
    path_to_smudl = "TODO: PUT path_to_smudl INTO CONSTANTS"    
    logfile_path = "TODO: PUT logfile_path INTO CONSTANTS"
    notifications_address = "TODO: PUT notifications_address INTO CONSTANTS"

if cluster == Cluster.AQUILA:
    python_path = "/mnt/software/unstowable/anaconda/envs/tensorflow/bin/python"


# === Generating Training Data === #

# reference FASTA locations and lists of allowed cancer datasets for generate_training_data
if cluster == Cluster.AQUILA:
    ref_path = "/mnt/projects/huangwt/wgs/genomes/seq/GRCh37.fa"
    valid_cancer_datasets = ['lung', 'liver']
elif cluster == Cluster.NSCC:    
    ref_path = "/seq/astar/gis/projects/skandera/SMUDL/GRCh37.fa"
    valid_cancer_datasets = ['gastric', 'goldset', 'goldset_smurf']
elif cluster == Cluster.WORKSTATION:
    ref_path = "TODO: PUT ref_path INTO CONSTANTS"
    valid_cancer_datasets = "TODO: PUT valid_cancer_datasets INTO CONSTANTS"

# Parallization Parameters:
# allowable ratio between largest amount of work and smallest amount of work when dividing work between nodes
max_node_work_ratio = 1.1 # don't go lower than 1.1

# when searching for optimal way to divide patients, timeout after this many seconds
divide_patients_timeout = 1.0 # in practice, never exceeds 1.0
allowable_pos_per_batch = 5000
allowable_pos_per_shuffled_batch = 100000 # should be at least 100000


# === Pre-filtering Genome Settings === #

# chromosomes_list is ordered by chromosome size!
chromosomes_list = ['1','2','3','4','5','6','7','X','8','9','10','11','12','13','14','15','16','17','18','19','20','Y','22','21','MT']
num_chromosomes = len(chromosomes_list)

# Filtering options:
DETAILED_ANALYSIS = False
ONLY_FILTER_TUMOR = False
#MIN_FREQ = (2, 2) # (min freq for insertions, min freq for deletions)
#MIN_MQ = (30, 20) # (min mapping quality for insertions, min mapping quality for deletions)
#MIN_AVG_BQ = 22 # min avg base quality for insertions
#MIN_TYPE_DIS = 0
MIN_FREQ = (2,2)
MIN_MQ = (35,35)
MIN_AVG_BQ = 0
MIN_TYPE_DIS = 0

PREFILTER_SETTINGS = [('DETAILED_ANALYSIS', DETAILED_ANALYSIS),
                      ('ONLY_FILTER_TUMOR', ONLY_FILTER_TUMOR),
                      ('MIN_FREQ', MIN_FREQ),
                      ('MIN_MQ', MIN_MQ),
                      ('MIN_AVG_BQ', MIN_AVG_BQ)]

candidates_encoding_name = ""

for pair in PREFILTER_SETTINGS:
    candidates_encoding_name += "_%s=%s" % ( pair[0], str(pair[1]) )

# Check for illegal filtering options:
assert MIN_FREQ[0] > 0 and MIN_FREQ[1] > 0 # minimum frequencies must be positive

# if DETAILED_ANALYSIS is true, the following parameters will be used to perform candidate position analysis
candidates_size_cutoffs = [1,2,4,6,10,15,20,30]
candidates_freq_cutoffs = [1,2,3,5,8,12,16,20]
candidates_bq_cutoffs = [5,10,15,20,25,30,35]
candidates_mq_cutoffs = [5,10,15,20,25,30,35]
candidates_disagreement_cutoffs = [10,20,30,50] # percents
candidates_depth_cutoffs = [20,40,60,100,150,300]
candidates_percent_indel_cutoffs = [3,5,8,11,15,19,25,31] # percents
candidates_proximity_cutoffs = [2,5,10,20,50,100,500]
candidates_STR_size_cutoffs = [1,2,3,4,5,6,7]
candidates_STR_reps_cutoffs = [1,2,3,4,5,7,10,15,25]

if DETAILED_ANALYSIS:
    prefilter_column_names = ['chrX', 'position', 'type', 'length_bp', 'normal_freq', 'tumor_freq', 'min_bq', 'avg_bq', 'read_mq', 'indel_type_disagreement', 'seq_disagreement', 'normal_coverage', 'tumor_coverage', 'soft_clip_proximity', 'near_indel_proximity', 'STR_length_bp', 'STR_repeats', 'is_real']
else:
    prefilter_column_names = ['chrX', 'position', 'is_real']

# File and folder names
if user == User.ALEX_AQUILA:
    candidates_path = "/mnt/projects/aswift-scott/indels/candidate_positions"
elif user == User.ALEX_NSCC:
    candidates_path = "/home/users/astar/gis/aswiftsc/scratch/candidate_positions"
elif user == User.ALEX_WORKSTATION:
    candidates_path = "TODO: PUT candidates_path INTO CONSTANTS"
elif user == User.DYLAN_AQUILA:
    candidates_path = "/mnt/projects/delu/DeepLearning/candidate_positions"
elif user == User.DYLAN_NSCC:
    candidates_path = "/scratch/users/astar/gis/delu/candidate_positions"
elif user == User.DYLAN_WORKSTATION:
    candidates_path = "TODO: PUT candidates_path INTO CONSTANTS"


# === Training Parameters === #

# training data paths
if user == User.ALEX_AQUILA:
    training_data_folder = "/mnt/projects/aswift-scott/indels/training_data/"
elif user == User.ALEX_NSCC:
    training_data_folder = "/home/users/astar/gis/aswiftsc/scratch/training_data"
elif user == User.ALEX_WORKSTATION:
    training_data_folder = "TODO: PUT training_data_folder INTO CONSTANTS"
elif user == User.DYLAN_AQUILA:
    training_data_folder = "/mnt/projects/delu/DeepLearning/new_generate_training_data_test/"
elif user == User.DYLAN_NSCC:
    training_data_folder = "TODO: PUT training_data_folder INTO CONSTANTS"
elif user == User.DYLAN_WORKSTATION:
    training_data_folder = "TODO: PUT training_data_folder INTO CONSTANTS"


# === Input Encoding Settings === #

# Input Window:

SEQ_LENGTH = 75 # length of sequence. must be odd
NUM_READS = 140 # max number of reads per image. must be even
FOCUS = 5 # number of bases in focus, inclusive. must be odd

# how to postition indels in the image window. allowed options:
#  - FIXEDFOCUS  : indel starts in beginning of FOCUS region, which is itself centered in window
#  - CENTERED    : indels are centered (if length varies, use longest). for even numbers, "center" index is rounded UP
#  - LEFT        : leftmost pixel in image is indel's first position
#  - RIGHT       : rightmost pixel in image is indel's last position (will not find last position if indel is longer than SEQ_LENGTH)
# all indels will start on the right extreme of the Focus. If False, all indels will be centered in the window
OFFSET = "FIXEDFOCUS"
OFFSET_OPTIONS = ["FIXEDFOCUS", "CENTERED", "LEFT", "RIGHT"]

# how to encode bases and possibly also information about indels. allowed options:
#  - BQIa : 2 channels - base (mapping A, T, C, G, D to values from 0 to 125, and the inserted versions of those in 125 to 255) and base quality
#  - BQIb : 2 channels - base (mapping A, AI, T, TI, C, CI, G, GI, D to values from 0 to 255) and base quality
#  - BQ   : 3 channels - base (mapping A, T, C, G, to values from 0 to 255), base quality, and I/D
#  - 2B   : 3 channels - AT (A negative, T positive, magnitude = quality), CG (C negative, G positive, magnitude = quality) and I/D
#  - 4BI  : 4 channels - A, C, T, G (magnitude = quality, negative = I)
#  - 4B   : 5 channels - A, C, T, G (magnitude = quality), and I/D
BASE_ENCODING = "BQIa"
BASE_ENCODING_OPTIONS = ["BQIa", "BQIb", "BQ", "2B", "4BI", "4B"]

# how to encode insertions, and aligning other reads around them. allowed options:
#  - PLACEHOLDER  :
#  - NEWLAYER     :
#  - COLLAPSE     :
#  - UNALIGNED    :
HANDLE_INSERTIONS = "UNALIGNED"
HANDLE_INSERTIONS_OPTIONS = ["COLLAPSE", "PLACEHOLDER", "NEWLAYER", "UNALIGNED"]

# how to encode insertions that are not the target but still occur within the window. allowed options:
#  - IGNORE       : simply do not include nontarget insertions
#  - FLAG         : have flag at insertion position (i.e. I/D channel) but do not record any info about insertion itself
#  - COLLAPSE     : same as for HANDLE_INSERTIONS
#  - COPYTARGET   : use the same method as HANDLE_INSERTIONS (exlcusively for PLACEHOLDER, NEWLAYER and
#                    UNALIGNED, which only make sense to use for nontarget insertions if you are also
#                    using them for target insertions)
HANDLE_NONTARGET_INSERTIONS = "FLAG"
HANDLE_NONTARGET_INSERTIONS_OPTIONS = ["IGNORE", "FLAG", "COLLAPSE", "COPYTARGET"]

# how to handle positions where read coverage is greater than NUM_READS. allowed options:
#  - FIRST     : chose first NUM_READS reads to occur when fetched
#  - MIDDLE    : chose the middle NUM_READS reads from the semi-ordering from fetch (assuming this maximizes reads containing section of interest)
#  - SMART     : chose reads to keep the proportion of indel-containing and non-indel-containing reads fixed. prioritize reads containing more of middle section given this constraint
HANDLE_OVERCOVERAGE = "MIDDLE"
HANDLE_OVERCOVERAGE_OPTIONS = ["FIRST", "MIDDLE", "SMART"]

# how to order reads in the window. allowed options:
#  - NONE        : use ordering from fetch as-is
#  - RANDOM      : random ordering
#  - CLUMP       : clump all indel-containing reads at top, and all non-indel-containing reads at bottom, use ordering from fetch as-is
#  - RANDOMCLUMP : clump all indel-containing reads at top, and all non-indel-containing reads at bottom, with random ordering in each clump
READ_ORDER = "NONE"
READ_ORDER_OPTIONS = ["NONE", "RANDOM", "CLUMP", "RANDOMCLUMP"]

# tumor and normal images adjacent in the encoding. If False, tumor and normal stacked one behind another
# *note: TUMOR_NORMAL_ADJACENT must be True if HANDLE_INSERTIONS is "UNALIGNED", because unaligned reads can't be stacked on top of the same reference
TUMOR_NORMAL_ADJACENT = True

# combine the mapping quality and strand direction channels into a single channel (+/- and magnitude)
COMBINE_SD_MQ = False

# when encoding ref bases for an insertion (which has no ref), extend the ref at the position of the insertion to fill the whole thing, as oppsed to leaving it all zeros
EXTEND_REF_INS = True

# check for illegal encoding parameters:
assert SEQ_LENGTH % 2
assert FOCUS % 2
assert not NUM_READS % 2
assert OFFSET in OFFSET_OPTIONS
assert BASE_ENCODING in BASE_ENCODING_OPTIONS
assert HANDLE_INSERTIONS in HANDLE_INSERTIONS_OPTIONS
assert HANDLE_NONTARGET_INSERTIONS in HANDLE_NONTARGET_INSERTIONS_OPTIONS
assert HANDLE_OVERCOVERAGE in HANDLE_OVERCOVERAGE_OPTIONS
assert READ_ORDER in READ_ORDER_OPTIONS

if HANDLE_NONTARGET_INSERTIONS == "COPYTARGET":
    assert HANDLE_INSERTIONS == "NEWLAYER" or HANDLE_INSERTIONS == "PLACEHOLDER" or HANDLE_INSERTIONS == "UNALIGNED"

if HANDLE_INSERTIONS == "UNALIGNED" or (OFFSET == "CENTERED" and HANDLE_INSERTIONS != "PLACEHOLDER"):
    assert TUMOR_NORMAL_ADJACENT

if READ_ORDER == "CLUMP" or READ_ORDER == "RANDOMCLUMP":
    assert TUMOR_NORMAL_ADJACENT

# determine FETCH_START and FETCH_END (how many bases to the left and right of point of interest do we fetch)
if OFFSET == "FIXEDFOCUS":
    FETCH_START = (SEQ_LENGTH - FOCUS) / 2
    FETCH_END = FETCH_START + FOCUS - 1

elif OFFSET == "LEFT":
    FETCH_START = 0
    FETCH_END = SEQ_LENGTH - 1

elif OFFSET == "RIGHT":
    FETCH_START = SEQ_LENGTH # we don't know how long the indel is, so need to fetch entire SEQ_LENGTH on either side
    FETCH_END = SEQ_LENGTH - 1

elif OFFSET == "CENTERED":
    if HANDLE_INSERTIONS == "PLACEHOLDER":
        FETCH_START = (SEQ_LENGTH - 1) / 2
        FETCH_END = FETCH_START
    else:
        FETCH_START = (SEQ_LENGTH - 1) / 2
        FETCH_END = SEQ_LENGTH - 1 # we don't know how long the indel is, so need to fetch extra half SEQ_LENGTH on right side

ROW_LENGTH = FETCH_END + FETCH_START + 1 # the initial length of the rows to be fetched by

# determine NUM_CHANNELS_PER_IMAGE based on BASE_ENCODING and COMBINE_SD_MQ
if BASE_ENCODING == "BQIa" or BASE_ENCODING == "BQIb":
    BASE_CHANNELS = 2
    REF_CHANNELS = 1
    COMBINE_BQ = False
    NEWLAYER_CHANNELS = 2
    
    if BASE_ENCODING == "BQIa":
        BASE_TO_COLOUR = {'A': 25, 'T': 50, 'C': 75, 'G': 100, 'D': 125, 'N': 0, 'W': 0, 'S': 5, 'M': 0, 'K': 0, 'R': 0, 'Y': 0, 'B': 0, 'H': 0, 'V': 0}
        INSERTED_BASE_TO_COLOUR = {'A': 150, 'T': 175, 'C': 200, 'G': 225, 'D': 125, 'N': 0, 'W': 0, 'S': 5, 'M': 0, 'K': 0, 'R': 0, 'Y': 0, 'B': 0, 'H': 0, 'V': 0}
    else:
        BASE_TO_COLOUR = {'A': 25, 'T': 75, 'C': 125, 'G': 175, 'D': 225, 'N': 0, 'W': 0, 'S': 5, 'M': 0, 'K': 0, 'R': 0, 'Y': 0, 'B': 0, 'H': 0, 'V': 0}
        INSERTED_BASE_TO_COLOUR = {'A': 50, 'T': 100, 'C': 150, 'G': 200, 'D': 225, 'N': 0, 'W': 0, 'S': 5, 'M': 0, 'K': 0, 'R': 0, 'Y': 0, 'B': 0, 'H': 0, 'V': 0}
    
    PLACEHOLDER_COLOUR = 255

elif BASE_ENCODING == "BQ":
    BASE_CHANNELS = 3
    REF_CHANNELS = 1
    COMBINE_BQ = False
    NEWLAYER_CHANNELS = 2
    BASE_TO_COLOUR = {'A': 55, 'T': 105, 'C': 155, 'G': 205, 'D': 0, 'N': 0, 'W': 0, 'S': 10, 'M': 0, 'K': 0, 'R': 0, 'Y': 0, 'B': 0, 'H': 0, 'V': 0}
    INSERTED_BASE_TO_COLOUR = BASE_TO_COLOUR
    PLACEHOLDER_COLOUR = 255

elif BASE_ENCODING == "2B":
    BASE_CHANNELS = 3
    REF_CHANNELS = 2
    COMBINE_BQ = True
    NEWLAYER_CHANNELS = 2
    BASE_CHANNEL_COEFS = {'A':[1,0], 'T':[-1,0], 'C':[0,1], 'G':[0,-1], 'D':[0,0], 'N':[0,0], 'W':[0,0], 'S':[0,0], 'M':[0,0], 'K':[0,0], 'R':[0,0], 'Y':[0,0], 'B':[0,0], 'H':[0,0], 'V':[0,0]}
    INSERTED_BASE_CHANNEL_COEFS = BASE_CHANNEL_COEFS
    ZERO_QUALITY_POINT = 155
    PLACEHOLDER_COLOUR = 155

elif BASE_ENCODING == "4BI":
    BASE_CHANNELS = 4
    REF_CHANNELS = 4
    COMBINE_BQ = True
    NEWLAYER_CHANNELS = 4
    BASE_CHANNEL_COEFS = {'A':[1,0,0,0], 'T':[0,1,0,0], 'C':[0,0,1,0], 'G':[0,0,0,1], 'D':[1,1,1,1], 'N':[0,0,0,0], 'W':[0,0,0,0], 'S':[0,0,0,0], 'M':[0,0,0,0], 'K':[0,0,0,0], 'R':[0,0,0,0], 'Y':[0,0,0,0], 'B':[0,0,0,0], 'H':[0,0,0,0], 'V':[0,0,0,0]}
    
    if HANDLE_INSERTIONS == "NEWLAYER":
        INSERTED_BASE_CHANNEL_COEFS = BASE_CHANNEL_COEFS
        ZERO_QUALITY_POINT = 0
    else:
        INSERTED_BASE_CHANNEL_COEFS = {'A':[-1,0,0,0], 'T':[0,-1,0,0], 'C':[0,0,-1,0], 'G':[0,0,0,-1], 'D':[0,0,0,0], 'N':[0,0,0,0], 'W':[0,0,0,0], 'S':[0,0,0,0], 'M':[0,0,0,0], 'K':[0,0,0,0], 'R':[0,0,0,0], 'Y':[0,0,0,0], 'B':[0,0,0,0], 'H':[0,0,0,0], 'V':[0,0,0,0]}
        ZERO_QUALITY_POINT = 155

    PLACEHOLDER_COLOUR = 25

elif BASE_ENCODING == "4B":
    BASE_CHANNELS = 5
    REF_CHANNELS = 4
    COMBINE_BQ = True
    NEWLAYER_CHANNELS = 4
    BASE_CHANNEL_COEFS = {'A':[1,0,0,0], 'T':[0,1,0,0], 'C':[0,0,1,0], 'G':[0,0,0,1], 'D':[0,0,0,0], 'N':[0,0,0,0], 'W':[0,0,0,0], 'S':[0,0,0,0], 'M':[0,0,0,0], 'K':[0,0,0,0], 'R':[0,0,0,0], 'Y':[0,0,0,0], 'B':[0,0,0,0], 'H':[0,0,0,0], 'V':[0,0,0,0]}
    INSERTED_BASE_CHANNEL_COEFS = BASE_CHANNEL_COEFS
    ZERO_QUALITY_POINT = 0
    PLACEHOLDER_COLOUR = 0

if BASE_ENCODING == "2B" or BASE_ENCODING == "4BI" or BASE_ENCODING == "4B":
    BASE_QUALITY_SCALE = 255 - ZERO_QUALITY_POINT
else:
    BASE_QUALITY_SCALE = 255
if COMBINE_SD_MQ:
    NUM_CHANNELS_PER_IMAGE = BASE_CHANNELS + 1
else:
    NUM_CHANNELS_PER_IMAGE = BASE_CHANNELS + 2

# determine INPUT_SHAPE based on TUMOR_NORMAL_ADJACENT and HANDLE_INSERTIONS
if HANDLE_INSERTIONS == "NEWLAYER":
    CHANNEL_MIDPOINT = REF_CHANNELS + NUM_CHANNELS_PER_IMAGE + NEWLAYER_CHANNELS
    if TUMOR_NORMAL_ADJACENT:
        INPUT_SHAPE = [ NUM_READS, 2*SEQ_LENGTH, REF_CHANNELS + NUM_CHANNELS_PER_IMAGE + NEWLAYER_CHANNELS ]
    else:
        INPUT_SHAPE = [ NUM_READS, SEQ_LENGTH, REF_CHANNELS + 2*(NUM_CHANNELS_PER_IMAGE + NEWLAYER_CHANNELS) ]
else:
    CHANNEL_MIDPOINT = REF_CHANNELS + NUM_CHANNELS_PER_IMAGE
    if TUMOR_NORMAL_ADJACENT:
        INPUT_SHAPE = [ NUM_READS, 2*SEQ_LENGTH, REF_CHANNELS + NUM_CHANNELS_PER_IMAGE ]
    else:
        INPUT_SHAPE = [ NUM_READS, SEQ_LENGTH, REF_CHANNELS + 2*NUM_CHANNELS_PER_IMAGE ]

# Summarize settings and generate encoding name:
SETTINGS = {"OFFSET" : OFFSET,
            "BASE_ENCODING" : BASE_ENCODING,
            "HANDLE_INSERTIONS" : HANDLE_INSERTIONS,
            "HANDLE_NONTARGET_INSERTIONS" : HANDLE_NONTARGET_INSERTIONS,
            "HANDLE_OVERCOVERAGE" : HANDLE_OVERCOVERAGE,
            "READ_ORDER" : READ_ORDER,
            "TUMOR_NORMAL_ADJACENT" : TUMOR_NORMAL_ADJACENT,
            "COMBINE_SD_MQ" : COMBINE_SD_MQ,
            "EXTEND_REF_INS" : EXTEND_REF_INS,
            "SEQ_LENGTH" : SEQ_LENGTH,
            "NUM_READS" : NUM_READS}
            
if OFFSET == "FIXEDFOCUS":
    SETTINGS["FOCUS"] = FOCUS

"""
encoding_name = "{}={}".format(SETTINGS[0][0], SETTINGS[0][1])
for pair in SETTINGS[1:]:
    encoding_name += ",{}={}".format(pair[0], pair[1])
"""

##############################################################################################
#                NEXT, CONVERT ALL VARIABLES IN CONSTANTS TO A JSON STRING                   #
##############################################################################################

def is_json_serializable(key, val):
    """
    determine whether a key-value pair in the __dict__ attribute of constants.py
    can be converted into a JSON string
    """
    if key.startswith('_') or key.endswith('_'):
        return False
    else:
        try:
            json.dumps(val)
            return True
        except TypeError:
            return False

# produce a single dictionary of all variables in constants.py, to be converted to a JSON string
data = { key : val for key, val in list(vars().items()) if is_json_serializable(key, val) }

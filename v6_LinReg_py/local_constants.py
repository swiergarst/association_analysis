PGDATABASE = "PGDATABASE"

COUNT_FUNCTION = "count"
AVG_FUNCTION = "avg"
MAX_FUNCTION = "max"
MIN_FUNCTION = "min"
SUM_FUNCTION = "sum"
STD_SAMP_FUNCTION = "stddev_samp"
POOLED_STD_FUNCTION = "pooled_std"
COUNT_NULL = "count_null"
COUNT_DISCRETE = "count_discrete"

HISTOGRAM = "histogram"
BOXPLOT = "boxplot"
QUARTILES = "quartiles"

VARIABLE = "variable"
TABLE = "table"
FUNCTIONS = "functions"
REQUIRED_FUNCTIONS = "required_functions"
REQUIRED_METHODS = "required_methods"
METHOD = "METHOD"
CONCEPT_ID = "concept_id"

ERROR = "error"
WARNING = "warning"
MESSAGE = "MESSAGE"

NAME = "NAME"
CALL = "CALL"
FETCH = "FETCH"
FETCH_ONE = "FETCH_ONE"
FETCH_ALL = "FETCH_ALL"

TABLE_MINIMUM = "TABLE_MINIMUM"
TABLE_MINIMUM_DEFAULT = 10
COUNT_MINIMUM = "COUNT_MINIMUM"
COUNT_MINIMUM_DEFAULT = 5
BIN_WIDTH_MINIMUM = "BIN_WIDTH_MINIMUM"
BIN_WIDTH_MINIMUM_DEFAULT = 2
IQR_THRESHOLD_DEFAULT = 1.5

BIN_WIDTH = "BIN_WIDTH"
IQR_THRESHOLD = "IQR_THRESHOLD"
COHORT = "COHORT"
COHORT_DEFINITION = "definition"
OPERATOR = "operator"
VALUE = "value"

PERSON_ID = "person_id"
PERSON_TABLE = "person"
OBSERVATION_TABLE = "observation"
OBSERVATION_CONCEPT_ID = "observation_concept_id"
MEASUREMENT_TABLE = "measurement"
MEASUREMENT_CONCEPT_ID = "measurement_concept_id"
CONDITION_TABLE = "condition_occurrence"
CONDITION_CONCEPT_ID = "condition_concept_id"

VALUE_AS_STRING = "value_as_string"
VALUE_AS_CONCEPT_ID = "value_as_concept_id"

DEFAULT_FUNCTIONS = [
    MAX_FUNCTION, MIN_FUNCTION, AVG_FUNCTION, POOLED_STD_FUNCTION
]

## data settings key values
NORMALIZE = "normalize"
USE_AGE = "use_age"
USE_DM = "use_dm"
USE_DELTAS = "use_deltas"
NORM_CAT = "norm_cat"
TARGET = "target"
ALL_COLS = "all_cols"
DATA_COLS = "data_cols"
DIRECT_COLS = "direct_cols" # the columns that are directly available in the postgres DB
SYNTH_COLS = "synth_cols" # the columns that have to be synthesized (like lag-time)
MODEL_LEN = "model_len"
BIN_WIDTH_BOXPLOT = "bw_boxplot"
GLOBAL_MEAN = "global_mean"
GLOBAL_STD = "global_std"
MODEL_COLS = "model_cols"
SENS = "sensitivity" # whether to run sensitivity analysis on lag time
STRATIFY = "stratify"
OPTION_COLS = "option_cols"
CAT_COLS = "cat_cols"
DEFINES = "defines"
LAG_TIME_COL = "lag_time_col"
AGE_COL = "age_col"
DATE_METABOLOMICS_COL = "date_metabolomics_col"
DATE_MRI_COL = "date_mri_col"
BIRTH_YEAR_COL = "birth_year_col"
METABO_AGE_COL = "metabo_age_col"
BRAIN_AGE_COL = "brain_age_col"
EDUCATION_CATEGORY_COL = "education_category_col"
EDUCATION_CATEGORIES_LIST = "education_category_list"
ID_COL = "id_col"
BP_1 = "bp_1"
STRATIFY_GROUPS = "stratify_groups"
STRATIFY_VALUES = "stratify_values"
CLASSIF_TARGETS = "classif_targets"

## v6_info ##
CLIENT = "client"
IMAGE_NAME = "image_name"
ORG_IDS = "org_ids"
COLLAB_ID = "collab_id"

# classifier settings keys
LR = "lr"
SEED = "seed"
COEF = "coef"

# fit round return keys
LOCAL_COEF = "local_coef"
TRAIN_MAE = "train_mae"
TEST_MAE = "test_mae"
TEST_LOSS = "test_loss"
LOCAL_TRAIN_SIZE = "local_train_size"
BP = "boxplot"

# calc_se return keys
TOP = "top"
BOT = "bot"
SIZE = "size"
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

## data column values
ID = "id"
METABO_AGE = "metabo_age"
METABO_HEALTH = "metabo_health"
BRAIN_AGE = "brain_age"
DATE_METABOLOMICS = "date_metabolomics"
DATE_MRI = "date_mri"
BIRTH_YEAR = "birth_year"
SEX = "sex"
DM = "dm"
BMI = "bmi"
EDUCATION_CATEGORY = "education_category_3"
EC1 = "ec_1"
EC2 = "ec_2"
EC3 = "ec_3"
SENSITIVITY_1 = "Sens_1"
SENSITIVITY_2 = "Sens_2"
LAG_TIME = "Lag_time"
AGE = "Age"


# data column defines
ALL_EXISTING_COLS_VALUES = [ID, METABO_AGE, BRAIN_AGE, DATE_METABOLOMICS, DATE_MRI, BIRTH_YEAR, SEX, DM, BMI, EDUCATION_CATEGORY] # all columns in the postgres database
CAT_COLS_VALUES = [EDUCATION_CATEGORY, SEX, DM, EC1, EC2, EC3] # categorical columns
EXTRA_COLS_VALUES = [EDUCATION_CATEGORY, SENSITIVITY_1, SENSITIVITY_2, LAG_TIME, AGE] # columns we might want, but need to be synthesized




## data settings key values
NORMALIZE = "normalize"
USE_AGE = "use_age"
USE_DM = "use_dm"
USE_DELTAS = "use_deltas"
NORMALIZE_CAT = "normalize_cat"

NORM_CAT = "norm_cat"
TARGET = "target"
DATA_COLS = "data_cols"
DIRECT_COLS = "direct_cols" # the columns that are directly available in the postgres DB
SYNTH_COLS = "synth_cols" # the columns that have to be synthesized (like lag-time)



## v6_info ##
CLIENT = "client"
IMAGE_NAME = "image_name"
ORG_IDS = "org_ids"
COLLAB_ID = "collab_id"
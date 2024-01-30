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
# EC2 = "ec_2"
EC3 = "ec_3"
SENSITIVITY_1 = "Sens_1"
SENSITIVITY_2 = "Sens_2"
LAG_TIME = "Lag_time"
AGE = "Age"
DEMENTIA = "dementia"


# data column defines
ALL_EXISTING_COLS_VALUES = [ID, METABO_AGE, BRAIN_AGE, DATE_METABOLOMICS, DATE_MRI, BIRTH_YEAR, SEX, DM, BMI, EDUCATION_CATEGORY, DEMENTIA, METABO_HEALTH] # all columns in the postgres database
CAT_COLS_VALUES = [EDUCATION_CATEGORY, SEX, DM, EC1, EC3, DEMENTIA] # categorical columns
EXTRA_COLS_VALUES = [EDUCATION_CATEGORY, SENSITIVITY_1, SENSITIVITY_2, LAG_TIME, AGE] # columns we might want, but need to be synthesized
OPTION_COLS_VALUES = [EDUCATION_CATEGORY, SENSITIVITY_1, SENSITIVITY_2] # these will never be used 'as is'
# STRATIFY_GROUPS_VALUES = [DM, DEMENTIA, [DM, DEMENTIA]]
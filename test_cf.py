from vantage6.client import Client
import os
import numpy as np
import json
from io import BytesIO
import datetime
import psycopg2
from utils import init_global_params, average, define_model, get_results, normalize_workflow
import time
import matplotlib.pyplot as plt
dir_path = os.path.dirname(os.path.realpath(__file__))
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LinearRegression

def test_central(X_full, y_full):

    #full_X,full_Y = construct_XY(dset_paths)
    lm = LinearRegression(fit_intercept=False, copy_X=True)
    lm.fit(X_full, y_full)

    return lm.coef_

def test_central_hase(X_full, y_full):
    A = np.matmul(X_full.T, X_full)
    B = X_full.T * y_full

    A_inv = np.linalg.pinv(A)

    beta_hat = np.matmul(A_inv, B)

    beta_hat_total = np.sum(beta_hat, axis = 1)
    return(beta_hat_total)


def test_central_se(X_full, y_full, n_tot):
    lm = LinearRegression(fit_intercept=False, copy_X=True)
    lm.fit(X_full, y_full)
    pred = lm.predict(X_full)

    top = np.sum((y_full - pred)**2)
    #print(f'sum of squares, central: {top}')
    bot = np.sum((X_full - np.mean(X_full, axis = 0))**2, axis = 0)
    #print(X_full[:,0])
    #print(bot)
    pre = 1/(n_tot - X_full.shape[1])
    se = np.sqrt(pre * top * bot)
    return se

def test_central_hase_se(X_full, y_full, n_tot):
    lm = LinearRegression(fit_intercept=False, copy_X=True)
    lm.fit(X_full, y_full)
    pred = lm.predict(X_full)

    top = np.sum((y_full - pred)**2)
    print(f' sum of squares central: {top}')
    A = X.T @ X
    A_inv_diag = np.diag(np.linalg.pinv(A))
    bot = np.diag(np.linalg.pinv(X.T @ X))
    print(f' X_full shape: {X_full.shape}')
    df = (n_tot - X_full.shape[1])
    se = 1/np.sqrt( df / (bot * top))
    return se

client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "password")
client.setup_encryption(None)
ids = [org['id'] for org in client.collaboration.get(1)['organizations']]


model = "M3"
use_dm = True
use_age = True
use_deltas = True
normalize_cat = False
normalize = "global"

all_cols =  ["id", "metabo_age", "brain_age", "date_metabolomics", "date_mri","birth_year", "sex",  "dm", "education_category_3", "bmi"]
#all_cols = [None]
image_name = "sgarst/association-analysis:haseTest"
data_cols, extra_cols, to_norm_cols = define_model(model, use_dm = use_dm, use_age = use_age)


global_avg, global_std, global_size, avg_cols = normalize_workflow(client, image_name,  data_cols, extra_cols, use_deltas, normalize, normalize_cat=normalize_cat)

print(f'global_avg shape: {global_avg.shape}')
print(f'global avg cols: {avg_cols}')


task = client.post_task(
    input_= {
        "method" : "calc_ABC",
        "kwargs" : {
            #"data_cols" : data_cols,
            #"data_cols" : data_cols_norm,
            #"data_cols" : ['brain_age', "metabo_age"],#, "bmi"],
            "all_cols" : all_cols,
            "data_cols" : data_cols,
            "extra_cols" : extra_cols,
            "use_deltas" : use_deltas,
            "normalize" : normalize,
            "global_mean" : global_avg,
            "global_std" : global_std,
            "normalize_cat" : normalize_cat
            # "col_name" :  ['metabo_age', 'brain_age']
        }
    }, 
    name = "calculate A, B and C",
    image=image_name,
    organization_ids=ids,
    collaboration_id=1
)

ABC_results = get_results(client, task, print_log=False)

As = np.array([ABC_result['A'] for ABC_result in ABC_results])
Bs = np.array([ABC_result['B'] for ABC_result in ABC_results])
Cs = np.array([ABC_result['C'] for ABC_result in ABC_results])


#full_B = np.array(ABC_results[0]['B'])
#full_C = np.array(ABC_results[0]['C'])


#for i in range(1, len(ABC_results)):
    #full_B = np.concatenate((full_B, ABC_results[i]['B']), axis = 0)
    #full_C = np.append(full_C, ABC_results[i]['C'])


full_A = np.sum(As, axis = 0)
full_B = np.sum(Bs, axis = 0)
full_C = np.sum(Cs, axis = 0)
#full_B = np.concatenate((Bs), axis = 1)

A_inv = np.linalg.pinv(full_A)


print(full_A.shape, full_B.shape)
beta_hat = np.matmul(A_inv, full_B)
#beta_hat = np.sum(beta_hat_part, axis = 1)

A_inv_diag = np.diag(A_inv)
print(f"A_inv_diag shape: {A_inv_diag.shape}")
se_part = np.matmul(full_B.T,  A_inv)
se_part2 = np.matmul(se_part, full_B)
se_part3 = full_C - se_part2

bot = full_C - full_B.T @ A_inv @ full_B
print(f' sum of squares: {bot}')
print(f' full_B shape: {full_B.shape}')
df = global_size - full_B.shape[0]

print(f'C shape: {full_C.shape}')


se = 1/np.sqrt(df / (A_inv_diag * bot))
#se = np.sqrt(np.sum(se_part3) / (df * np.sum(np.diag(full_A))))

data_task = client.post_task(
    input_={
        "method" : "return_data",
        "kwargs" : {
            "all_cols" : all_cols,
            "data_cols" : data_cols,
            "extra_cols" : extra_cols,
            "use_deltas" : use_deltas,
            "normalize" : normalize,
            "global_mean" : global_avg,
            "global_std" : global_std,
            "normalize_cat" :normalize_cat
        }
    },
    name = "get data",
    image = image_name,
    organization_ids=ids,
    collaboration_id=1
)

data_results = get_results(client, data_task, print_log = False)

full_data = pd.concat([data_result['data'] for data_result in data_results])

central_data_cols = data_results[0]['data_cols']
#print(central_data_cols)
X = full_data[central_data_cols].values.astype(float)
y = full_data["metabo_age"].values.astype(float)
print(full_data.head())


beta_hat_central = test_central_hase(X, y)
beta_ground = test_central(X, y)

se_central = test_central_se(X, y, global_size)
se_hase_central = test_central_hase_se(X, y, global_size)
print(f'federated hase se: {se}')
print(f' central se: {se_central}')
print(f'central hase se: {se_hase_central}')


print(f' allclose A: {np.allclose(X.T@X, full_A)}')
print(f' allclose B: {np.allclose(X.T@y, full_B)}')
print(f' allclose C: {np.allclose(y.T@y, full_C)}')

# print(beta_ground.shape, beta_hat_central.shape, beta_hat.shape)

plt.plot(beta_ground, beta_hat, ".")
plt.plot(np.arange(-2, 10, 1), np.arange(-2, 10, 1))
plt.grid(True)
plt.xlabel("federated betas")
plt.ylabel("central betas")
plt.title("federated hase vs central lin reg")


plt.show()

plt.plot(beta_hat_central, beta_hat, ".")
plt.plot(np.arange(-2, 10, 1), np.arange(-2, 10, 1))
plt.grid(True)
plt.xlabel("federated betas")
plt.ylabel("central betas_hat")
plt.title("federated hase vs central hase")

plt.show()
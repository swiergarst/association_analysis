from vantage6.algorithm.tools.mock_client import MockAlgorithmClient
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import math

def test_central(X_full, y_full):

    #full_X,full_Y = construct_XY(dset_paths)
    lm = LinearRegression(fit_intercept=False)
    lm.fit(X_full, y_full)

    return lm.coef_

def test_central_hase(X_full, y_full):
    A = np.matmul(X_full.T, X_full)
    B = X_full.T * y_full

    A_inv = np.linalg.pinv(A)

    beta_hat = np.matmul(A_inv, B)
    print(beta_hat.shape)
    beta_hat_total = np.sum(beta_hat, axis = 1)
    return(beta_hat_total)


def make_federated(X_full, y_full, n_clients = 3):
    s_per_clients = math.floor(X_full.shape[0]/n_clients)

    paths = []
    cwd = os.path.dirname(os.path.realpath(__file__))
    for i in range(n_clients):

        X_fed = X_full[i * s_per_clients: (i+1) * s_per_clients, :]
        y_fed = y_full[i * s_per_clients: (i+1) * s_per_clients]
        y_fed = y_fed.reshape(-1, 1)
        data_client = np.concatenate((X_fed, y_fed), axis = 1)
        columns = [f"col_{j}" for j in range(X_fed.shape[1])]
        columns.append("target")
        df = pd.DataFrame(data=data_client, columns = columns)
        save_str = f"{cwd}/tmp_data/X_client{i}.csv"
        df.to_csv(save_str, index=False)
        paths.append(save_str)

    return paths, s_per_clients * n_clients



#dset_paths = [f"{cwd}/tmp_data/MNIST_2Class_IID_client{i}.csv" for i in range(3)]

X_full, y_full = load_diabetes(return_X_y = True)

dset_paths, tot_points  = make_federated(X_full, y_full)

X_full = X_full[:tot_points, :]
y_full = y_full[:tot_points]
print(tot_points)
#print(dset_paths)

client = MockAlgorithmClient(
    datasets= [[
        {
            "database" : dset_paths[0],
            "db_type" : "csv",
            "input_data" : {}
        }],[
        {
            "database" : dset_paths[1],
            "db_type" : "csv",
            "input_data" : {}            
        }
        ],[
        {
            "database" : dset_paths[2],
            "db_type" : "csv",
            "input_data" : {}            
        }
    ]],
    module="v6_LinReg_py"
### connect to server
)

organizations = client.organization.list()
org_ids = [organization["id"] for organization in organizations]

task = client.task.create(
    input_ = 
    {
        'method' : 'calc_ABC',
        'kwargs' : {}
    },
    organizations=org_ids
)
results = client.wait_for_results(task.get("id"))

full_A = np.sum([np.array(results[i]['A']) for i in range(len(org_ids))], axis = 0)
full_B = np.concatenate([np.array(results[i]['B']) for i in range(len(org_ids))], axis = 1)
#full_C = np.sum([np.array(results[i]['C']) for i in range(len(org_ids))], axis = 0)



A_inv = np.linalg.pinv(full_A)
#B_inv = np.invert(full_B)
#C_inv = np.invert(full_C)
beta_hat = np.matmul(A_inv, full_B)
full_beta_hat = np.sum(beta_hat, axis = 1) #unsure if this is correct

central_beta = test_central(X_full, y_full)

beta_hat_central = test_central_hase(X_full, y_full)

# print(central_beta.shape, full_beta_hat.shape)
# print(full_A.shape, A_inv.shape, beta_hat.shape)

plt.plot(central_beta, full_beta_hat, ".")
plt.plot(np.arange(-800, 800, 1), np.arange(-800, 800 ,1))
plt.grid(True)
plt.xlabel("federated betas")
plt.ylabel("central betas")
plt.title("federated hase vs central lin reg")


plt.show()

plt.plot(beta_hat_central, full_beta_hat, ".")
plt.plot(np.arange(-800, 800, 1), np.arange(-800, 800 ,1))
plt.grid(True)
plt.xlabel("federated betas")
plt.ylabel("central betas_hat")
plt.title("federated hase vs central hase")

plt.show()






def construct_XY(dset_paths):
    for i, path in enumerate(dset_paths):
        df = pd.read_csv(path)
        X = df.drop(columns=['label', 'test/train']).values
        Y = df['label'].values
        if i == 0:
            full_X = X
            full_Y = Y
        else:
            full_X = np.concatenate((full_X, X), axis = 0)
            full_Y = np.concatenate((full_Y, Y))

    print(full_X.shape, full_Y.shape)
    return full_X, full_Y

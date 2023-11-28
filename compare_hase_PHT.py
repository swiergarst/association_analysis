from vantage6.client import Client
from run import run
from run_hase import run_hase
from V6_implementation.utils2 import generate_data_settings, generate_v6_info
import matplotlib.pyplot as plt
import numpy as np

client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "password")
client.setup_encryption(None)
ids = [org['id'] for org in client.collaboration.get(1)['organizations']]
image_name = "sgarst/association-analysis:1.8.1"
ids = [2]
v6_info = generate_v6_info(client, image_name, ids, 1)

model = "M3" # options : M1, M2, M3, M4, M5, M6, M7
normalize = "global" # options: global, local, none
use_age = True # whether to use age as a covariate
use_dm = True # whether to use dm as a covariate
use_deltas = False # whether to look at delta metabo/brainage
normalize_cat = True # whether to normalize categorical variables
bin_width = 0.2
data_settings = generate_data_settings(model, normalize, use_age, use_dm, use_deltas, normalize_cat, bin_width)

n_runs = 1
n_rounds = 10

pht_results, _ = run(v6_info, data_settings, n_runs=n_runs, n_rounds= n_rounds)

hase_results = run_hase(v6_info, data_settings)

print(f'pht mae: {pht_results["test_mae"]}, hase mae: {hase_results["mae"]}')
print(f'pht columns : {pht_results["coef_names"]}, hase columns : {hase_results["coef_names"]}')
# print(f'local ec coefs: {np.array(pht_results["local_betas"]).shape}')
# print(f'local ec coefs: {np.array(pht_results["local_betas"])[0]}')

plt.close('all')
plt.plot(pht_results['global_betas'][0], hase_results['global_betas'], ".", label = "all")
plt.plot(pht_results["global_betas"][0][-3:-1], hase_results["global_betas"][-3:-1], ".", label = "ec")
plt.legend()
plt.show()
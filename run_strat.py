from vantage6.client import Client
import numpy as np
import sys
import os
import datetime
import json
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../V6_implementation'))

from V6_implementation.utils2 import generate_v6_info, generate_data_settings, generate_classif_settings, post_vantage_task, average, get_results
from V6_implementation.run_constants import *
from V6_implementation.v6_LinReg_py.local_constants import *
from V6_implementation.run import run



client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "password")
client.setup_encryption(None)
ids = [org['id'] for org in client.collaboration.get(1)['organizations']]
# ids = [3]
image_name = "sgarst/association-analysis:1.8"
v6_info = generate_v6_info(client, image_name, ids, 1)

## data settings ##
model = "M3" # options : M1, M2, M3, M4, M5, M6, M7
normalize = "global" # options: global, local, none
use_age = False # whether to use age as a covariate
use_dm = True # whether to use dm as a covariate
use_deltas = False # whether to look at delta metabo/brainage
normalize_cat = True # whether to normalize categorical variables
stratify_settings = {
    "dm" : [0, 1],
    "dem" : [0, 1]
}

stratify_values = [[0, 1], [0, 1]]
data_settings = generate_data_settings(model, normalize, use_age, use_dm, use_deltas, normalize_cat)
# data_settings[STRATIFY_GROUPS] = stratify_categories

def run_strat(v6_info, data_settings, stratify_settings, model):
    # there should be a way to non-hardcode this, but for now not necessary
    data_settings[STRATIFY] = True
    data_settings[MODEL_COLS].append(DEMENTIA)
    for dm_val in stratify_settings[DM]:
        for ad_val in stratify_settings[DEMENTIA]:
            data_settings[STRATIFY_GROUPS] = [DM, DEMENTIA]
            data_settings[STRATIFY_VALUES] = [dm_val, ad_val]

            results, _ = run(v6_info, data_settings, 1, 10)


            date = datetime.datetime.now()
            file_str = f'stratify_model_{model}_dm_{dm_val}_dem_{ad_val}_{date.strftime("%d")}_{date.strftime("%m")}_{date.strftime("%Y")}'
            jo = json.dumps(results)

            with open(file_str + ".json", "w", encoding="utf8") as f:
                f.write(jo)


if __name__ == "__main__":
    run_strat(v6_info, data_settings, stratify_settings, model)
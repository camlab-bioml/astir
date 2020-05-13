import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import yaml
# import anndata as ad
from astir.astir import Astir

## Todo: We probably need a class
def from_csv_yaml(csv_input, marker_yaml, random_seed = 1234):
        df_gex = pd.read_csv(csv_input, index_col = 0)
        with open(marker_yaml, 'r') as stream:
            marker_dict = yaml.safe_load(stream)
        return Astir(df_gex, marker_dict, random_seed)

def anndata_reader(read_ann, marker_yaml, random_seed = 1234):
    ann = ad.read_h5ad(read_ann)
    with open(marker_yaml, 'r') as stream:
            marker_dict = yaml.safe_load(stream)
    return Astir(ann.obs, marker_dict, random_seed)

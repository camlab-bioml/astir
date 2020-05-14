import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import yaml
# import anndata as ad
from astir.astir import Astir

## Todo: We probably need a class
def from_csv_yaml(expr_csv, marker_yaml, design_csv = None, random_seed = 1234):
    df_gex = pd.read_csv(expr_csv, index_col = 0)

    if design_csv != None:
        design = pd.read_csv(design_csv, index_col=0)
    else:
        design = None
    with open(marker_yaml, 'r') as stream:
        marker_dict = yaml.safe_load(stream)
    return Astir(df_gex, marker_dict, design=design, random_seed=random_seed)

def anndata_reader(read_ann, marker_yaml, random_seed = 1234):
    ann = ad.read_h5ad(read_ann)
    with open(marker_yaml, 'r') as stream:
            marker_dict = yaml.safe_load(stream)
    return Astir(ann.obs, marker_dict, random_seed=random_seed)

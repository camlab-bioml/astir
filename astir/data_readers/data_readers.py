import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import yaml
# import anndata as ad
from astir.astir import Astir

## Todo: We probably need a class
def from_csv_yaml(csv_input, marker_yaml, design_csv = None, random_seed = 1234, include_beta = True):
        df_gex = pd.read_csv(csv_input, index_col = 0)

        design = None
        if design_csv is not None:
            design = pd.read_csv(design_csv, index_col=0)
        with open(marker_yaml, 'r') as stream:
            marker_dict = yaml.safe_load(stream)
        return Astir(df_gex, marker_dict, design, random_seed, include_beta)

def anndata_reader(read_ann, marker_yaml, random_seed = 1234):
    ann = ad.read_h5ad(read_ann)
    with open(marker_yaml, 'r') as stream:
            marker_dict = yaml.safe_load(stream)
    return Astir(ann.obs, marker_dict, random_seed=random_seed)

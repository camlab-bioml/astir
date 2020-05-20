import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import yaml
import os
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from astir.astir import Astir


def from_csv_yaml(csv_input: str, marker_yaml: str, design_csv = None, random_seed = 1234, include_beta = False):
    """Create an Astir object from an expression CSV and marker YAML

    :param csv_input: Path to input csv containing expression for cells (rows) by proteins (columns). First column is 
    cell identifier, and additional column names are gene identifiers.
    :type df_gex: str
    :param marker_yaml: Path to input YAML file containing marker gene information. Should include cell_type and cell_state entries. See documention.
    :type marker_dict: str
    :param design_csv: Path to design matrix as a CSV. Rows should be cells, and columns covariates. First column is cell identifier, and additional column names are covariate identifiers.
    """
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


def from_csv_dir_yaml(input_dir: str, marker_yaml: str, random_seed = 1234, include_beta=False):
    # Parse the input directory
    csv_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith("csv")]

    # Read to gene expression df and parse
    dfs = [pd.read_csv(f, index_col=0) for f in csv_files]
    df_gex = pd.concat(dfs, axis=0)

    # Construct a sample specific design matrix
    design_list = [np.repeat(str(i), dfs[i].shape[0]) for i in range(len(dfs))]
    design = OneHotEncoder().fit_transform(np.concatenate(design_list, axis=0).reshape(-1,1)).todense()
    design = design[:,:-1] # remove final column
    design = np.concatenate([np.ones((design.shape[0],1)), design], axis=1) # add in intercept!

    with open(marker_yaml, 'r') as stream:
        marker_dict = yaml.safe_load(stream)

    return Astir(df_gex, marker_dict, design, random_seed, include_beta)
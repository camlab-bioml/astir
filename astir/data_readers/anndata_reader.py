import pandas as pd
import anndata as ad
import yaml
from astir.astir import Astir

def anndata_reader(read_ann, marker_yaml, random_seed = 1234):
    ann = ad.read_h5ad(read_ann)
    with open(marker_yaml, 'r') as stream:
            marker_dict = yaml.safe_load(stream)
    return Astir(ann.obs, marker_dict, random_seed)

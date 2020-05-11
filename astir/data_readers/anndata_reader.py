import pandas as pd
import anndata as ad
from astir.astir import Astir

def anndata_reader(read_ann, marker_yaml, random_seed = 1234):
    ann = ad.read_h5ad(read_ann)
    return Astir(ann.obs, marker_yaml, random_seed)

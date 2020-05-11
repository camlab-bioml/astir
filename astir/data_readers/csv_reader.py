import pandas as pd
from astir.astir import Astir

def csv_reader(read_csv, marker_yaml, random_seed = 1234):
        df_gex = pd.read_csv(read_csv, index_col = 0)
        return Astir(df_gex, marker_yaml, random_seed)
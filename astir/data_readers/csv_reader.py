import pandas as pd
import yaml
from astir.astir import Astir

## Todo: We need a class
def csv_reader(csv_input, marker_yaml, random_seed = 1234):
        df_gex = pd.read_csv(csv_input, index_col = 0)
        with open(marker_yaml, 'r') as stream:
            marker_dict = yaml.safe_load(stream)
        return Astir(df_gex, marker_dict, random_seed)

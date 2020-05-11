from .astir import Astir
from .data_readers.csv_reader import *

__all__ = [
    "Astir"
    "csv_reader"
]

# ast = Astir("./BaselTMA_SP43_115_X4Y8.csv", "jackson-2020-markers.yml", 
#     "./BaselTMA_SP43_115_X4Y8_assignment.csv")
# ast.fit()
from .astir import Astir
from .data_readers.data_readers import *

__all__ = [
    "Astir",
    "NotClassifiableError",
    "from_csv_yaml"
]

# ast = Astir("./BaselTMA_SP43_115_X4Y8.csv", "jackson-2020-markers.yml", 
#     "./BaselTMA_SP43_115_X4Y8_assignment.csv")
# ast.fit()
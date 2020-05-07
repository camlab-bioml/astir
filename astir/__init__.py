from .astir import Astir

name = "Astir"

print(__package__)

__all__ = [
    "Astir" 
]

ast = Astir("./BaselTMA_SP43_115_X4Y8.csv", "jackson-2020-markers.yml", 
    "./BaselTMA_SP43_115_X4Y8_assignment.csv")
ast.fit()
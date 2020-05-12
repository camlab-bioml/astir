# astir

astir: Automated aSsignmenT of sIngle-cell pRoteomics

Software for automated assignment of cell type and state (phenotype) from single-cell proteomic data, with a focus on imaging mass cytometry (IMC).

## Installation

```bash
git clone https://github.com/camlab-bioml/astir.git
pip install astir
```

## Basic usage

From the command line:

```bash
astir expression_mat.csv markers.yml output_assignments.csv
```

From within Python with existing data:

```python
from astir import Astir

a = Astir(expr_df, marker_dict)
a.fit()

cell_probabilities = a.get_assignments()
```

From within Python reading from csv and yaml:

```python
from astir.data_readers import from_csv_yaml

a = from_csv_yaml('expression_mat.csv', 'markers.yml')

a.fit()

cell_probabilities = a.get_assignments()
```

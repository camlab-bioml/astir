# astir

astir: automated assignment of cell type and state (phenotype) from single-cell proteomic data, with a focus on imaging mass cytometry (IMC).

![Image of astir](doc/source/_static/figs/astir.png)

## Installation

```bash
git clone https://github.com/camlab-bioml/astir.git
cd astir
pip install -e .
```

## Basic usage

### Input

`astir` takes as input a cell-by-gene expression matrix, which can optionally be read in from a csv:

```csv
"","EGFR","Ruthenium_1","Ruthenium_2","Ruthenium_3","Ruthenium_4"
"BaselTMA_SP41_186_X5Y4_3679",0.346787047240784,0.962155972321163,0.330768187877474,1.21347557766054,1.26704845953417
"BaselTMA_SP41_153_X7Y5_246",0.833751713754184,1.07555159349581,0.419977137830632,1.36904891724053,1.38510442154998
"BaselTMA_SP41_20_X12Y5_197",0.110005567928629,0.908453513733461,0.301166333489085,1.28738891851379,1.30072755877247
"BaselTMA_SP41_14_X1Y8_84",0.282666026986334,0.865982850277527,0.35037342731126,1.24080330000694,1.26476734524879
```

and a python dictionary or equivalently `yaml` file relating markers to cell types and cell states:

```yaml
cell_states:
  RTK_signalling:
    - Her2
    - EGFR
...
cell_types:
  Epithelial (basal):
    - E-Cadherin
    - pan Cytokeratin
    - Cytokeratin 5
    - Cytokeratin 14
    - Her2
  B cells:
    - CD45
    - CD20
...
```


### Running

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
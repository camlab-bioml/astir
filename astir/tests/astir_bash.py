import argparse

import torch
import pandas as pd
import yaml

import os
import sys

print(sys.path)
module_path = os.path.abspath(os.path.join('.'))
print("####", module_path)
# if module_path not in sys.path:
#     sys.path.append(module_path)
# print(sys.path)

# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
# print(sys.path)

# module_path = os.path.abspath(os.path.join('../..'))
# print(module_path)
# if module_path not in sys.path:
#     sys.path.append(module_path)
# os.system("export PYTHONPATH=.")
#
# print(sys.path)

from astir.data_readers import from_csv_yaml


print("###############################1")

parser = argparse.ArgumentParser(description='Run astir')


parser.add_argument("criteria", help="Enter type to classify cells by type or enter state to classify cells by state")
parser.add_argument("expr_csv", help="Path to CSV expression matrix with cells as rows and proteins as columns. First column is cell ID")
parser.add_argument("marker_yaml", help="Path to YAML file of cell markers")
parser.add_argument("output_csv", help="Output CSV of cell assignment probabilities")
parser.add_argument("--design",
                    help="Path to design matrix CSV",
                    type=str,
                    default="None")
parser.add_argument("--max_epochs",
                    help="Number of training epochs",
                    type=int,
                    default=50)
parser.add_argument("--learning_rate",
                    help="Learning rate",
                    type=float,
                    default=1e-3)
parser.add_argument("--batch_size",
                    help="Batch size",
                    type=int,
                    default=128)
parser.add_argument("--random_seed",
                    help="Random seed",
                    type=int,
                    default=1234)
parser.add_argument("--n_init",
                    help="number of inits",
                    type=int,
                    default=5)
parser.add_argument("--n_init_epochs",
                    help="number of epochs per inits",
                    type=int,
                    default=5)
parser.add_argument("--dtype",
                    help="data type",
                    type=str,
                    choices=["torch.float64", "torch.float32"],
                    default="torch.float64")
parser.add_argument("--delta_loss",
                    help="loss to convergence",
                    type=float,
                    default=1e-3)
parser.add_argument("--delta_loss_batch",
                    help="number of losses to calculate delta_loss",
                    type=int,
                    default=10)

args = parser.parse_args()

if args.design == "None":
    args.design = None

if args.dtype == "torch.float64":
    args.dtype = torch.float64
elif args.dtype == "torch.float32":
    args.dtype = torch.float32

a = from_csv_yaml(args.expr_csv, args.marker_yaml, design_csv=args.design, random_seed=args.random_seed)
print("###############################2")
if args.criteria == "type":
    a.fit_type(max_epochs = args.max_epochs,
        learning_rate = args.learning_rate,
        batch_size = args.batch_size,
       delta_loss=args.delta_loss,
       n_init=args.n_init,
        n_init_epochs = args.n_init_epochs)
    a.type_to_csv(args.output_csv)
elif args.criteria == "state":
    print("###############################3")
    a.fit_state(max_epochs=args.max_epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                delta_loss=args.delta_loss,
                n_init=args.n_init,
                n_init_epochs=args.n_init_epochs,
                delta_loss_batch=args.delta_loss_batch)
    a.state_to_csv(args.output_csv)
    read_output = pd.read_csv(args.output_csv, index_col=0)
    print("###############################4")
else:
    print("Please specify the criteria for classification. (astir type [option] / astir state [option])")

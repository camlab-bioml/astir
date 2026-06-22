import argparse
import os
import subprocess
import torch
from astir.data import from_csv_yaml


parser = argparse.ArgumentParser(description="Run astir")

subparsers = parser.add_subparsers(
    title="Criterion",
    help="specify if the cells are to be classified by types or states or convert data",
)

type_parser = subparsers.add_parser("type")

type_parser.add_argument(
    "--prob_max",
    "-pm",
    help="Classify cell types using max probability instead of threshold",
    default=False,
    action="store_true",
)


def parser_setup(parser):
    parser.add_argument(
        "expr_csv",
        help="CSV expression matrix (cells x proteins, first column = cell ID)",
    )
    parser.add_argument(
        "marker_yaml",
        help="YAML file of cell markers",
    )
    parser.add_argument(
        "output_csv",
        help="Output CSV of cell assignments",
    )
    parser.add_argument(
        "--design",
        "-d",
        type=str,
        default="None",
        help="Design matrix CSV",
    )
    parser.add_argument(
        "--random_seed",
        "-s",
        type=int,
        default=1234,
        help="Random seed",
    )
    parser.add_argument(
        "--dtype",
        "-t",
        type=str,
        default="torch.float64",
        help="torch.float32 or torch.float64",
    )
    parser.add_argument(
        "--max_epochs",
        "-m",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--learning_rate",
        "-r",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--delta_loss",
        "-l",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--n_init",
        "-n",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--n_init_epochs",
        "-i",
        type=int,
        default=5,
    )


parser_setup(type_parser)

state_parser = subparsers.add_parser("state")
parser_setup(state_parser)

state_parser.add_argument(
    "--delta_loss_batch",
    type=int,
    default=10,
)
state_parser.add_argument(
    "--const",
    "-c",
    type=int,
    default=2,
)
state_parser.add_argument(
    "--dropout_rate",
    type=float,
    default=0.0,
)
state_parser.add_argument(
    "--batch_norm",
    type=bool,
    default=False,
)


convert_parser = subparsers.add_parser("convert")

convert_parser.add_argument("in_rds", type=str)
convert_parser.add_argument("out_csv", type=str)

convert_parser.add_argument("--assay", "-a", type=str, default="logcounts")
convert_parser.add_argument("--design_col", type=str, default="")
convert_parser.add_argument("--design_csv", type=str, default="")
convert_parser.add_argument("--winsorize", "-w", type=float, default=0.0)


def _resolve_dtype(dtype_str):
    if dtype_str == "torch.float64":
        return torch.float64
    if dtype_str == "torch.float32":
        return torch.float32
    return torch.float64

def run_type(args):
    design = None if args.design == "None" else args.design
    dtype = _resolve_dtype(args.dtype)

    a = from_csv_yaml(
        args.expr_csv,
        args.marker_yaml,
        design_csv=design,
        random_seed=args.random_seed,
        dtype=dtype,
    )

    a.fit_type(
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        delta_loss=args.delta_loss,
        n_init=args.n_init,
        n_init_epochs=args.n_init_epochs,
    )

    assignment_type = "max" if args.prob_max else "threshold"
    a.type_to_csv(args.output_csv, assignment_type=assignment_type)

    prob_fn = os.path.splitext(args.output_csv)[0] + ".probabilities.csv"
    a.get_celltype_probabilities().to_csv(prob_fn)

def run_state(args):
    design = None if args.design == "None" else args.design
    dtype = _resolve_dtype(args.dtype)

    a = from_csv_yaml(
        args.expr_csv,
        args.marker_yaml,
        design_csv=design,
        random_seed=args.random_seed,
        dtype=dtype,
    )

    a.fit_state(
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        delta_loss=args.delta_loss,
        n_init=args.n_init,
        n_init_epochs=args.n_init_epochs,
        delta_loss_batch=args.delta_loss_batch,
        const=args.const,
        dropout_rate=args.dropout_rate,
        batch_norm=args.batch_norm,
    )

    a.state_to_csv(args.output_csv)


def convert_rds(args):

    script_path = os.path.join(
        os.path.dirname(__file__),
        "data/rds_reader.R",
    )

    cmd = [
        "Rscript",
        script_path,
        args.in_rds,
        args.out_csv,
        "--assay",
        args.assay,
        "--design_col",
        args.design_col,
        "--design_csv",
        args.design_csv,
        "--winsorize",
        str(args.winsorize),
    ]

    subprocess.call(cmd, cwd=os.getcwd())

type_parser.set_defaults(func=run_type)
state_parser.set_defaults(func=run_state)
convert_parser.set_defaults(func=convert_rds)

def main():
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
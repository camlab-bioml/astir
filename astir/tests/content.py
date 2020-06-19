#!/usr/bin/env python

import argparse

import torch
import pandas as pd

from astir.data_readers import from_csv_yaml


parser = argparse.ArgumentParser(description='Run')

parser.add_argument("filename", help="filename")

# parser.add_argument("--content",
#                     help="content",
#                     type=str,
#                     default="")

write_content = "This is a file for testing!!!!!!!!!!!!!"

args = parser.parse_args()

with open(args.filename, "w") as f:
    f.write(write_content)

f.close()

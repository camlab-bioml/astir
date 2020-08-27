#!/usr/bin/env bash

set -e
set -x

# mypy astir --disallow-untyped-defs
# black astir tests --check
black tests --check
black astir/astir.py
black astir tests --check
# isort --multi-line=3 --trailing-comma --force-grid-wrap=0 --combine-as --line-width 88 --recursive --check-only --thirdparty astir astir tests

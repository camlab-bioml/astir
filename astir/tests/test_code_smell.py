""" Tests the following for SCDataset:
- Whether all methods have docstrings
- Whether all methods contain type hints and return type hints
- Whether return type hints are correct
- Any other code smells like a subclass has the same method definitions as
its superclass
"""
import unittest

from astir import Astir
from astir.models import (
    CellStateModel,
    CellTypeModel,
    StateRecognitionNet,
    TypeRecognitionNet,
    AstirModel,
)
from astir.data import SCDataset


class TestCodeSmells(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCodeSmells, self).__init__(*args, **kwargs)
        self.classes = [
            Astir,
            CellStateModel,
            CellTypeModel,
            StateRecognitionNet,
            TypeRecognitionNet,
            AstirModel,
            SCDataset,
        ]
        self.class_paths = [
            "astir/astir.py",
            "astir/models/cellstate.py",
            "astir/models/celltype.py",
            "astir/models/cellstate_recognet.py",
            "astir/models/celltype_recognet.py",
            "astir/models/abstract.py",
            "astir/data/scdataset.py",
        ]

    def test_docstrings_exists_all_methods(self):
        for cl in self.classes:
            cl_name = cl.__name__
            class_dict = dict(cl.__dict__)

            class_dict.pop("__module__")
            class_dict.pop("__init__")

            no_doc = []
            for key, value in class_dict.items():
                if value.__doc__ is None:
                    no_doc.append(key)

            err_msg = "{} has methods without docstring: ".format(cl_name)
            for method in no_doc:
                err_msg += method + ", "
            self.assertTrue(no_doc == [], err_msg[:-2])

    def test_type_hints_exist_all_methods(self):
        from typing import get_type_hints
        from inspect import signature

        for cl in self.classes:
            cl_name = cl.__name__
            class_dict = dict(cl.__dict__)

            class_dict.pop("__module__")
            class_dict.pop("__doc__")

            no_type_hint = []
            no_return_hint = []
            for key, value in class_dict.items():
                try:
                    type_hints = get_type_hints(value)
                except:
                    continue

                if not type_hints.__contains__("return"):
                    no_return_hint.append(key)
                else:
                    type_hints.pop("return")
                params = list(signature(value).parameters)
                params.remove("self")

                if len(params) != len(type_hints):
                    no_type_hint.append(key)

            err_msg_type_hint = "{} is missing type hints for methods: " "".format(
                cl_name
            )
            for method in no_type_hint:
                err_msg_type_hint += method + ", "

            err_msg_return_hint = "{} is missing return hints for " "methods: ".format(
                cl_name
            )
            for method in no_return_hint:
                err_msg_return_hint += method + ", "
            self.assertTrue(no_type_hint == [], err_msg_type_hint[:-2])
            self.assertTrue(no_return_hint == [], err_msg_return_hint[:-2])

    def test_correct_return_types_and_detect_code_smells(self):
        import subprocess, rootpath, os

        for cl_path in self.class_paths:
            root_dir = rootpath.detect()
            path = os.path.join(root_dir, cl_path)

            process = subprocess.Popen(
                ["mypy", path, "--ignore-missing-imports", "--no-site-packages"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            stdout, stderr = process.communicate()
            stdout = stdout.decode("utf-8")
            stderr = stderr.decode("utf-8")
            output = str(stdout).split("\n")

            errors = []
            for line in output:
                if line.__contains__("error:"):
                    errors.append(line)

            self.assertTrue(
                errors == [],
                "Following errors were produced by "
                "MyPy: \n{}".format("\n".join(errors)),
            )
            self.assertTrue(
                stderr == "", "Following error were produced by MyPy: {}".format(stderr)
            )

    def test_data_reader_code_smell(self):
        from typing import get_type_hints
        from inspect import signature
        from astir.data.data_readers import (
            from_csv_yaml,
            from_csv_dir_yaml,
            from_loompy_yaml,
            from_anndata_yaml,
        )

        funcs = [from_csv_yaml, from_csv_dir_yaml, from_loompy_yaml, from_anndata_yaml]

        # Test whether all of them have type hints
        param_msgs = []
        docstring_msgs = []
        for func in funcs:
            type_hints = get_type_hints(func)
            param_with_hints = list(type_hints.keys())
            all_params = list(signature(func).parameters)

            param_diff = list(set(all_params) - set(param_with_hints))

            for param in param_diff:
                param_msgs.append(
                    "astir.data.data_readers.{} needs type hint "
                    "for parameter {}".format(func.__name__, param)
                )

            # Docstring test
            if func.__doc__ is None:
                docstring_msgs.append(
                    "astir.data.data_readers.{} needs a "
                    "docstring".format(func.__name__)
                )

        self.assertTrue(param_msgs == [], "\n".join(param_msgs))
        self.assertTrue(docstring_msgs == [], "\n".join(docstring_msgs))


if __name__ == "__main__":
    unittest.main()

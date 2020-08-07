import unittest
import os
import nbformat
import rootpath
import warnings

from nbconvert.preprocessors import ExecutePreprocessor


def run_notebook(notebook_path):
    nb_name, _ = os.path.splitext(os.path.basename(notebook_path))

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Configure the notebook execution mode
    proc = ExecutePreprocessor(timeout=600, kernel_name="python3")
    proc.allow_errors = True

    # Run the notebook
    root_path = rootpath.detect()
    path = root_path + "/docs/tutorials/notebooks"
    proc.preprocess(nb, {"metadata": {"path": path}})
    # Collect all errors
    errors = []
    for cell in nb.cells:
        if "outputs" in cell:
            for output in cell["outputs"]:
                if output.output_type == "error":
                    errors.append(output)

    return nb, errors


class TestNotebook(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestNotebook, self).__init__(*args, **kwargs)
        self.maxDiff = None

    def test_for_errors(self):
        warnings.filterwarnings("ignore")
        root_path = rootpath.detect()
        dirname = os.path.join(root_path, "docs/tutorials/notebooks")

        nb_names = [
            os.path.join(dirname, fn)
            for fn in os.listdir(dirname)
            if os.path.splitext(fn)[1] == ".ipynb"
        ]

        for fn in nb_names:
            _, errors = run_notebook(fn)
            self.assertEqual(errors, [], "Unexpected error in {}".format(fn))


if __name__ == "__main__":
    unittest.main()

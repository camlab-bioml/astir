import contextlib
import os
import warnings
from unittest import TestCase

import numpy as np
import pandas as pd
import torch
import yaml
import h5py
import subprocess as sp
import os

from astir import Astir
from astir.data import from_csv_yaml, from_csv_dir_yaml, from_anndata_yaml
from astir.data import SCDataset


class TestAstir(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestAstir, self).__init__(*args, **kwargs)
        warnings.filterwarnings("ignore", category=UserWarning)
        self.expr_csv_file = os.path.join(
            os.path.dirname(__file__), "test-data/test_data.csv"
        )
        self.marker_yaml_file = os.path.join(
            os.path.dirname(__file__), "test-data/jackson-2020-markers.yml"
        )
        self.design_file = os.path.join(
            os.path.dirname(__file__), "test-data/design.csv"
        )
        self.test_dir = os.path.join(
            os.path.dirname(__file__), "test-data/test-dir-read"
        )
        self.adata_file = os.path.join(
            os.path.dirname(__file__), "test-data/adata_small.h5ad"
        )

        self.expr = pd.read_csv(self.expr_csv_file, index_col=0)
        with open(self.marker_yaml_file, "r") as stream:
            self.marker_dict = yaml.safe_load(stream)

        self.a = Astir(self.expr, self.marker_dict)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_basic_instance_creation(self):
        """ Tests basic instance creation
        """

        self.assertIsInstance(self.a, Astir)
        # self.assertTrue(isinstance(a, str))

    def test_csv_reading(self):
        """ Test from_csv_yaml function
        """
        a = from_csv_yaml(self.expr_csv_file, self.marker_yaml_file)

        self.assertIsInstance(a, Astir)

    def test_dir_reading(self):

        a = from_csv_dir_yaml(self.test_dir, self.marker_yaml_file)

        self.assertIsInstance(a, Astir)

        ## Make sure the design matrix has been constructed correctly
        self.assertTrue(a._type_dset._design.shape[0] == len(a._type_dset))
        files = os.listdir(self.test_dir)
        files = [f for f in files if f.endswith(".csv")]
        self.assertTrue(a._type_dset._design.shape[1] == len(files))

    def test_csv_reading_with_design(self):

        a = from_csv_yaml(
            self.expr_csv_file, self.marker_yaml_file, design_csv=self.design_file
        )

        self.assertIsInstance(a, Astir)

    def test_fitting_type(self):

        epochs = 2
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                self.a.fit_type(max_epochs=epochs)

        # Check probability matrix looks ok
        probabilities = self.a.get_celltype_probabilities()

        self.assertTrue(probabilities.shape[0] == self.expr.shape[0])
        self.assertIsInstance(probabilities, pd.DataFrame)

        # Check assignments look ok
        assignments = self.a.get_celltypes()
        self.assertIsInstance(assignments, pd.DataFrame)
        self.assertTrue(assignments.shape[0] == self.expr.shape[0])
        self.assertTrue(assignments.columns[0] == "cell_type")

        # Check diagnostics look ok
        type_diagnostics = self.a.diagnostics_celltype(threshold=0.2, alpha=0)
        self.assertIsInstance(type_diagnostics, pd.DataFrame)
        self.assertTrue(
            type_diagnostics.shape[1] == 7
        )  # make sure we have the standard 6 columns

    def test_no_overlap(self):
        bad_file = os.path.join(os.path.dirname(__file__), "test-data/bad_data.csv")
        bad_data = pd.read_csv(bad_file)
        raised = False
        try:
            test = Astir(bad_data, self.marker_dict)
        except (RuntimeError):
            raised = True
        self.assertTrue(raised == True)

    def test_missing_marker(self):
        bad_marker = os.path.join(os.path.dirname(__file__), "test-data/bad_marker.yml")
        with open(bad_marker, "r") as stream:
            bad_dict = yaml.safe_load(stream)
        raised = False
        try:
            test = Astir(self.expr, bad_dict)
            test.fit_type()
        except (RuntimeError):
            raised = True
        self.assertTrue(raised)

    # # Uncomment below test functions to test private variables
    # # Commented it out because these tests can be highly overlapping with
    # # future unittests

    def test_sanitize_dict_state(self):
        """ Testing the method _sanitize_dict
        """
        expected_state_dict = self.marker_dict["cell_states"]
        (_, actual_state_dict, _) = self.a._sanitize_dict(self.marker_dict)

        expected_state_dict = {
            i: sorted(j) if isinstance(j, list) else j
            for i, j in expected_state_dict.items()
        }
        actual_state_dict = {
            i: sorted(j) if isinstance(j, list) else j
            for i, j in actual_state_dict.items()
        }

        self.assertDictEqual(
            expected_state_dict,
            actual_state_dict,
            "state_dict is different from its expected value",
        )

    def test_state_names(self):
        """ Test _state_names field
        """
        expected_state_names = sorted(self.marker_dict["cell_states"].keys())
        (_, actual_state_dict, _) = self.a._sanitize_dict(self.marker_dict)
        actual_state_names = sorted(actual_state_dict.keys())

        self.assertListEqual(
            expected_state_names, actual_state_names, "unexpected state_names value"
        )

    def test_celltype_same_seed_same_result(self):
        """ Test whether the loss after one epoch one two different models
        with the same random seed have the same losses after one epochs
        """
        warnings.filterwarnings("ignore", category=UserWarning)
        model1 = Astir(
            input_expr=self.expr,
            marker_dict=self.marker_dict,
            design=None,
            random_seed=42,
        )
        model2 = Astir(
            input_expr=self.expr,
            marker_dict=self.marker_dict,
            design=None,
            random_seed=42,
        )

        model1.fit_type(max_epochs=10)
        model1_loss = model1.get_type_losses()
        model2.fit_type(max_epochs=10)
        model2_loss = model2.get_type_losses()

        self.assertTrue(np.abs(model1_loss - model2_loss)[-1] < 1e-6)

    def test_celltype_diff_seed_diff_result(self):
        """ Test whether the loss after one epoch one two different models
        with the different random seed have different losses after one epoch
        """
        warnings.filterwarnings("ignore", category=UserWarning)
        model1 = Astir(
            input_expr=self.expr,
            marker_dict=self.marker_dict,
            design=None,
            random_seed=42,
        )
        model2 = Astir(
            input_expr=self.expr,
            marker_dict=self.marker_dict,
            design=None,
            random_seed=1234,
        )

        model1.fit_type(max_epochs=10)
        model1_loss = model1.get_type_losses()
        model2.fit_type(max_epochs=10)
        model2_loss = model2.get_type_losses()

        self.assertFalse(np.abs(model1_loss - model2_loss)[-1] < 1e-6)

    def test_cellstate_same_seed_same_result(self):
        """ Test whether the loss after one epoch one two different models
        with the same random seed have the same losses after one epochs
        """
        warnings.filterwarnings("ignore", category=UserWarning)
        model1 = Astir(
            input_expr=self.expr,
            marker_dict=self.marker_dict,
            design=None,
            random_seed=42,
        )
        model2 = Astir(
            input_expr=self.expr,
            marker_dict=self.marker_dict,
            design=None,
            random_seed=42,
        )

        model1.fit_state(max_epochs=5)
        model1_loss = model1.get_state_losses()
        model2.fit_state(max_epochs=5)
        model2_loss = model2.get_state_losses()

        self.assertTrue(np.abs(model1_loss - model2_loss)[-1] < 1e-6)

    # @pytest.mark.filterwarnings("ignore")
    def test_cellstate_diff_seed_diff_result(self):
        """ Test whether the loss after one epoch one two different models
        with the different random seed have different losses after one epoch
        """
        warnings.filterwarnings("ignore", category=UserWarning)
        model1 = Astir(
            input_expr=self.expr,
            marker_dict=self.marker_dict,
            design=None,
            random_seed=42,
        )
        model2 = Astir(
            input_expr=self.expr,
            marker_dict=self.marker_dict,
            design=None,
            random_seed=1234,
        )

        model1.fit_state(max_epochs=5)
        model1_loss = model1.get_state_losses()
        model2.fit_state(max_epochs=5)
        model2_loss = model2.get_state_losses()

        self.assertFalse(np.abs(model1_loss - model2_loss)[-1] < 1e-6)

    def test_cellstate_assignment(self):
        warnings.filterwarnings("ignore", category=UserWarning)
        self.a.fit_state(max_epochs=50, n_init=1)

        state_assignments = self.a.get_cellstates()

        n_classes = len(list(self.marker_dict.keys()))

        self.assertTrue(state_assignments.shape, (len(self.expr), n_classes))

    def test_cellstate_predicted_assignment(self):
        warnings.filterwarnings("ignore", category=UserWarning)
        dset = SCDataset(
            expr_input=self.expr,
            marker_dict=self.marker_dict["cell_states"],
            design=None,
            include_other_column=False,
            device=self._device,
        )

        self.a.fit_state(max_epochs=50, n_init=1)

        state_assignments = self.a.predict_cellstates(dset)

        self.assertTrue(state_assignments.shape, (len(dset), dset.get_n_classes()))

    def test_celltype_assignment(self):
        warnings.filterwarnings("ignore", category=UserWarning)
        self.a.fit_type(max_epochs=50, n_init=1)

        type_assignments = self.a.get_celltypes()

        n_classes = len(list(self.marker_dict.keys()))

        self.assertTrue(type_assignments.shape, (len(self.expr), n_classes + 1))

    def test_celltype_predicted_assignment(self):
        warnings.filterwarnings("ignore", category=UserWarning)

        self.a.fit_type(max_epochs=50, n_init=1)

        type_predict = self.a.predict_celltypes()
        type_assignment = self.a.get_celltype_probabilities()
        comp = type_predict == type_assignment
        self.assertTrue(comp.all().all())

    def test_adata_reading(self):
        ast = from_anndata_yaml(
            self.adata_file,
            self.marker_yaml_file,
            protein_name="protein",
            cell_name="cell_name",
            batch_name="batch",
        )

        self.assertTrue(ast.get_type_dataset().get_n_features() == 14)
        self.assertTrue(ast.get_type_dataset().get_n_classes() == 6)
        self.assertIsInstance(ast.get_type_dataset().get_exprs(), torch.Tensor)
        self.assertEqual(ast.get_type_dataset().get_exprs().shape[0], 10)

    def test_cellstate_diagnostics(self):
        warnings.filterwarnings("ignore", category=UserWarning)
        self.a.fit_state(max_epochs=50, n_init=1)

        state_diagnostics = self.a.diagnostics_cellstate()
        self.assertIsInstance(state_diagnostics, pd.DataFrame)

    def test_type_hdf5_summary(self):
        hdf5_summary = "celltype_summary.hdf5"
        info = {
            "max_epochs": 5,
            "learning_rate": 0.001,
            "batch_size": 24,
            "delta_loss": 0.001,
            "n_init": 1,
            "n_init_epochs": 1,
        }
        self.a.fit_type(
            max_epochs=info["max_epochs"],
            learning_rate=info["learning_rate"],
            batch_size=info["batch_size"],
            delta_loss=info["delta_loss"],
            n_init=info["n_init"],
            n_init_epochs=info["n_init_epochs"],
        )
        self.a.save_models(hdf5_summary)
        params = list(self.a.get_type_model().get_data().items()) + list(
            self.a.get_type_model().get_variables().items()
        )

        recog_params = []
        for key, val in self.a.get_type_model().get_recognet().named_parameters():
            recog_params.append((key, val.detach().cpu().numpy()))
        same = True
        with h5py.File(hdf5_summary, "r") as f:
            f_params = f["/celltype_model/parameters"]
            for key, val in params:
                if not (val.detach().cpu().numpy() == f_params[key][()]).all().all():
                    same = False
            f_recog = f["/celltype_model/recog_net"]
            for key, val in recog_params:
                if not (f_recog[key][()] == val).all():
                    same = False
            f_info = f["/celltype_model/run_info"]
            for key, val in info.items():
                if val != f_info[key][()]:
                    same = False
            if not (
                self.a.get_type_model().get_losses().cpu().numpy()
                == f["/celltype_model/losses"]["losses"][()]
            ).all():
                same = False
        self.assertTrue(same)

    def test_state_summary(self):
        hdf5_summary = "cellstate_summary.hdf5"
        info = {
            "max_epochs": 5,
            "learning_rate": 0.001,
            "batch_size": 24,
            "delta_loss": 0.001,
            "n_init": 1,
            "n_init_epochs": 1,
            "delta_loss_batch": 2,
        }
        self.a.fit_state(
            max_epochs=info["max_epochs"],
            learning_rate=info["learning_rate"],
            batch_size=info["batch_size"],
            delta_loss=info["delta_loss"],
            n_init=info["n_init"],
            n_init_epochs=info["n_init_epochs"],
            delta_loss_batch=info["delta_loss_batch"],
        )
        self.a.save_models(hdf5_summary)
        params = list(self.a.get_state_model().get_data().items()) + list(
            self.a.get_state_model().get_variables().items()
        )
        recog_params = []
        for key, val in self.a.get_state_model().get_recognet().named_parameters():
            recog_params.append((key, val.detach().cpu().numpy()))
        same = True
        with h5py.File(hdf5_summary, "r") as f:
            f_params = f["/cellstate_model/parameters"]
            for key, val in params:
                if not (val.detach().cpu().numpy() == f_params[key][()]).all().all():
                    same = False
            f_recog = f["/cellstate_model/recog_net"]
            for key, val in recog_params:
                if not (f_recog[key][()] == val).all():
                    same = False
            f_info = f["/cellstate_model/run_info"]
            for key, val in info.items():
                if val != f_info[key][()]:
                    same = False
            if not (
                self.a.get_state_model().get_losses().cpu().numpy()
                == f["/cellstate_model/losses"]["losses"][()]
            ).all():
                same = False
        self.assertTrue(same)

    def test_hierarchy_assignment(self):
        self.a.fit_type(max_epochs=5, n_init=1, n_init_epochs=1)
        original_assignment = self.a.get_celltype_probabilities()
        hier_dict = self.a.get_hierarchy_dict()
        # expected_assignment = pd.DataFrame()
        # for key, cells in hier_dict.items():
        #     expected_assignment[key] = original_assignment[cells].sum(axis=1)
        actual_assignment = self.a.assign_celltype_hierarchy(depth=3)
        # self.assertTrue((original_assignment == actual_assignment).all().all())
        for cell in actual_assignment.columns:
            self.assertTrue(
                (actual_assignment[cell] == original_assignment[cell]).all()
            )

    def test_hdf5_load(self):
        hdf5_summary = "celltype_summary.hdf5"
        orig_ast = Astir(self.expr, self.marker_dict)
        orig_ast.fit_type(max_epochs=5, n_init=1, n_init_epochs=1)
        orig_ast.fit_state(max_epochs=5, n_init=1, n_init_epochs=1)
        orig_ast.save_models(hdf5_summary)
        new_ast = Astir()
        new_ast.load_model(hdf5_summary)

        orig_type_run_info = orig_ast.get_type_run_info()
        orig_state_run_info = orig_ast.get_state_run_info()
        new_type_run_info = new_ast.get_type_run_info()
        new_state_run_info = new_ast.get_state_run_info()
        for key, val in orig_type_run_info.items():
            if val != new_type_run_info[key]:
                raise AssertionError(
                    "variable "
                    + key
                    + " is different in original model and loaded model"
                )
        for key, val in orig_state_run_info.items():
            if val != new_state_run_info[key]:
                raise AssertionError(
                    "variable "
                    + key
                    + " is different in original model and loaded model"
                )

        orig_type_losses = orig_ast.get_type_losses()
        orig_state_losses = orig_ast.get_state_losses()
        new_type_losses = new_ast.get_type_losses()
        new_state_losses = new_ast.get_state_losses()
        if not (
            all(orig_type_losses == new_type_losses)
            and all(orig_state_losses == new_state_losses)
        ):
            raise AssertionError("loss is different in original model and loaded model")

    # def test_make_html(self):
    #     path = os.path.dirname(os.path.realpath(__file__)) + "/../../docs"
    #     cmd = ["make", "html"]
    #     p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, cwd=path)
    #     output, err = p.communicate(b"input data that is passed to subprocess' stdin")
    #     print(err)
    #     print(path)
    #     rc = p.returncode
    #     self.assertTrue(rc == 0)

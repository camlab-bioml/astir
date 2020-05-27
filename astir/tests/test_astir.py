from unittest import TestCase
import pandas as pd
import os

import pytest
import yaml
import numpy as np
import warnings

from astir import Astir
from astir.data_readers import from_csv_yaml, from_csv_dir_yaml

import os, contextlib


class TestAstir(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestAstir, self).__init__(*args, **kwargs)

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

        self.expr = pd.read_csv(self.expr_csv_file)
        with open(self.marker_yaml_file, "r") as stream:
            self.marker_dict = yaml.safe_load(stream)

        self.a = Astir(self.expr, self.marker_dict)

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
        self.assertTrue(a._design.shape[0] == len(a._type_dset))
        files = os.listdir(self.test_dir)
        files = [f for f in files if f.endswith(".csv")]
        self.assertTrue(a._design.shape[1] == len(files))

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

        assignments = self.a.get_celltype_assignments()
        losses = self.a.get_type_losses()

        self.assertTrue(assignments.shape[0] == self.expr.shape[0])

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
        except (RuntimeError):
            raised = True
        self.assertTrue(raised == True)

    # # Uncomment below test functions to test private variables
    # # Commented it out because these tests can be highly overlapping with
    # # future unittests

    def test_sanitize_dict_state(self):
        """ Testing the method _sanitize_dict
        """
        expected_state_dict = self.marker_dict["cell_states"]
        (_, actual_state_dict) = self.a._sanitize_dict(self.marker_dict)

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
        (_, actual_state_dict) = self.a._sanitize_dict(self.marker_dict)
        actual_state_names = sorted(actual_state_dict.keys())

        self.assertListEqual(
            expected_state_names, actual_state_names, "unexpected state_names value"
        )

    # def test_state_marker_genes(self):
    #     """ Test _mstate_genes field
    #     """
    #     state_dict = self.marker_dict["cell_states"]
    #     expected_marker_genes = sorted(
    #         list(set([l for s in state_dict.values() for l in s]))
    #     )
    #     actual_marker_genes = sorted(self.a._mstate_genes)
    #
    #     self.assertListEqual(
    #         expected_marker_genes, actual_marker_genes, "unexpected marker_genes value"
    #     )

    # def test_constant_N(self):
    #     """ Test constant N
    #     """
    #     expected_N = self.expr.shape[0]
    #     actual_N = self.a._N
    #
    #     self.assertEqual(expected_N, actual_N, "unexpected N value")

    # def test_state_constants_G_C(self):
    #     """ Test state constants G and C
    #     """
    #     state_dict = self.marker_dict["cell_states"]
    #     expected_G = len([item for l in state_dict.values() for item in l])
    #     expected_C = len(self.marker_dict["cell_states"])
    #
    #     actual_G = self.a._G_s
    #     actual_C = self.a._C_s
    #
    #     self.assertEqual(expected_G, actual_G, "unexpected G value")
    #     self.assertEqual(expected_C, actual_C, "unexpected C value")

    # def test_core_names(self):
    #     """ Test core names
    #     """
    #     expected_core_names = sorted(self.expr.index)
    #     actual_core_names = sorted(self.a._core_names)
    #
    #     self.assertListEqual(
    #         expected_core_names, actual_core_names, "unexpected _core_names value"
    #     )

    # def test_expr_gene_names(self):
    #     """ Test _expression_genes
    #     """
    #     expected_gene_names = sorted(self.expr.columns)
    #     actual_gene_names = sorted(self.a._expression_genes)
    #
    #     self.assertListEqual(
    #         expected_gene_names, actual_gene_names, "unexpected _expression_genes value"
    #     )

    # def test_state_mat(self):
    #     """ Test state_mat variable
    #     """
    #     state_dict = self.marker_dict["cell_states"]
    #     marker_genes = list(set([l for s in state_dict.values() for l in s]))
    #     state_names = state_dict.keys()
    #     G = len([item for l in state_dict.values() for item in l])
    #     C = len(self.marker_dict["cell_states"])
    #     expected_state_mat = np.zeros(shape=(G, C))
    #
    #     for g, gene in enumerate(marker_genes):
    #         for ct, state in enumerate(state_names):
    #             if gene in state_dict[state]:
    #                 expected_state_mat[g, ct] = 1
    #
    #     actual_state_mat = self.a._state_mat
    #
    #     np.testing.assert_array_equal(expected_state_mat, actual_state_mat)

    # def test_get_cellstates(self):
    #     """ Tests get_cellstates()
    #     """
    #     self.a.fit_state(max_epochs=2)
    #     state_assignments = self.a.get_cellstates()
    #
    #     self.assertTrue(state_assignments.shape[0] == len(self.a._core_names))
    #     self.assertTrue(state_assignments.shape[1] == len(self.a._cell_states))

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
            include_beta=True,
        )
        model2 = Astir(
            input_expr=self.expr,
            marker_dict=self.marker_dict,
            design=None,
            random_seed=42,
            include_beta=True,
        )

        model1.fit_type(max_epochs=10)
        model1_loss = model1.get_type_losses()
        model2.fit_type(max_epochs=10)
        model2_loss = model2.get_type_losses()

        self.assertTrue(np.abs(model1_loss - model2_loss)[-1] < 1e-6)

    # @pytest.mark.filterwarnings("ignore")
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
            include_beta=True,
        )
        model2 = Astir(
            input_expr=self.expr,
            marker_dict=self.marker_dict,
            design=None,
            random_seed=1234,
            include_beta=True,
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
            include_beta=True,
        )
        model2 = Astir(
            input_expr=self.expr,
            marker_dict=self.marker_dict,
            design=None,
            random_seed=42,
            include_beta=True,
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
            include_beta=True,
        )
        model2 = Astir(
            input_expr=self.expr,
            marker_dict=self.marker_dict,
            design=None,
            random_seed=1234,
            include_beta=True,
        )

        model1.fit_state(max_epochs=5)
        model1_loss = model1.get_state_losses()
        model2.fit_state(max_epochs=5)
        model2_loss = model2.get_state_losses()

        self.assertFalse(np.abs(model1_loss - model2_loss)[-1] < 1e-6)

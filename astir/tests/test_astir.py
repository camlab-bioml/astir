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

        assignments = self.a.get_celltype_probabilities()
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

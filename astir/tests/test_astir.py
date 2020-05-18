from unittest import TestCase
import pandas as pd
import os
import yaml

from astir import Astir
from astir.data_readers import from_csv_yaml


class TestAstir(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestAstir, self).__init__(*args, **kwargs)

        self.expr_csv_file = os.path.join(os.path.dirname(__file__), 'test-data/test_data.csv')
        self.marker_yaml_file = os.path.join(os.path.dirname(__file__), 'test-data/jackson-2020-markers.yml')
        self.design_file = os.path.join(os.path.dirname(__file__), 'test-data/design.csv')

        self.expr = pd.read_csv(self.expr_csv_file)
        with open(self.marker_yaml_file, 'r') as stream:
            self.marker_dict = yaml.safe_load(stream)

        self.a = Astir(self.expr, self.marker_dict)

    def test_basic_instance_creation(self):

        self.assertIsInstance(self.a, Astir)
        # self.assertTrue(isinstance(a, str))

    def test_csv_reading(self):

        a = from_csv_yaml(self.expr_csv_file, self.marker_yaml_file)

        self.assertIsInstance(a, Astir)

    def test_csv_reading_with_design(self):

        a = from_csv_yaml(self.expr_csv_file, self.marker_yaml_file, design_csv=self.design_file)

        self.assertIsInstance(a, Astir)

    def test_fitting_type(self):

        epochs = 2
        self.a.fit_type(epochs=epochs)

        assignments = self.a.get_celltypes()
        losses = self.a.get_type_losses()

        self.assertTrue(assignments.shape[0] == self.expr.shape[0])
        self.assertTrue(len(losses) == epochs)

    def test_no_overlap(self):
        bad_file = os.path.join(os.path.dirname(__file__), 'test-data/bad_data.csv')
        bad_data = pd.read_csv(bad_file)
        raised = False
        try:
            test = Astir(bad_data, self.marker_dict)
        except(RuntimeError):
            raised = True
        self.assertTrue(raised == True)

    def missing_marker(self):
        bad_marker = os.path.join(os.path.dirname(__file__), 'test-data/bad_marker.yml')
        bad_file = os.path.join(os.path.dirname(__file__),
                                'test-data/bad_data.csv')
        bad_data = pd.read_csv(bad_file)
        with open(bad_marker, 'r') as stream:
            bad_dict = yaml.safe_load(stream)
        raised = False
        try:
            test = Astir(bad_data, self.marker_dict)
        except(RuntimeError):
            raised = True
        self.assertTrue(raised == True)

    def test_sanitize_dict_state(self):
        """ Testing the method _sanitize_dict
        """
        expected_state_dict = self.marker_dict["cell_states"]
        actual_state_dict = self.a._state_dict

        expected_state_dict = {i: sorted(j) if isinstance(j, list) else j
                               for i, j in expected_state_dict.items()}
        actual_state_dict = {i: sorted(j) if isinstance(j, list) else j
                             for i, j in actual_state_dict.items()}

        self.assertDictEqual(expected_state_dict, actual_state_dict,
                             "state_dict is different from its expected value")

    def test_state_names(self):
        """ Test _state_names field
        """
        expected_state_names = sorted(self.marker_dict["cell_states"].keys())
        actual_state_names = sorted(self.a._cell_states)

        self.assertListEqual(expected_state_names, actual_state_names,
                             "unexpected state_names value")

    def test_state_marker_genes(self):
        """ Test _mstate_genes field
        """
        state_dict = self.marker_dict["cell_states"]
        expected_marker_genes = sorted(list(
            set([l for s in state_dict.values() for l in s])))
        actual_marker_genes = sorted(self.a._mstate_genes)

        self.assertListEqual(expected_marker_genes, actual_marker_genes,
                             "unexpected marker_genes value")

    def test_constant_N(self):
        """ Test constant N
        """
        expected_N = self.expr.shape[0]
        actual_N = self.a._N

        self.assertEqual(expected_N, actual_N, "unexpected N value")

    def test_state_constants_G_C(self):
        """ Test state constants G and C
        """
        state_dict = self.marker_dict["cell_states"]
        expected_G = len([item for l in state_dict.values() for item in l])
        expected_C = len(self.marker_dict["cell_states"])

        actual_G = self.a._G_s
        actual_C = self.a._C_s

        self.assertEqual(expected_G, actual_G, "unexpected G value")
        self.assertEqual(expected_C, actual_C, "unexpected C value")

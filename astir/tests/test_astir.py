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
        with open(bad_marker, 'r') as stream:
            bad_dict = yaml.safe_load(stream)
        raised = False
        try:
            test = Astir(bad_data, self.marker_dict)
        except(RuntimeError):
            raised = True
        self.assertTrue(raised == True)   


    

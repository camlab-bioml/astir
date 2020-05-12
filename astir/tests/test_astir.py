from unittest import TestCase
import pandas as pd
import os
import yaml

from astir import Astir
from astir.data_readers import csv_reader

class TestAstir(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestAstir, self).__init__(*args, **kwargs)

        self.expr_csv_file = os.path.join(os.path.dirname(__file__), 'test-data/test_data.csv')
        self.marker_yaml_file = os.path.join(os.path.dirname(__file__), 'test-data/jackson-2020-markers.yml')

        self.expr = pd.read_csv(self.expr_csv_file)
        with open(self.marker_yaml_file, 'r') as stream:
            self.marker_dict = yaml.safe_load(stream)

        self.a = Astir(self.expr, self.marker_dict)


    def test_basic_instance_creation(self):
    
        self.assertIsInstance(self.a, Astir)
        # self.assertTrue(isinstance(a, str))

    def test_csv_reading(self):

        a = csv_reader(self.expr_csv_file, self.marker_yaml_file)

        self.assertIsInstance(a, Astir)
    
    def test_fitting(self):

        epochs = 2
        self.a.fit(epochs=epochs)

        assignments = self.a.get_assignments()
        losses = self.a.get_losses()

        self.assertTrue(assignments.shape[0] == self.expr.shape[0])
        self.assertTrue(len(losses) == epochs)

    

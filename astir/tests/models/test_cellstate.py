from unittest import TestCase
import pandas as pd
import os
import yaml
import numpy as np

from astir.models import CellStateModel
from astir.data_readers import from_csv_yaml


class TestCellStateModel(TestCase):
    """ Unittest class for CellStateModel class

    This class assumes that all data initializating functions in Astir class
    are working
    """
    def __init__(self, *args, **kwargs):
        super(TestCellStateModel, self).__init__(*args, **kwargs)

        self.expr_csv_file = \
            os.path.join(os.path.dirname(__file__),
                         '../../test-data/sce.csv')
        self.marker_yaml_file = \
            os.path.join(os.path.dirname(__file__),
                         '../../test-data/jackson-2020-markers'
                         '.yml')

        self.expr = pd.read_csv(self.expr_csv_file)
        with open(self.marker_yaml_file, 'r') as stream:
            self.marker_dict = yaml.safe_load(stream)

        self.random_seed = 42
        self.state_dict = self.marker_dict["cell_states"]

        self.marker_genes = sorted(list(
            set([l for s in self.state_dict.values() for l in s])))
        self.Y_np = self.expr[self.marker_genes].to_numpy()

        self.N = self.expr.shape[0]
        self.G = len([item for l in self.state_dict.values() for item in l])
        self.C = len(self.marker_dict["cell_states"])

        state_mat = np.zeros(shape=(self.G, self.C))

        for g, gene in enumerate(self.marker_genes):
            for ct, state in enumerate(self.marker_dict["cell_states"].keys()):
                if gene in self.state_dict[state]:
                    state_mat[g, ct] = 1

        self.model = CellStateModel(Y_np=self.Y_np,
                                    state_dict=self.state_dict,
                                    N=self.N, G=self.G, C=self.C,
                                    state_mat=state_mat,
                                    include_beta=True, alpha_random=True,
                                    random_seed=self.random_seed)

    def test_basic_instance_creation(self):
        """ Testing if the instance is created or not
        """
        self.assertIsInstance(self.model, CellStateModel)

    def test_include_beta(self):
        """ Test include_beta variable
        """
        self.assertTrue(self.model.include_beta)

    def test_alpha_random(self):
        """ Test alpha_random variable
        """
        self.assertTrue(self.model.alpha_random)

    def test_optimizer(self):
        """ Test initial optimizer
        """
        self.assertIsNone(self.model.optimizer)

import warnings
from unittest import TestCase
import pandas as pd
import os
import yaml

import torch

from astir.models import CellStateModel
from astir.data_readers import from_csv_yaml
from astir.astir import SCDataset


class TestCellStateModel(TestCase):
    """ Unittest class for CellStateModel class

    This class assumes that all data initializating functions in Astir class
    are working
    """

    def __init__(self, *args, **kwargs):
        super(TestCellStateModel, self).__init__(*args, **kwargs)

        self.expr_csv_file = os.path.join(
            os.path.dirname(__file__), "../test-data/sce.csv"
        )
        self.marker_yaml_file = os.path.join(
            os.path.dirname(__file__), "../test-data/jackson-2020-markers.yml"
        )

        input_expr = pd.read_csv(self.expr_csv_file)
        with open(self.marker_yaml_file, "r") as stream:
            marker_dict = yaml.safe_load(stream)

        state_dict = marker_dict["cell_states"]

        self._dset = SCDataset(
            include_other_column=False,
            expr_input=input_expr,
            marker_dict=state_dict,
            design=None,
        )

        self.model = CellStateModel(
            dset=self._dset, include_beta=True, alpha_random=True, random_seed=42
        )

    def test_basic_instance_creation(self):
        """ Testing if the instance is created or not
        """
        self.assertIsInstance(self.model, CellStateModel)

    def test_include_beta(self):
        """ Test include_beta variable
        """
        self.assertTrue(self.model._include_beta)

    def test_alpha_random(self):
        """ Test alpha_random variable
        """
        self.assertTrue(self.model._alpha_random)

    def test_optimizer(self):
        """ Test initial optimizer
        """
        self.assertIsNone(self.model._optimizer)

    def test_dtype(self):
        self.model.fit(max_epochs=1)
        data = self.model.get_data()
        variables = self.model.get_variables()
        s = list(data.values()) + list(variables.values())
        comp = [ss.dtype == torch.float32 for ss in s]
        assertTrue(all(comp))

    # def test_same_seed_same_result(self):
    #     """ Test whether the loss after one epoch one two different models
    #     with the same random seed have the
    #     """
    #     warnings.filterwarnings("ignore", category=UserWarning)
    #     model1 = CellStateModel(
    #         dset=self._dset,
    #         include_beta=True,
    #         alpha_random=True,
    #         random_seed=42
    #     )
    #
    #
    #     model1_loss = model1.fit(max_epochs=1)
    #     # model2_loss = model2.fit(max_epochs=1)
    #     print("\nmodel1_loss: ", model1_loss)
    #     # print("\nmodel2_loss: ", model2_loss)
    #
    #     # self.assertTrue(np.abs(model1_loss - model2_loss)[0] < 1e-6)
    #
    # def test_same_seed_same_result2(self):
    #     warnings.filterwarnings("ignore", category=UserWarning)
    #     model2 = CellStateModel(
    #         dset=self._dset,
    #         include_beta=True,
    #         alpha_random=True,
    #         random_seed=42
    #     )
    #     model2_loss = model2.fit(max_epochs=1)
    #
    #     print("\nmodel2_loss: ", model2_loss)
    #
    # def test_diff_seed_diff_result(self):
    #     """ Test whether the loss after one epoch one two different models
    #     with the same random seed have the
    #     """
    #     warnings.filterwarnings("ignore", category=UserWarning)
    #     model1 = CellStateModel(
    #         dset=self._dset,
    #         include_beta=True,
    #         alpha_random=True,
    #         random_seed=42
    #     )
    #     model2 = CellStateModel(
    #         dset=self._dset,
    #         include_beta=True,
    #         alpha_random=True,
    #         random_seed=1234
    #     )
    #
    #     model1_loss = model1.fit(max_epochs=1)
    #     model2_loss = model2.fit(max_epochs=1)
    #
    #     self.assertFalse(np.abs(model1_loss - model2_loss)[0] < 1e-6)

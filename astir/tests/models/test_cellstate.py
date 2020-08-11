import warnings
from unittest import TestCase
import pandas as pd
import os
import yaml
import torch

import torch

from astir.models import CellStateModel
from astir.data import from_csv_yaml
from astir.data import SCDataset


class TestCellStateModel(TestCase):
    """ Unittest class for CellStateModel class

    This class assumes that all data initializating functions in Astir class
    are working
    """

    def __init__(self, *args, **kwargs):
        super(TestCellStateModel, self).__init__(*args, **kwargs)

        warnings.filterwarnings("ignore", category=UserWarning)

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
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._dset = SCDataset(
            include_other_column=False,
            expr_input=input_expr,
            marker_dict=state_dict,
            design=None,
            dtype=torch.float32,
            device=self._device,
        )

        self.model = CellStateModel(
            dset=self._dset, random_seed=42, dtype=torch.float32, device=self._device
        )

        self.model.fit(max_epochs=1)
        self.data = self.model.get_data()
        self.variables = self.model.get_variables()

    def test_basic_instance_creation(self):
        """ Testing if the instance is created or not
        """
        self.assertIsInstance(self.model, CellStateModel)

    def test_dtype(self):
        params = list(self.data.values()) + list(self.variables.values())
        comp = [ss.dtype == torch.float32 for ss in params]
        self.assertTrue(all(comp))

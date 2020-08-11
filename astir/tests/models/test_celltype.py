import warnings
from unittest import TestCase
import pandas as pd
import os
import yaml
import torch

import torch

from astir.models import CellTypeModel
from astir.data import from_csv_yaml
from astir.data import SCDataset


class TestCellTypeModel(TestCase):
    """ Unittest class for CellTypeModel class

    This class assumes that all data initializating functions in Astir class
    are working
    """

    def __init__(self, *args, **kwargs):
        super(TestCellTypeModel, self).__init__(*args, **kwargs)

        self.expr_csv_file = os.path.join(
            os.path.dirname(__file__), "../test-data/sce.csv"
        )
        self.marker_yaml_file = os.path.join(
            os.path.dirname(__file__), "../test-data/jackson-2020-markers.yml"
        )

        self.input_expr = pd.read_csv(self.expr_csv_file)
        with open(self.marker_yaml_file, "r") as stream:
            marker_dict = yaml.safe_load(stream)

        self.type_dict = marker_dict["cell_types"]
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._dset = SCDataset(
            include_other_column=True,
            expr_input=self.input_expr,
            marker_dict=self.type_dict,
            design=None,
            device=self._device,
        )

        self.model = CellTypeModel(dset=self._dset, random_seed=42, device=self._device)
        self.model.fit(max_epochs=1)

    def test_basic_instance_creation(self):
        """ Testing if the instance is created or not
        """
        self.assertIsInstance(self.model, CellTypeModel)

    def test_dtype32(self):
        ds = SCDataset(
            include_other_column=True,
            expr_input=self.input_expr,
            marker_dict=self.type_dict,
            design=None,
            dtype=torch.float32,
        )
        m = CellTypeModel(
            dset=ds, random_seed=42, dtype=torch.float32, device=self._device
        )
        m.fit(max_epochs=1)
        data = m.get_data()
        variables = m.get_variables()
        params = list(data.values()) + list(variables.values())
        comp = [ss.dtype == torch.float32 for ss in params]
        self.assertTrue(all(comp))

    def test_dtype64(self):
        ds = SCDataset(
            include_other_column=True,
            expr_input=self.input_expr,
            marker_dict=self.type_dict,
            design=None,
            dtype=torch.float64,
        )
        m = CellTypeModel(
            dset=ds, random_seed=42, dtype=torch.float64, device=self._device
        )
        m.fit(max_epochs=1)
        data = m.get_data()
        variables = m.get_variables()
        params = list(data.values()) + list(variables.values())
        comp = [ss.dtype == torch.float64 for ss in params]
        self.assertTrue(all(comp))

    def test_trainability(self):
        s = [param.requires_grad for param in self.model._recog.parameters()]
        self.assertTrue(all(s))

import warnings
from unittest import TestCase
import pandas as pd
import os
import yaml

from astir.models import CellTypeModel
from astir.data_readers import from_csv_yaml
from astir.astir import SCDataset


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

        input_expr = pd.read_csv(self.expr_csv_file)
        with open(self.marker_yaml_file, "r") as stream:
            marker_dict = yaml.safe_load(stream)

        type_dict = marker_dict["cell_types"]

        self._dset = SCDataset(
            include_other_column=True,
            expr_input=input_expr,
            marker_dict=type_dict,
            design=None,
        )

        self.model = CellTypeModel(
            dset=self._dset, include_beta=True, random_seed=42
        )

    def test_basic_instance_creation(self):
        """ Testing if the instance is created or not
        """
        self.assertIsInstance(self.model, CellTypeModel)

    def test_dtype(self):
        data = self.model.get_data()
        variables = self.model.get_variables()
        s = list(data.values()) + list(variables.values())
        comp = [ss.dtype == torch.float32 for ss in s]
        assertTrue(all(comp))

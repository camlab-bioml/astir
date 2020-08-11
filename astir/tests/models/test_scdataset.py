import unittest
import pandas as pd
import yaml
import torch

from astir.data import SCDataset

import os


class TestSCDataset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSCDataset, self).__init__(*args, **kwargs)

        self.expr_csv_file = os.path.join(
            os.path.dirname(__file__), "../test-data/test_data.csv"
        )

        self.marker_yaml_file = os.path.join(
            os.path.dirname(__file__), "../test-data/jackson-2020-markers.yml"
        )

        self.design_file = os.path.join(
            os.path.dirname(__file__), "../test-data/design.csv"
        )

        # Initializing expected values for unittesting
        # self._param_init_expr_pd()
        self.input_expr = pd.read_csv(self.expr_csv_file, index_col=0)
        with open(self.marker_yaml_file, "r") as stream:
            self.marker_dict = yaml.safe_load(stream)

        self.state_markers = self.marker_dict["cell_states"]

        self.marker_genes = list(
            set([l for s in self.state_markers.values() for l in s])
        )

        self.expr = self.input_expr[self.marker_genes]

        self.design = pd.read_csv(self.design_file, index_col=0)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initializing the actual model
        self.ds = SCDataset(
            include_other_column=False,
            expr_input=self.input_expr,
            marker_dict=self.state_markers,
            design=None,
            dtype=torch.float32,
            device=self._device,
        )

    def _expr_input_tuple(self):
        pass

    def test_basic_instance_creation(self):

        self.assertIsInstance(self.ds, SCDataset)

    def test_marker_genes(self):
        """ Testing if _m_proteins field is declared correctly
        Also tests get_features() and get_n_features() methods
        """
        expected_gene_count = len(self.marker_genes)
        actual_gene_count = self.ds.get_n_features()

        expected_gene_names = sorted(self.marker_genes)
        actual_gene_names = sorted(self.ds.get_features())

        self.assertEqual(expected_gene_count, actual_gene_count)

        self.assertEqual(expected_gene_names, actual_gene_names)

    def test_len_constant_N(self):
        self.assertEqual(self.expr.shape[0], len(self.ds))

    def test_get_classes(self):
        """ Testing if _classes field is declared correctly
        Also tests get_classes() and get_n_classes() methods
        """
        expected_class_count = len(self.state_markers.keys())
        actual_class_count = self.ds.get_n_classes()

        expected_classes = sorted(self.state_markers.keys())
        actual_classes = sorted(self.ds.get_classes())

        self.assertEqual(expected_class_count, actual_class_count)

        self.assertEqual(expected_classes, actual_classes)

    def test_marker_mat_not_include_other(self):
        """
        Also tests constant G and C
        """
        expected_G = len(self.marker_genes)
        # actual_G = self.ds.get_protein_amount()

        expected_C = len(self.state_markers)
        # actual_C = self.ds.get_class_amount()

        expected_marker_mat = torch.zeros((expected_G, expected_C)).to(self._device)
        actual_marker_mat = self.ds.get_marker_mat()

        for g, protein in enumerate(sorted(self.marker_genes)):
            for c, state in enumerate(self.state_markers):
                if protein in self.state_markers[state]:
                    expected_marker_mat[g, c] = 1.0

        self.assertTrue(
            torch.all(torch.eq(expected_marker_mat, actual_marker_mat)).item()
        )

    def test_cell_names(self):

        expected_cell_names = sorted(self.expr.index)
        actual_cell_names = sorted(self.ds.get_cell_names())

        self.assertTrue(expected_cell_names, actual_cell_names)

    # To implement: significant
    def test_marker_mat_include_other(self):
        self.type_markers = self.marker_dict["cell_types"]
        self.marker_genes = list(
            set([l for s in self.type_markers.values() for l in s])
        )

        self.ds = SCDataset(
            include_other_column=True,
            expr_input=self.input_expr,
            marker_dict=self.type_markers,
            design=None,
            dtype=torch.float32,
            device=self._device,
        )

        G = self.ds.get_n_features()
        C = self.ds.get_n_classes()

        expected_marker_mat = torch.zeros((G, C + 1)).to(self._device)
        actual_marker_mat = self.ds.get_marker_mat()
        for g, feature in enumerate(sorted(self.marker_genes)):
            for c, cell_class in enumerate(self.type_markers):
                if feature in self.type_markers[cell_class]:
                    expected_marker_mat[g, c] = 1.0

        self.assertTrue(
            torch.all(
                torch.eq(expected_marker_mat, actual_marker_mat).to(self._device)
            ).item()
        )

    def test_fix_design_none(self):

        expected_design = torch.ones((len(self.ds), 1)).to(
            device=self._device, dtype=torch.float64
        )
        actual_design = self.ds.get_design()

        self.assertTrue(
            torch.all(torch.eq(expected_design, actual_design).to(self._device)).item()
        )

    def test_fix_design_not_none(self):
        self.design = self.design.to_numpy()

        self.ds = SCDataset(
            include_other_column=False,
            expr_input=self.input_expr,
            marker_dict=self.state_markers,
            design=self.design,
            dtype=torch.float64,
            device=self._device,
        )

        expected_design = torch.from_numpy(self.design).to(
            device=self._device, dtype=torch.float64
        )
        actual_design = self.ds.get_design()

        self.assertTrue(
            torch.all(torch.eq(expected_design, actual_design).to(self._device)).item()
        )

    def test_dtype(self):
        comp = []
        comp.append(self.ds.get_exprs().dtype == torch.float32)
        comp.append(self.ds.get_design().dtype == torch.float32)
        comp.append(self.ds.get_mu().dtype == torch.float32)
        comp.append(self.ds.get_sigma().dtype == torch.float32)
        self.assertTrue(all(comp))


if __name__ == "__main__":
    unittest.main()

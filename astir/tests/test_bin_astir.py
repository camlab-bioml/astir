import unittest
import rootpath
import pandas as pd
import yaml
import os
import sys
import subprocess
import torch
import warnings

from astir.astir import Astir

from astir.data_readers import from_csv_yaml


class TestBinAstir(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBinAstir, self).__init__(*args, **kwargs)
        self.exec_path = os.path.join(
            os.path.dirname(__file__), "astir_bash.py"
        )
        print(self.exec_path)
        self.expr_csv_file = os.path.join(
            os.path.dirname(__file__), "test-data/test_data.csv"
        )
        self.marker_yaml_file = os.path.join(
            os.path.dirname(__file__), "test-data/jackson-2020-markers.yml"
        )
        self.output_file = os.path.join(
            os.path.dirname(__file__), "output"
        )
        # print(self.output_file)
        # print(rootpath.detect())
        #
        # print(os.path.isdir(self.expr_csv_file))
        # print(os.path.isdir(self.output_file))
        # print(os.path.isdir(self.marker_yaml_file))

        self.expr = pd.read_csv(self.expr_csv_file, index_col=0)
        with open(self.marker_yaml_file, "r") as stream:
            self.marker_dict = yaml.safe_load(stream)

        # print("####rootpath", rootpath.detect())
        # module_path = os.path.abspath(__file__)
        # print("####0", module_path)
        # module_path = os.path.abspath(os.path.join('.'))
        # print("####1", module_path)
        # if module_path not in sys.path:
        #     sys.path.append(module_path)
        # print(sys.path)
        #
        # module_path = os.path.abspath(os.path.join('..'))
        # print("####2", module_path)
        # if module_path not in sys.path:
        #     sys.path.append(module_path)
        # print(sys.path)


    # def test_basic(self):
    #     write_content = "This is a file for testing!!!!!!!!!!!!!"
    #     filename = "test_argparse.txt"
    #
    #     bash_command = "python content.py {}".format(filename)
    #     process = subprocess.check_output(bash_command.split())
    #     print(str(process))
    #     # output, error = process.communicate()
    #     # self.assertIsNone(error)
    #     #
    #     with open(filename, "r") as fr:
    #         contents = fr.read()
    #         print(fr.read())
    #
    #     self.assertEqual(contents, write_content)

    def test_basic_command(self):
        warnings.filterwarnings("ignore", category=UserWarning)

        bash_command = "python -W ignore {} {} {} {} {}".format(
            self.exec_path, "state", self.expr_csv_file,
            self.marker_yaml_file, self.output_file
        )
        # process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        process = subprocess.Popen(bash_command.split())
        output, error = process.communicate()
        self.assertIsNone(error)

        # a = from_csv_yaml(self.expr_csv_file, self.marker_yaml_file,
        #                   design_csv=None, random_seed=42)
        #
        # a.fit_state()
        # a.state_to_csv(self.output_file)

        read_output = pd.read_csv(self.output_file, index_col=0)
        self.assertEqual(len(read_output), len(self.expr))

        states = self.marker_dict["cell_states"].keys()
        self.assertEqual(len(read_output.columns), len(states))

    # def test_command_all_flags(self):
    #     warnings.filterwarnings("ignore", category=UserWarning)
    #     design, max_epochs, lr, batch_size, random_seed, n_init, \
    #     n_init_epochs, dtype,\
    #     delta_loss, delta_loss_batch = None, 2, 1e-1, 128, 1234, 1, 1, \
    #                                    torch.float64, 1e-3, 10
    #     bash_command = "python -W ignore {} {} {} {} {}".format(
    #         self.exec_path, "state", self.expr_csv_file, self.marker_yaml_file,
    #         self.output_file
    #     )
    #     # bash_command += " --design {}".format(design)
    #     # bash_command += " --max_epochs {}".format(max_epochs)
    #     # bash_command += " --learning_rate {}".format(lr)
    #     # bash_command += " --batch_size {}".format(batch_size)
    #     # bash_command += " --random_seed {}".format(random_seed)
    #     # bash_command += " --n_init {}".format(n_init)
    #     # bash_command += " --n_init_epochs {}".format(n_init_epochs)
    #     # bash_command += " --dtype {}".format(dtype)
    #     # bash_command += " --delta_loss {}".format(delta_loss)
    #     # bash_command += " --delta_loss_batch {}".format(delta_loss_batch)
    #     # process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    #     # output, error = process.communicate()
    #     os.system(bash_command)
    #     # self.assertIsNone(error)
    #
    #     # Create Astir object to compare
    #     ast = Astir(
    #         input_expr=self.expr,
    #         marker_dict=self.marker_dict,
    #         design=design,
    #         random_seed=random_seed,
    #         dtype=dtype
    #     )
    #
    #     ast.fit_state(
    #         max_epochs=max_epochs,
    #         learning_rate=lr,
    #         batch_size=batch_size,
    #         delta_loss=delta_loss,
    #         n_init=n_init,
    #         n_init_epochs=n_init_epochs,
    #         delta_loss_batch=delta_loss_batch
    #     )
    #
    #     expected_assign = ast.get_cellstates()
    #     actual_assign = pd.read_csv(self.output_file, index_col=0)
    #     self.assertEqual(len(expected_assign), len(actual_assign))
    #     self.assertTrue((expected_assign.columns ==
    #                      actual_assign.columns).all())
    #
    #     self.assertTrue((abs(actual_assign.to_numpy()
    #                          - expected_assign.to_numpy()) < 0.01).all())

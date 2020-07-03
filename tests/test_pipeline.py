import os
from pathlib import Path
import numpy as np
import time
import pytest


class TestCmdLine:
    @pytest.mark.parametrize("model", ["lstm", "ealstm"])
    def test_model(self, capsys, model):
        if model == "lstm":
            config_file = "tests/test_config/test_config.yml"
            expt_name = "test"
        elif model == "ealstm":
            config_file = "tests/test_config/test_config_ealstm.yml"
            expt_name = "test_ealstm"

        # ------ run train command ------- #
        run_cmd = f"ipython main.py train -- --config_file {config_file}"
        with capsys.disabled():
            print(f"Running training for model {model}:\n\t{run_cmd}")
            os.system(run_cmd)

        time_now = time.gmtime()

        run_dir = max(
            [
                p
                for p in Path("runs").glob(
                    f"{expt_name}_{time_now.tm_mday:02}{time_now.tm_mon:02}*"
                )
            ]
        )
        assert run_dir.exists()

        # check models saved / checkpointed
        assert all(
            np.isin(
                ["model_epoch001.pt", "model_epoch002.pt"],
                [d.name for d in run_dir.iterdir()],
            )
        )
        # check features created
        assert all(
            np.isin(
                ["train_data.h5", "train_data_scaler.p"],
                [d.name for d in (run_dir / "train_data").iterdir()],
            )
        )

        # ------- run evaluation command ------- #
        eval_cmd = f"ipython main.py evaluate -- --run_dir {run_dir}"
        with capsys.disabled():
            print(f"Running evaluation:\n\t{eval_cmd}")
            os.system(eval_cmd)
        assert (
            run_dir / "test/model_epoch002"
        ).exists(), "Expect 2nd Epoch to have produced test directory"
        assert (
            run_dir / "test/model_epoch002/test_results.p"
        ).exists(), "Expect 2nd Epoch to have produced test results"

        # ------- clear the data ------- #
        # rm test directory
        os.system(f"rm -rf {run_dir}")

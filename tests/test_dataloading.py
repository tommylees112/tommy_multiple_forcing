import os
from pathlib import Path
import numpy as np
import time


class TestCmdLine:
    def test_cmd_line(self, capsys):
        # run Train command
        run_cmd = "ipython main.py train -- --config_file tests/test_config/test_config.yml"
        print(f"Running training:\n\t{run_cmd}")
        os.system(run_cmd)

        captured = capsys.readouterr()
        time_now = time.gmtime()

        run_dir = max([p for p in Path("runs").glob(f"test_{time_now.tm_mday:02}{time_now.tm_mon:02}*")])
        assert run_dir.exists()

        # check models saved / checkpointed
        assert all(np.isin(["model_epoch001.pt", "model_epoch002.pt"], [d.name for d in run_dir.iterdir()]))
        # check features created
        assert all(np.isin(["train_data.h5", "train_data_scaler.p"], [d.name for d in (run_dir / "train_data").iterdir()]))

        # run evaluation command
        eval_cmd = f"ipython main.py evaluate -- --run_dir {run_dir}"
        print(f"Running evaluation:\n\t{eval_cmd}")
        os.system(eval_cmd)
        assert (run_dir / "test/model_epoch002").exists(), "Expect 2nd Epoch to have produced test directory"
        assert (run_dir / "test/model_epoch002/test_results.p").exists(), "Expect 2nd Epoch to have produced test results"

        # rm test directory
        # os.system(f"rm -rf {run_dir}")
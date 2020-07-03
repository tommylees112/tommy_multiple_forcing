from pathlib import Path

from codebase.modelzoo.cudalstm import CudaLSTM
from codebase.config import read_config
from codebase.training.regressiontrainer import RegressionTrainer
from codebase.training.basetrainer import BaseTrainer


class Test:
    def test_base(self):
        config_file = Path("tests") / "test_config/test_config.yml"
        cfg = read_config(config_file)
        trainer = BaseTrainer(cfg)

    def test_cudalstm(self):
        config_file = Path("tests") / "test_config/test_config.yml"
        cfg = read_config(config_file)
        trainer = RegressionTrainer(cfg)

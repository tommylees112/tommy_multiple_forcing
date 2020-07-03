from pathlib import Path

from codebase.modelzoo.basemodel import BaseModel
from codebase.modelzoo.cudalstm import CudaLSTM
from codebase.modelzoo.ealstm import EALSTM
from codebase.config import read_config
from codebase.training.regressiontrainer import RegressionTrainer


class Test:
    def test_base(self):
        config_file = Path("tests") / "test_config/test_config.yml"
        cfg = read_config(config_file)
        model = BaseModel(cfg)

    def test_cudalstm(self):
        config_file = Path("tests") / "test_config/test_config.yml"
        cfg = read_config(config_file)
        model = CudaLSTM(cfg)

        # trainer object
        trainer = RegressionTrainer(cfg)
        trainer.initialize_training()
        trainer.train_and_validate()

    def test_ealstm(self):
        config_file = Path("tests") / "test_config/test_config_ealstm.yml"
        cfg = read_config(config_file)
        # check that initialisation works
        model = EALSTM(cfg)

        # trainer object
        trainer = RegressionTrainer(cfg)
        trainer.initialize_training()
        trainer.train_and_validate()

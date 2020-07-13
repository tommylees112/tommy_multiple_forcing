from pathlib import Path
import time

from codebase.modelzoo.basemodel import BaseModel
from codebase.modelzoo.cudalstm import CudaLSTM
from codebase.modelzoo.ealstm import EALSTM
from codebase.config import read_config
from codebase.training.regressiontrainer import RegressionTrainer
from codebase.evaluation.tester import Tester
from analysis.datautils import create_results_csv


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

        #  tester object
        time_now = time.gmtime()
        expt_name = cfg["experiment_name"]
        run_dir = max(
            [
                p
                for p in Path("runs").glob(
                    f"{expt_name}_{time_now.tm_mday:02}{time_now.tm_mon:02}*"
                )
            ]
        )
        assert run_dir.exists()

        tester = Tester(cfg, run_dir)

        epoch = int(
            sorted(list(run_dir.glob("*.pt")))[-1]
            .name.split("epoch")[-1]
            .replace(".pt", "")
        )
        tester.evaluate(epoch=epoch, save_results=True, metrics=cfg.get("metrics", []))

        assert (run_dir / "test/model_epoch002/test_results.p").exists()

        # analysis object (-> csv)
        create_results_csv(run_dir=run_dir, epoch=epoch)
        assert (run_dir / f"results_{run_dir.name}_E002.csv").exists


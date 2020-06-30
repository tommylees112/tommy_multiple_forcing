import sys


class TestAll:

    def run_cmdline_arguments(self, capsys):
        f"ipython --pdb main.py train -- --config_file tests/test_config/test_config.yml"
        captured = capsys.readouterr()
        "### Folder structure*\n"
        f"ipython --pdb main.py evaluate -- --run_dir runs/{}"
        "ipython --pdb analysis/datautils.py -- --run_dir /Users/tommylees/github/tommy_multiple_forcing/runs/test_3006_1514/"
# Use the Gcloud Deep Learning VM Image
# https://cloud.google.com/deep-learning-vm?_ga=2.39022439.-552125985.1585819808

# get the runoff code
git clone https://github.com/tommylees112/tommy_multiple_forcing.git

# install packages
pip install ruamel.yaml
pip install xarray
pip install tensorboard

# move data
cp -r /home/jovyan/CAMELS_GB_DATASET .

# run the runoff code
# cd tommy_multiple_forcing
# python main.py train --config_file configs/gcloud_config.yml
# mv tommy_camels_gb_??? tommy_camels_gb_2606_1444
# python main.py evaluate --run_dir runs/tommy_camels_gb_2906_0958/
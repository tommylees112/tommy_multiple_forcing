# ---- -----
ipython main.py train -- --config_dir configs/aws_lstm.yml
ipython main.py train -- --config_dir configs/aws_ealstm.yml

dir='/home/tommy/tommy_multiple_forcing/runs/'

# ---- run the evaluation -----
unset -v lstm_run
for file in "$dir"/tomm*; do
  [[ $file -nt $lstm_run ]] && lstm_run=$file
done

unset -v ealstm_run
for file in "$dir"/lstm*; do
  [[ $file -nt $ealstm_run ]] && ealstm_run=$file
done

# ---- run the analysis -----
ipython analysis/datautils.py -- --run_dir runs/tommy_camels_gb_0607_1221/

dir='/home/tommy/tommy_multiple_forcing/runs'

# ---- run the LSTM -----
ipython main.py train -- --config_file configs/aws_lstm.yml

unset -v lstm_run
for file in "$dir"/lstm*; do
  [[ $file -nt $lstm_run ]] && lstm_run=$file
done

ipython analysis/analyse_all_epochs.py -- --run_dir $lstm_run

# ---- run the EALSTM -----
ipython main.py train -- --config_file configs/aws_ealstm.yml

unset -v ealstm_run
for file in "$dir"/ealstm*; do
  [[ $file -nt $ealstm_run ]] && ealstm_run=$file
done

ipython analysis/analyse_all_epochs.py -- --run_dir $ealstm_run

# # ---- run the evaluation -----
# ipython main.py evaluate -- --run_dir $lstm_run
# ipython main.py evaluate -- --run_dir $ealstm_run

# # ---- run the analysis -----
# ipython analysis/datautils.py -- --run_dir $lstm_run
# ipython analysis/datautils.py -- --run_dir $ealstm_run

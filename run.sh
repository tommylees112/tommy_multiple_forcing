# ---- run the training -----
ipython main.py train -- --config_file configs/aws_lstm.yml
ipython main.py train -- --config_file configs/aws_ealstm.yml

dir='/home/tommy/tommy_multiple_forcing/runs'

# ---- get the runs -----
unset -v lstm_run
for file in "$dir"/lstm*; do
  [[ $file -nt $lstm_run ]] && lstm_run=$file
done

unset -v ealstm_run
for file in "$dir"/lstm*; do
  [[ $file -nt $ealstm_run ]] && ealstm_run=$file
done

# ---- run the evaluation -----
ipython main.py evaluate -- --run_dir $lstm_run
ipython main.py evaluate -- --run_dir $ealstm_run

# ---- run the analysis -----
ipython analysis/datautils.py -- --run_dir $lstm_run
ipython analysis/datautils.py -- --run_dir $ealstm_run

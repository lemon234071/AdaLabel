DATA_DIR="data_daily"
#DATA_DIR="data_ost"
DATASET="bert"
EMB=512
STEPS=1000000
BS=64
ACCUM=2
SAVESTEPS=1000


python3 train.py -adalab -bidecoder \
  -world_size 1 -gpu_ranks 0 \
  -log_file ./log_dir/"$DATASET"_transformer_adalab.log -data "$DATA_DIR"/"$DATASET" \
  -save_model checkpoint/"$DATASET"_trainsformer_adalab \
  -train_steps "$STEPS" -save_checkpoint_steps "$SAVESTEPS" -valid_steps "$SAVESTEPS"  -report_every 100 \
  -max_generator_batches 0 -dropout 0.1 -max_grad_norm 1 \
  -encoder_type transformer -decoder_type transformer -position_encoding \
  -param_init 0 -param_init_glorot -transformer_ff 512 -heads 8 \
  -batch_size "$BS" -accum_count "$ACCUM" -layers 6 -rnn_size "$EMB" -word_vec_size "$EMB" \
  -optim adam -learning_rate 1e-4 -start_decay_steps 100000000 -early_stopping 10
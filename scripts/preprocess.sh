DATA_DIR="data_daily"
#DATA_DIR="data_ost"
DATASET="bert"
VOCAB="vocab.txt"


python3 preprocess.py -train_src "$DATA_DIR"/src-train.txt -train_tgt "$DATA_DIR"/tgt-train.txt \
  -valid_src "$DATA_DIR"/src-valid.txt -valid_tgt "$DATA_DIR"/tgt-valid.txt \
  -save_data "$DATA_DIR"/"$DATASET" -share_vocab \
  -src_vocab_size 300000 -tgt_vocab_size 300000 \
  -src_vocab "$DATA_DIR"/"$VOCAB" -tgt_vocab "$DATA_DIR"/"$VOCAB" \
  -src_seq_length 512 -tgt_seq_length 512 \
  -tokenizer bert
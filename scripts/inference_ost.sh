# DATA_DIR="data_daily"
DATA_DIR="data_ost"
DATASET="bert"

python3 translate.py -gpu "$1" -model "$2" \
    -output result/"$DATASET"_adalab_"$DATA_DIR".txt -beam 1 -batch_size 128 \
    -src "$DATA_DIR"/src-test.txt -max_length 30 -tokenizer bert

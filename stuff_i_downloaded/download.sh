SPLIT="train"
OUT_DIR="/home/bjafek/Nuro/benj_prac/fathomnet/data/$SPLIT"
DATASET="/home/bjafek/Nuro/benj_prac/fathomnet/data/dataset_$SPLIT.json"
python download.py \
    $DATASET \
    $OUT_DIR

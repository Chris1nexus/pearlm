DATASET=$1
NPATHS=$2
HOPS=$3
NPROC=4

TASK='end-to-end'

LOGDIR=$(echo "dataset_${DATASET}__hops_${HOPS}__npaths_${NPATHS}")

pip install ../.. && python3 main.py --dataset $DATASET --max_n_paths $NPATHS --max_hop $HOPS --collaborative TRUE --nproc $NPROC
find ./statistics/$DATASET/$LOGDIR -name '*.txt' -exec cat {} \; >> concatenated_rw_file.txt
python3 prune_dataset.py --filepath concatenated_rw_file.txt
mv concatenated_rw_file.txt ../data/$DATASET/paths_random_walk/paths_${TASK}_${NPATHS}_${HOPS}.txt
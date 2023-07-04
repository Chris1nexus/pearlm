DATASET=$1
NPATHS=$2
HOPS=$3
MODEL=$4
NPROC=8
    
pip install ../ && python3 models/lm/from_scratch_main.py --dataset $DATASET \
                    --sample_size_finetune $NPATHS \
                    --sample_size_hop $HOPS \
                    --model $MODEL \
                    --nproc $NPROC \
                    --n_hop $HOPS


DEVICE_NUM=$1
NPROC=6
for MODEL in distilgpt2@plm-rec ;
	do
	for HOPS in 5 3 ;
	  do
	  	for NPATHS in  1000 500 250;
				do
					for DATASET in  ml1m lfm1m ; 
						do			
									echo 'Running: model' $MODEL 'dataset-' $DATASET ' npaths-' $NPATHS ' hops-' $HOPS
									pip install . && export CUDA_VISIBLE_DEVICES=$DEVICE_NUM && python3 pathlm/models/lm/from_scratch_main.py --dataset $DATASET \
									                    --sample_size_finetune $NPATHS \
									                    --sample_size_hop $HOPS \
									                    --model $MODEL \
									                    --nproc $NPROC \
									                    --n_hop $HOPS \
									                    --emb_filename 'transe_embed.pkl' \
									                    --emb_size 100 \
											    		--batch_size  2048 \
									                    --eval_device cuda:$DEVICE_NUM \
									                    --infer_batch_size 128 \
														--logit_processor_type 'plm'
									echo 'Completed run: model' $MODEL 'dataset-' $DATASET ' npaths-' $NPATHS ' hops-' $HOPS
									echo
						done
					done
			done    	
	done

DEVICE_NUM=$1
NPROC=6
for MODEL in gpt2-large gpt2 distilgpt2  ;
	do
	for HOPS in 5 3 ;
		do
	  	for NPATHS in  250 500 1000 10000 3000 ;
				do
					for DATASET in ml1m lfm1m;
						do			
									echo 'Running: model' $MODEL 'dataset-' $DATASET ' npaths-' $NPATHS ' hops-' $HOPS
									export CUDA_VISIBLE_DEVICES=$DEVICE_NUM && python3 -m pathlm.models.lm.from_scratch_main --dataset $DATASET \
									                    --sample_size $NPATHS \
									                    --model $MODEL \
									                    --nproc $NPROC \
									                    --n_hop $HOPS \
											    		--batch_size  128 \
									                    --infer_batch_size 64 \
									                    --eval_device cuda:0 \
														--logit_processor_type 'gcd'	\
														--num_training_steps 60000 \
														--wandb 								                    
									echo 'Completed run: model' $MODEL 'dataset-' $DATASET ' npaths-' $NPATHS ' hops-' $HOPS
									echo
						done
				done
		done    	
	done

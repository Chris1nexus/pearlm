DEVICE_NUM=$1
NPROC=6
for MODEL in gpt2  ;
	do
	for HOPS in 3  ;
		do
	  	for NPATHS in  250  ;
				do
					for DATASET in lfm1m;
						do			
									echo 'Running: model' $MODEL 'dataset-' $DATASET ' npaths-' $NPATHS ' hops-' $HOPS
									export CUDA_VISIBLE_DEVICES=$DEVICE_NUM && python3 -m pathlm.models.lm.from_scratch_main --dataset $DATASET \
									                    --sample_size $NPATHS \
									                    --model $MODEL \
									                    --nproc $NPROC \
									                    --n_hop $HOPS \
											    		--batch_size  256 \
									                    --infer_batch_size 128 \
									                    --eval_device cuda:0 \
														--logit_processor_type 'gcd'	\
														--num_training_steps 100000 \
														--validation_interval 3000 \
														--wandb 	
														#--load_data True \
																				                    
									echo 'Completed run: model' $MODEL 'dataset-' $DATASET ' npaths-' $NPATHS ' hops-' $HOPS
									echo
						done
				done
		done    	
	done

DEVICE_NUM=$1
NPROC=8

for HOPS in 3 5 ;
	do
	for MODEL in gpt2  distilgpt2 ;
		do
	  	for NPATHS in 250 500 1000 2000 3000 ;
				do
					for DATASET in  lfm1m  ml1m; 
						do
						      echo 'Tokenizing dataset-' $DATASET ' npaths-' $NPATHS ' hops-' $HOPS
						      python3 -m pathlm.models.lm.tokenize_dataset --dataset $DATASET --sample_size $NPATHS --nproc $NPROC --n_hop $HOPS
									echo 'Running: model' $MODEL 'dataset-' $DATASET ' npaths-' $NPATHS ' hops-' $HOPS
									export CUDA_VISIBLE_DEVICES=$DEVICE_NUM && python3 -m pathlm.models.lm.pearlm_main --dataset $DATASET \
									                    --sample_size $NPATHS \
									                    --model $MODEL \
									                    --nproc $NPROC \
									                    --n_hop $HOPS \
											    		--batch_size  256 \
									                    --infer_batch_size 128 \
									                    --eval_device cuda:0 \
														--logit_processor_type 'gcd'	\
														--num_epochs 20 \
														--validation_interval 6000 \
														--wandb 	
														#--load_data True \
														#--num_training_steps 60000 \		                    
									echo 'Completed run: model' $MODEL 'dataset-' $DATASET ' npaths-' $NPATHS ' hops-' $HOPS
									echo
						done
				done
		done    	
	done

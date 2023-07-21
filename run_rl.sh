DEVICE_NUM=$1
NPROC=6


LR=5e-4
PPO_BATCH=400
for MODEL in gpt2 ;
	do
	for HOPS in 5 10 15;
	  do
	  	for NPATHS in  2000;
				do
					for DATASET in  ml1m ; 
						do			
							for NEG_RW in -1.0 0.0 ;
								do
									for PPO_EPOCHS in 5 20 50 ;
										do
											for PPO_INTERVAL in 20 100 500 ;
												do
													for PPO_ENT_COEF in 0.0 1.0 ;
														do 
															echo 'Running: model' $MODEL 'dataset-' $DATASET ' npaths-' $NPATHS ' hops-' $HOPS
															pip install . && export CUDA_VISIBLE_DEVICES=$DEVICE_NUM && python3 pathlm/models/lm_rl/train_rl.py --dataset $DATASET \
															                    --sample_size $NPATHS \
															                    --n_hop $HOPS \
															                    --model $MODEL \
															                    --nproc $NPROC \
															                    --n_hop $HOPS \
															                    --eval_device cuda:0 \
																				--wandb \
																				--pos_item_rw  1.0 \
																				--neg_item_rw  $NEG_RW \
																				--minibatch_size $PPO_BATCH \
																				--n_epochs   $PPO_EPOCHS  \
																				--update_interval $PPO_INTERVAL   \
																				--ent_coef $PPO_ENT_COEF
																				#--lr     $LR \
															echo 'Completed run: model' $MODEL 'dataset-' $DATASET ' npaths-' $NPATHS ' hops-' $HOPS
															echo
													done
											done
									done
							done
						done
					done
			done    	
	done

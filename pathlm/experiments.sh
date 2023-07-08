for MODEL in distilgpt2 gpt2-medium gpt2-large ;
	do
	for HOPS in 3 ;
	  do
	  	for NPATHS in  2000 ;
				do
					for DATASET in lfm1m;
						do			
									echo 'Running: model' $MODEL 'dataset-' $DATASET ' npaths-' $NPATHS ' hops-' $HOPS
									pip install ../ && python3 models/lm/from_scratch_main.py --dataset $DATASET \
									                    --sample_size_finetune $NPATHS \
									                    --sample_size_hop $HOPS \
									                    --model $MODEL \
									                    --nproc $NPROC \
									                    --n_hop $HOPS
									echo 'Completed run: model' $MODEL 'dataset-' $DATASET ' npaths-' $NPATHS ' hops-' $HOPS
									echo
						done
					done
			done    	
	done

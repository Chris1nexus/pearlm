for hops in 3 ;
  do
  	for npaths in  2000 ;
			do
	  		for dataset in lfm1m;
					do			
						echo 'Creating: dataset-' $dataset ' npaths-' $npaths ' hops-' $hops
						bash create_dataset.sh $dataset $npaths $hops
						echo 'Completed'
						echo 
					done
			done    	
  done

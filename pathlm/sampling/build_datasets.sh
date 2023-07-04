for hops in 7 ;
  do
  	for npaths in  250 500 1000 ;
			do
	  		for dataset in ml1m lfm1m;
					do			
						echo 'Creating: dataset-' $dataset ' npaths-' $npaths ' hops-' $hops
						bash create_dataset.sh $dataset $npaths $hops
						echo 'Completed'
						echo 
					done
			done    	
  done
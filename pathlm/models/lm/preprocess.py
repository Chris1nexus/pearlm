import multiprocessing as mp
import os
import pandas as pd


class Mapper:
	def __init__(self, map_filepath, path_filepath):
		e_map_path = os.path.join(map_filepath, 'preprocessed', 'e_map.txt')
		r_map_path = os.path.join(map_filepath, 'preprocessed', 'r_map.txt') 

		#paths_filepath = os.path.join()

		df = pd.read_csv(e_map_path, sep='\t')
		with open(e_map_path) as f:
			for i,line in enumerate(f):
				if i == 0:
					continue
				data = line.split('\t')
				eid = data[0]
				ename = data[1]

		with open(r_map_path) as f:
			for i,line in enumerate(f):
				if i == 0:
					continue
				data = line.split('\t')
				rid = data[0]
				rname = data[1]

		print(df)


m = Mapper('../../data/ml1m')

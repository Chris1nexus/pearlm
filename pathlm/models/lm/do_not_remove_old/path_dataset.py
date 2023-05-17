from multiprocessing import Pool
from datasets import Dataset
from os import listdir
import os
from os.path import isfile, join
import pandas as pd
import torch

from pathlm.utils import get_eid_to_name_map, get_rid_to_name_map


class PathLMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, base_data_dir:str=''):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.base_data_dir = base_data_dir
        # Get eid2name and rid2name
        self.eid2name = get_eid_to_name_map(self.base_data_dir)
        self.rid2name = get_rid_to_name_map(self.base_data_dir)        

    def __getitem__(self, idx):
        #item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        txt_path = self.convert_numeric_path_to_textual_path(self.dataset[idx]) 
        #print('txtxt',txt_path)
        #item = self.tokenizer(txt_path, truncation=True, padding=True)
        #print('sssss',item)
        #item['labels'] = torch.tensor(self.labels[idx])
        return txt_path

    def __len__(self):
        return len(self.dataset)
    def convert_numeric_path_to_textual_path(self, path):
        path_list = path.split(" ")
        ans = []
        for pos, token in enumerate(path_list):
            if pos == 0:
                ans.append(token)
            elif pos % 2 == 0:
                #ans.append("<start_entity>")
                ans.append(self.eid2name[token])
                #ans.append("<end_entity>")
            else:

                #ans.append("<start_relation>")
                if pos == 1:
                    ans.append('interact')    
                else:
                    ans.append(self.rid2name[token])
                #ans.append("<end_relation>")
        return " ".join(ans)
class PathDataset:


    def __init__(self, dataset_name: str, tokenizer, base_data_dir: str=""):
    

        self.dataset_name = dataset_name
        self.base_data_dir = base_data_dir
        self.data_dir = join(self.base_data_dir) #'concatenated.txt')#"paths_random_walk")
        #print(self.data_dir)
        self.tokenizer = tokenizer
        self.dataset = self.read_csv_as_dataframe('concatenated.txt')
        self.dataset = Dataset.from_pandas(self.dataset)
        #self.read_multiple_csv_to_hf_dataset()

        # Get eid2name and rid2name
        self.eid2name = get_eid_to_name_map(self.base_data_dir)
        self.rid2name = get_rid_to_name_map(self.base_data_dir)

        #CACHE_PATH = 'pathds'
        #os.makedirs(CACHE_PATH, exist_ok=True)
        #####self.dataset = self.dataset.map(lambda x: {"path": self.convert_numeric_path_to_textual_path(x["path"])}, batched=True, num_proc=16, )
        #self.dataset = self.dataset.map(lambda x: {'path': [PathDataset.convert_numeric_path_to_textual_path(path, self.eid2name, self.rid2name) for path in x['path']] },
        #         batched=True, num_proc=8, )

        #load_from_cache_file=True,cache_file_name=CACHE_PATH  )

    #def __getitem__(self, idx):
    #    #item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    #    txt_path = self.convert_numeric_path_to_textual_path(self.dataset['path'][idx]) 
    #    print('txtxt',txt_path)
    #    item = self.tokenizer(txt_path)
    #    print('sssss',item)
    #    #item['labels'] = torch.tensor(self.labels[idx])
    #    return item

    #def __len__(self):
    #    return len(self.dataset)

    def convert_numeric_path_to_textual_path(path, eid2name, rid2name):
        path_list = path.split(" ")
        ans = []
        for pos, token in enumerate(path_list):
            if pos == 0:
                ans.append(token)
            elif pos % 2 == 0:
                #ans.append("<start_entity>")
                ans.append(eid2name[token])
                #ans.append("<end_entity>")
            else:

                #ans.append("<start_relation>")
                if pos == 1:
                    ans.append('interact')    
                else:
                    ans.append(rid2name[token])
                #ans.append("<end_relation>")
        return " ".join(ans)
    # Based on the path struct, for now it is p to p


    def read_csv_as_dataframe(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(join(self.data_dir, filename), header=None, names=["path"], index_col=None)

    def read_multiple_csv_to_hf_dataset(self):
        file_list = [f for f in listdir(self.data_dir) if isfile(join(self.data_dir, f))]

        # set up your pool
        with Pool(processes=8) as pool:  # orc whatever your hardware can support
            df_list = pool.map(self.read_csv_as_dataframe, file_list)

            # reduce the list of dataframes to a single dataframe
            combined_df = pd.concat(df_list, ignore_index=True)

        # Convert to HuggingFace Dataset
        self.dataset = Dataset.from_pandas(combined_df)

    def show_random_examples(self):
        print(self.dataset["path"][:10])
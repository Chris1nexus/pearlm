from multiprocessing import Pool
from datasets import Dataset
from os import listdir
from os.path import isfile, join
import pandas as pd

from pathlm.utils import get_eid_to_name_map, get_rid_to_name_map

class PathDataset:
    def __init__(self, dataset_name: str, base_data_dir: str=""):
        self.dataset_name = dataset_name
        self.base_data_dir = base_data_dir
        self.data_dir = join(self.base_data_dir, "paths_random_walk")
        self.read_multiple_csv_to_hf_dataset()

        # Get eid2name and rid2name
        self.eid2name = get_eid_to_name_map(self.base_data_dir)
        self.rid2name = get_rid_to_name_map(self.base_data_dir)

        self.dataset = self.dataset.map(lambda x: {"path": self.convert_numeric_path_to_textual_path(x["path"])})

    # Based on the path struct, for now it is p to p
    def convert_numeric_path_to_textual_path(self, path: str):
        path_list = path.split(" ")
        ans = []
        for pos, token in enumerate(path_list):
            # Handle user
            if pos == 0:
                ans.append(token)
            # Handle recommendation
            elif pos == len(path_list) - 1:
                #ans.append("<recommendation>")
                ans.append(self.eid2name[token])
            # Handle entity
            elif pos % 2 == 0:
                ans.append(self.eid2name[token])
            # Handle relation
            else:
                ans.append(self.rid2name[token])
        return " ".join(ans)

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
from multiprocessing import Pool
from datasets import Dataset
from os import listdir
from os.path import isfile, join
import pandas as pd

from pathlm.utils import get_eid_to_name_map, get_rid_to_name_map

class PathDataset:
    def __init__(self, dataset_name: str, data_dir: str=""):
        self.dataset_name = dataset_name
        self.base_data_dir = f'data/{dataset_name}'
        self.data_dir = join(self.base_data_dir, "paths_random_walk")
        self.read_multiple_csv_to_hf_dataset()

        # Get eid2name and rid2name
        eid2name = get_eid_to_name_map(self.base_data_dir)
        rid2name = get_rid_to_name_map(self.base_data_dir)

        # Based on the path struct, for now it is p to p
        def convert_numeric_path_to_textual_path(path: str):
            path_list = path.split(" ")
            ans = []
            for pos, token in enumerate(path_list):
                if pos % 2 == 0:
                    ans.append("<start_entity>")
                    ans.append(eid2name[token])
                    ans.append("<end_entity>")
                else:
                    ans.append("<start_relation>")
                    ans.append(rid2name[token])
                    ans.append("<end_relation>")
            return " ".join(ans)

        self.dataset = self.dataset.map(lambda x: {"path": convert_numeric_path_to_textual_path(x["path"])})

    def read_csv_as_dataframe(self, filename: str) -> pd.DataFrame:
        return pd.read_csv(join(self.data_dir, filename), header=None, names=["path"], index_col=None)

    """
    Reads the sharded csv files to pandas dataframe and merge them into a single one which is then converted to HuggingFace Dataset
    """
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


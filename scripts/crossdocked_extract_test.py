import os
import sys
import shutil
from pathlib import Path

import torch
from tqdm import tqdm

cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    data_dir, split_file = sys.argv[1], sys.argv[2]
    output_dir = f"./data/crossdocked_test"
    Path(output_dir).mkdir(exist_ok=True)
    test_split = torch.load(f"{split_file}")["test"]

    for pocket_fn, ligand_fn in tqdm(test_split):
        pdb_id = os.path.basename(pocket_fn).split("_")[0]
        lig_id = os.path.basename(ligand_fn).split("_")[4]
        shutil.copy(f"{data_dir}/{pocket_fn}", f"{output_dir}/{pdb_id}_{lig_id}.pdb")
        shutil.copy(f"{data_dir}/{ligand_fn}", f"{output_dir}/{pdb_id}_{lig_id}.sdf")

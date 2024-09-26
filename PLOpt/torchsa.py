import os

import torch

from PLOpt.scoring_model import PaiNNScoringModel
from PLOpt.ligand import Ligand

cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchSAScore(PaiNNScoringModel):
    """
    Model for approximating SA score.
    """

    def __init__(self):
        model_ckpt = os.path.join(cwd, "checkpoints", "torchsa.ckpt")
        super().__init__(model_ckpt)

    def score(self, ligands: list[Ligand]) -> torch.tensor:
        """
        Score ligands according to torchSA
        Args:
            ligands: ligands to score
        Returns:
            tensor of SA scores for each ligand
        """
        ans, coords = [], []
        for ligand in ligands:
            lig_ans_vec, lig_coords = ligand.atomic_probs.clone(), ligand.coords()
            lig_ans_vec = lig_ans_vec ** 4 / torch.sum(lig_ans_vec ** 4, axis=1).unsqueeze(1)
            ans.append(lig_ans_vec.clone())
            coords.append(lig_coords.clone())

        return torch.clamp(super().score(ans, coords), min=1, max=10)

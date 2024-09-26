import torch

from PLOpt.scoring_model import ScoringModel
from PLOpt.ligand import Ligand

class BondedScore(ScoringModel):
    """
    Score for enforcing bond constraints during geometry optimization.
    """

    @staticmethod
    def score(ligands: list[Ligand]):
        bonded_scores = []
        for ligand in ligands:
            conn_matrix, cutoffs = ligand.get_connectivity_and_cutoffs()
            dist = torch.cdist(ligand.coords(add_hydrogens=True)[None, :, :], ligand.coords(add_hydrogens=True)[None, :, :])[0]
            bond_score = torch.sum(torch.clamp(torch.masked_select(dist, conn_matrix) - cutoffs, min=0)).unsqueeze(0)
            bonded_scores.append(bond_score)

        return torch.cat(bonded_scores)
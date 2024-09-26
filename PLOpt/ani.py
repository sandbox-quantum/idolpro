import torch
import torchani

from PLOpt.ligand import Ligand
from PLOpt.protein import Protein
from PLOpt.scoring_model import ScoringModel
from PLOpt.plopt_utils import Batcher, _make_float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PADDING = {
    "species": -1,
    "atomic_numbers": -1,
    "cell": 1e-8,
    "coordinates": 0.0,
    "forces": 0.0,
    "energies": 0.0,
}

def collate_fn_torchani(
    samples: list[torch.tensor], padding: dict = None
) -> list[torch.tensor]:
    """
    Collate function when using torchani model
    """
    if padding is None:
        padding = PADDING
    outs = dict(torchani.utils.stack_with_padding(samples, padding))
    return _make_float32(outs)

class ANIScore(ScoringModel):
    """
    Model for computing binding energy with ANI2x
    Args:
        protein: Protein object defining protein pocket.
    """

    def __init__(self, protein: Protein):
        model = torchani.models.ANI2x(periodic_table_index=True).to(device)
        for param in model.parameters():
            param.requires_grad_(False)
        self.model = model
        self.pocket = protein.get_pocket_ans_and_coords(add_hydrogens=True)

    def score(self, ligands: list[Ligand], intra_only: bool = False) -> torch.tensor:
        """
        For each ligand compute energy of protein/ligand using ANI2x
        Args:
            ligands: list of ligands
        Returns:
            tensor of ligand/protein energies
        """
        ligs_ans, ligs_coords = [], []
        ans, coords = [], []
        pocket_ans, pocket_coords = self.pocket
        for ligand in ligands:
            lig_ans, lig_coords = ligand.ans(add_hydrogens=True).long().to(
                device
            ), ligand.coords(add_hydrogens=True)
            ligs_ans.append(lig_ans)
            ligs_coords.append(lig_coords)
            ans.append(torch.cat((lig_ans, pocket_ans)))
            coords.append(torch.cat((lig_coords, pocket_coords)))
        # pocket and ligand energies
        batch = next(
            iter(
                torch.utils.data.DataLoader(
                    Batcher(ligs_ans, ligs_coords) if intra_only else Batcher(ans, coords),
                    batch_size=len(ligs_ans) if intra_only else len(ans),
                    collate_fn=collate_fn_torchani,
                )
            )
        )
        pocket_lig_energies = self.model(
            (batch["atomic_numbers"], batch["coordinates"])
        ).energies

        return pocket_lig_energies

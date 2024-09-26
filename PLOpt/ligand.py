"""
ligand object
"""

from typing import List
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit.Geometry import Point3D
import torch
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs

device = "cuda" if torch.cuda.is_available() else "cpu"


class Ligand:
    """
    Ligand class holding all the necessary info and convenience methods

    Args:
        rdkit_mol: rdkit mol object
        coordinates: tensor of coordinates
        atomic numbers: tensor of atomic numbers
        atomic_probs: tensor of atomic prob distro
        scores: which scores we want to track
        protonated: whether the ligand is protonated

    """

    def __init__(
        self,
        rdkit_mol: Chem.Mol = None,
        coordinates: torch.tensor = None,
        atomic_numbers: torch.tensor = None,
        atomic_probs: torch.tensor = None,
        scores: list = ["torchvina", "torchsa"],
        protonated: bool = False
    ):
        self.set_ligand(rdkit_mol, coordinates, atomic_numbers, atomic_probs, protonated)
        self.scores = []
        self.energies = []
        self.trajectory = []
        # create dictionary for storing scores
        self.scores = dict([(score, []) for score in scores])
        self.latents = []

    def set_ligand(
        self,
        rdkit_mol: Chem.Mol = None,
        coordinates: torch.tensor = None,
        atomic_numbers: torch.tensor = None,
        atomic_probs: torch.tensor = None,
        protonated: bool = False
    ):
        """
        Set or reset the ligand, see args above
        """
        self.rdkit_mol = rdkit_mol
        if self.rdkit_mol is not None and coordinates is None:
            self.coordinates = torch.from_numpy(rdkit_mol.GetConformer().GetPositions()).float().to(device)
            self.atomic_numbers = torch.tensor([rdkit_mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(rdkit_mol.GetNumAtoms())]).to(device)
        else:
            self.coordinates = coordinates
            self.atomic_numbers = atomic_numbers
        self.atomic_probs = atomic_probs
        self.protonated = protonated

    def coords(self, add_hydrogens: bool = False) -> torch.tensor:
        """
        get the current coordinates
        """
        if not add_hydrogens or self.protonated:
            return self.coordinates

        mol = self.mol(add_hydrogens=True)
        conf = mol.GetConformer()
        h_coords = []
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetAtomicNum() == 1:
                h_coords.append(
                    [
                        conf.GetAtomPosition(i).x,
                        conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z,
                    ]
                )
        return torch.cat(
            (self.coordinates, torch.tensor(h_coords, device=device)),
            axis=0,
        )

    def mol(self, add_hydrogens: bool = False) -> Chem.Mol:
        """
        get the rdkit mol representation of the ligand.
        """
        if not add_hydrogens or self.protonated:
            return self.rdkit_mol
        return Chem.AddHs(Chem.Mol(self.rdkit_mol), addCoords=True)

    def get_murkco_scaffold_indices(self) -> list:
        """
        get indices of scaffold using rdkit
        """
        Chem.SanitizeMol(self.rdkit_mol)
        return list(self.rdkit_mol.GetSubstructMatch(GetScaffoldForMol(self.rdkit_mol)))

    def ans(self, add_hydrogens: bool = False) -> torch.tensor:
        """
        get the atomic numbers
        """
        if add_hydrogens and not self.protonated:
            mol = self.mol(add_hydrogens=True)
            return torch.tensor(
                [
                    mol.GetAtomWithIdx(i).GetAtomicNum()
                    for i in range(mol.GetNumAtoms())
                ],
                device=self.atomic_numbers.device,
            )
        return self.atomic_numbers

    def ase(self) -> Atoms:
        """
        get an ase Atoms object representation of the ligand
        """
        if self.mol() is not None:
            return Atoms(
                numbers=self.ans(add_hydrogens=True).cpu().numpy(),
                positions=self.coords(add_hydrogens=True).detach().cpu().numpy(),
            )
        return None

    def set_connectivity_and_cutoffs(self):
        """
        Sets the connectivity and corresponding cutoffs. Needed for bond enforcement during geometry optimization.
        """
        atoms = self.ase()
        nlist = NeighborList(cutoffs=natural_cutoffs(atoms), self_interaction=False, bothways=True)
        nlist.update(atoms)
        self.conn_matrix = torch.from_numpy(nlist.get_connectivity_matrix(sparse=False)).type(torch.bool).to(device)
        cutoffs = torch.tensor([natural_cutoffs(atoms)], device=device)
        cutoffs = cutoffs.repeat(cutoffs.shape[1], 1)
        cutoffs = cutoffs + cutoffs.T
        self.cutoffs = torch.masked_select(cutoffs, self.conn_matrix)

    def get_connectivity_and_cutoffs(self):
        """
        Get connectivity and cutoffs
        """
        return self.conn_matrix, self.cutoffs

    def update_coords(self, coordinates: torch.tensor):
        """
        update the coordinates
        """
        # update coordinates
        self.coordinates = coordinates
        # update molecule
        coordinates_np = self.coordinates.detach().cpu().numpy().astype(np.float64)
        mol = self.mol()
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            xcoord, ycoord, zcoord = (
                coordinates_np[i][0],
                coordinates_np[i][1],
                coordinates_np[i][2],
            )
            conf.SetAtomPosition(i, Point3D(xcoord, ycoord, zcoord))

    def append_to_trajectory(self, frame_scores: dict):
        """
        add an rdkit mol to the trajectory and scores to the global scores
        """
        self.trajectory.append(Chem.Mol(self.mol()))
        # this way will raise an error if all scores are not present
        for key in self.scores:
            self.scores[key].append(frame_scores[key])

    def get_score(self, score: str, ind: int = -1) -> np.float64:
        """
        get scores for a specific opt step
        """
        return self.scores[score][ind] if len(self.scores[score]) > 0 else np.nan

    def get_trajectory(self) -> List[Chem.Mol]:
        """
        get the list of rdkit mols and scores dictionary
        """
        return self.trajectory

    def get_scores(self):
        """
        get all the scores from the optimization
        """
        sum_scores = np.zeros(len(self.trajectory))
        for score in self.scores:
            sum_scores += self.scores[score]
        return sum_scores

    def reset_to(self, strategy: str = "best"):
        """
        reset the ligand to the best/last molecule in the trajectory
        """
        if strategy == "last":
            best_idx = -1
        else:
            sum_scores = self.get_scores()
            best_idx = np.argmin(sum_scores)
        mol = self.trajectory[best_idx]
        best_coords = mol.GetConformer().GetPositions()
        best_ans = np.array(
            [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(mol.GetNumAtoms())]
        )
        self.rdkit_mol = Chem.Mol(mol)
        self.atomic_numbers = torch.from_numpy(best_ans).to(device)
        self.update_coords(torch.from_numpy(best_coords).float().to(device))
        for score in self.scores:
            self.scores[score].append(self.scores[score][best_idx])
        self.trajectory.append(self.trajectory[best_idx])

    def __len__(self):
        return len(self.atomic_numbers)

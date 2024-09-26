from rdkit import Chem
import torch
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs
import networkx as nx
import numpy as np
from pymatgen.core import Molecule

from PLOpt.ligand import Ligand
from PLOpt.protein import Protein


class MoleculeValidater:
    """
    Class for performing structural and chemical validity checks on generated ligands.
    Args
        protein: Object defining the protein pocet
        supported_atom_types: Atom types supported during optimization
        verbose: If True, print which checks are failing
    """

    def __init__(self, protein: Protein, supported_atom_types: list[int], verbose: bool = False) -> None:
        self.pocket = protein.get_pocket_ans_and_coords(add_hydrogens=True)
        self.supported_atom_types = supported_atom_types
        self.verbose = verbose

    def validate_molecule(self, ligand: Ligand):
        """
        Function to validate a single molecule. Performs structural and
        chemical checks. See above for args.
        """
        # 1. Valence check
        mol = ligand.mol()

        try:
            mol.UpdatePropertyCache()
        except Exception:
            if self.verbose:
                print('Chem1: Invalid valence')
            return False
        # 2. Hydrogen check
        try:
            Chem.AddHs(mol, addCoords=True)
        except Exception:
            if self.verbose:
                print('Chem2: Could not add Hs')
            return False
        # 3. Fragment check
        mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        if len(mol_frags) != 1:
            if self.verbose:
                print('Chem3: Molecule disconnected')
            return False
        # 4. Santization check
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            if self.verbose:
                print('Chem4: Molecule could not be sanitized')
            return False
        # 5. Supported atom type check
        if not all([atom_type.item() in self.supported_atom_types for atom_type in ligand.ans()]):
            if self.verbose:
                print('Chem5: Atom types not supported')
            return False
        # 6. Atom overlap check
        if not self.check_connected(ligand):
            if self.verbose:
                print('Phys1: Interatomic distances too far')
            return False
        # 7. Atom overlap check
        if not self.check_atom_overlap(ligand):
            if self.verbose:
                print('Phys2: Interatomic distances too close')
            return False
        # 8. Pocket overlap check
        if self.check_pocket_overlap(ligand, self.pocket):
            if self.verbose:
                print('Phys3: Overlap with protein pocket')
            return False

        return True

    def check_connected(self, ligand: Ligand) -> bool:
        """
        check if a ligand is fully connected
        """
        nlist = NeighborList(
            cutoffs=natural_cutoffs(ligand.ase()), self_interaction=False, bothways=True
        )
        nlist.update(ligand.ase())
        if not nx.is_connected(nx.from_numpy_array(nlist.get_connectivity_matrix())):
            return False
        return True

    def check_pocket_overlap(self, ligand: Ligand, pocket: tuple[torch.tensor, torch.tensor]) -> bool:
        """
        check for overlap between ligand and pocket
        """
        pocket_ans, pocket_coords = pocket
        all_atoms = Atoms(
            numbers=np.concatenate((ligand.ase().numbers, pocket_ans.cpu().numpy())),
            positions=np.concatenate((ligand.ase().positions, pocket_coords.cpu().numpy())),
        )
        nlist = NeighborList(
            cutoffs=natural_cutoffs(all_atoms), self_interaction=False, bothways=True
        )
        nlist.update(all_atoms)
        connectivity_matrix = nlist.get_connectivity_matrix(sparse=False)
        if np.any(connectivity_matrix[: len(ligand.ase()), len(ligand.ase()) :]):
            return True
        return False


    def check_atom_overlap(self, ligand: Ligand) -> bool:
        """
        check to make sure if atoms within the ligand are not overlapping
        """
        pmg_struct = Molecule(
            species=ligand.ans(add_hydrogens=True).cpu().numpy(),
            coords=ligand.coords(add_hydrogens=True).detach().cpu().numpy(),
        )
        return pmg_struct.is_valid()
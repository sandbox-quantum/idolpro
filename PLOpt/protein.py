import torch
import numpy as np
from rdkit import Chem
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

from PLOpt.ligand import Ligand
from PLOpt.plopt_utils import PERIODIC_TABLE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Protein(object):
    """
    Protein class to query different files for protein
    Args:
        protein_file: pdb file of protein
        ref_ligand: sdf file of reference ligand (defines pocket)
    """

    def __init__(self, protein_file, ref_ligand_file):
        self.protein_file = protein_file
        self.ref_ligand = Ligand(rdkit_mol=Chem.SDMolSupplier(ref_ligand_file, sanitize=False)[0])
        # fix to make sure reference ligand has no Hs
        if 1 in self.ref_ligand.ans():
            emol = Chem.rdchem.EditableMol(self.ref_ligand.mol())
            for i, atom_type in zip(
                range(len(self.ref_ligand) - 1, 0, -1),
                reversed(self.ref_ligand.ans().tolist())
            ):
                if atom_type == 1:
                    emol.RemoveAtom(i)
            self.ref_ligand = Ligand(emol.GetMol())

        self.protein_file_w_Hs = protein_file[:-4] + '_Hs.pdb'

    def get_file(self):
        return self.protein_file

    def get_file_w_Hs(self):
        return self.protein_file_w_Hs

    def get_pdbqt_file(self):
        return f"{self.protein_file}qt"

    def get_pdbqt_file_w_Hs(self):
        return f"{self.protein_file_w_Hs}qt"

    def get_reference_ligand(self):
        return self.ref_ligand

    def get_reference_scaffold(self):
        return self.ref_ligand.get_murkco_scaffold_indices()

    def get_pocket_by_residue(
        self, dist_cutoff: int = 8.0, use_com: bool = False, add_hydrogens: bool = False
    ) -> list[object]:
        """
        get the pocket by finding residues within dist_cutoff of the ligand
        """
        ligand_coords = self.ref_ligand.coords()
        if use_com:  # only use center or mass to get pocket residues
            ligand_coords = torch.mean(ligand_coords, axis=0, keepdim=True)
        pdb_struct = PDBParser(QUIET=True).get_structure(
            "", self.get_file_w_Hs() if add_hydrogens else self.get_file()
        )[0]
        pocket_residues = []
        for residue in pdb_struct.get_residues():
            res_coords = torch.from_numpy(
                np.array([a.get_coord() for a in residue.get_atoms()])
            ).to(device)
            if (
                is_aa(residue.get_resname(), standard=True)
                and torch.cdist(res_coords, ligand_coords).min() < dist_cutoff
            ):
                pocket_residues.append(residue)

        return pocket_residues

    def get_pocket_by_atom(
        self, dist_cutoff: int = 8.0, use_com: bool = False, add_hydrogens: bool = False
    ) -> list[object]:
        """
        get the pocket by finding atoms within dist_cutoff of the ligand
        """
        ligand_coords = self.ref_ligand.coords()
        if use_com:  # only use center or mass to get pocket residues
            ligand_coords = torch.mean(ligand_coords, axis=0, keepdim=True)
        pdb_struct = PDBParser(QUIET=True).get_structure(
            "", self.get_file_w_Hs() if add_hydrogens else self.get_file()
        )[0]
        pocket_ans = []
        for residue in pdb_struct.get_residues():
            if is_aa(residue.get_resname(), standard=True):
                for at in residue.get_atoms():
                    at_coords = torch.from_numpy(np.array([at.get_coord()])).to(device)
                    if torch.cdist(at_coords, ligand_coords).min() < dist_cutoff:
                        pocket_ans.append(at)

        return pocket_ans


    def get_pocket_ans_and_coords(
        self, by_resi: bool = True, dist_cutoff: int = 8.0, use_com: bool = False, add_hydrogens: bool = False
    ) -> tuple[torch.tensor, torch.tensor]:
        """
        Get the pocket atomic numbers and coordinates
        """
        if by_resi:
            residues = self.get_pocket_by_residue(dist_cutoff, use_com, add_hydrogens)
            pocket_atoms = [a for res in residues for a in res.get_atoms()]
        else:
            pocket_atoms = self.get_pocket_by_atom(dist_cutoff, use_com, add_hydrogens)
        # some pdbs seem to have Hs even when not protonated
        return (
            torch.tensor(
                # some pdbs seem to have Hs even when not protonated
                [PERIODIC_TABLE.index(a.element) for a in pocket_atoms if a.element != 'H' or add_hydrogens],
                device=device,
            ),
            torch.tensor(
                np.array([a.get_coord() for a in pocket_atoms if a.element != 'H' or add_hydrogens]),
                device=device,
            ).float(),
        )


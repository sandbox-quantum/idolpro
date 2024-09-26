"""
util functions for optimization pipeline
"""

import os
import sys
import subprocess
import tempfile
import time
import datetime

from rdkit import Chem

import torch
import numpy as np
from meeko import MoleculePreparation, PDBQTWriterLegacy
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from openbabel import openbabel

from PLOpt.ligand import Ligand

sys.path.append(os.path.join(Chem.RDConfig.RDContribDir, "SA_Score"))
# now you can import sascore!
import sascorer

ob_log_handler = openbabel.obErrorLog
ob_log_handler.StopLogging()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PERIODIC_TABLE = (
    ["Dummy"]
    + """
    H                                                                                                                           He
    Li  Be                                                                                                  B   C   N   O   F   Ne
    Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
    K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
    Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
    Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
    Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
    """.strip().split()
)


def eval_qvina(
    receptor_pdbqt_file: str,
    ligand: Ligand,
    dock: bool = False,
    local: bool = False,
    add_hydrogens: bool = False,
    qvina_path: str = "./qvina2.1"
):
    """
    Evaluate a ligand stored in ligand_sdf_file with qvina.
    Args:
        receptor_pdbqt_file: the pdbqt file of the receptor
        ligand: the ligand
        dock: whether to dock the ligand in the receptor
        local: whether to only apply local optimization with qvina
        add_hydrogens: whether to add hyrdogens before running qvina
        qvina_path: path to qvina executable
    Returns:
        vina score
    """

    ligand_sdf_file = tempfile.NamedTemporaryFile(suffix='.sdf').name
    write_sdf_file(ligand_sdf_file, [ligand.mol(add_hydrogens=add_hydrogens)])

    ligand_pdbqt_file = tempfile.NamedTemporaryFile(suffix='.pdbqt').name
    subprocess.run(
        f"obabel {ligand_sdf_file} -O {ligand_pdbqt_file} -f 1 -l 1",
        shell=True,
        capture_output=True,
    )

    cx, cy, cz = (
        ligand.mol()
        .GetConformer()
        .GetPositions()
        .mean(0)
    )

    # Vina with docking
    if dock:
        ligand_docked_pdbqt_file = tempfile.NamedTemporaryFile(suffix='.pdbqt').name
        out = subprocess.check_output(
            f"{qvina_path} --receptor {receptor_pdbqt_file} "
            f"--ligand {ligand_pdbqt_file} "
            f"--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} "
            f"--size_x 20 --size_y 20 --size_z 20 "
            f"--exhaustiveness 16 "
            f"--out {ligand_docked_pdbqt_file}",
            shell=True
        )
        update_ligand_from_pdbqt(ligand_pdbqt_file, ligand_docked_pdbqt_file, ligand)
        write_sdf_file(ligand_sdf_file, [ligand.mol(add_hydrogens=add_hydrogens)])
        subprocess.run(
            f"obabel {ligand_sdf_file} -O {ligand_pdbqt_file} -f 1 -l 1",
            shell=True,
            capture_output=True,
        )

    elif local:
        ligand_docked_pdbqt_file = tempfile.NamedTemporaryFile(suffix='.pdbqt').name
        subprocess.check_output(
            f"{qvina_path} --receptor {receptor_pdbqt_file} "
            f"--ligand {ligand_pdbqt_file} "
            f"--center_x {cx:.4f} --center_y {cy:.4f} --center_z {cz:.4f} "
            f"--size_x 20 --size_y 20 --size_z 20 "
            f"--local_only "
            f"--out {ligand_docked_pdbqt_file}",
            shell=True
        )
        update_ligand_from_pdbqt(ligand_pdbqt_file, ligand_docked_pdbqt_file, ligand)
        write_sdf_file(ligand_sdf_file, [ligand.mol(add_hydrogens=add_hydrogens)])
        subprocess.run(
            f"obabel {ligand_sdf_file} -O {ligand_pdbqt_file} -f 1 -l 1",
            shell=True,
            capture_output=True,
        )

    # Score
    out = subprocess.run(
        f"{qvina_path} --receptor {receptor_pdbqt_file} "
        f"--ligand {ligand_pdbqt_file} "
        f"--score_only",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout

    score = None
    for ln in out.splitlines():
        if ln.startswith("Affinity"):
            score = float(ln.split(" ")[1])

    if score is None:
        raise RuntimeError(out)

    return score


def update_ligand_from_pdbqt(
    pdbqt_file_in: str, pdbqt_file_out: str, ligand: Ligand
) -> Ligand:
    """
    update the current ligand from a pdqbt file
    """
    old_coords = []
    with open(pdbqt_file_in, encoding="utf-8") as file_in:
        for line in file_in.readlines():
            if len(line) > 4 and line[:4] == "ATOM":
                if line[77:79].strip() not in ["H", "HD"]:
                    old_coords.append(
                        [
                            float(line[30:38].strip()),
                            float(line[38:46].strip()),
                            float(line[46:54].strip()),
                        ]
                    )
            if line[:6] == "ENDMDL":
                break

    new_coords = []
    with open(pdbqt_file_out, encoding="utf-8") as file_out:
        for line in file_out.readlines():
            if len(line) > 4 and line[:4] == "ATOM":
                if line[77:79].strip() not in ["H", "HD"]:
                    new_coords.append(
                        [
                            float(line[30:38].strip()),
                            float(line[38:46].strip()),
                            float(line[46:54].strip()),
                        ]
                    )
            if line[:6] == "ENDMDL":
                break

    pdbqt_to_sdf_map = torch.argmin(
        torch.cdist(
            ligand.coords()[None, :, :],
            torch.tensor(old_coords, device=device)[None, :, :],
        )[0],
        dim=-1,
    )
    new_coords = torch.tensor([new_coords[i] for i in pdbqt_to_sdf_map], device=device)
    ligand.update_coords(new_coords)


def rdkitmol_to_pdbqt_string(rdkit_mol: Chem.Mol) -> str:
    """
    function to map an rdkit moleculeto a pdbqt string (used by torchvina score)
    """
    preparator = MoleculePreparation(rigid_macrocycles=True)
    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(
        preparator.prepare(rdkit_mol)[0], bad_charge_ok=True
    )
    if is_ok:
        return pdbqt_string
    else:
        raise (RuntimeError(error_msg))


def write_sdf_file(sdf_path: str, molecules: list[Chem.Mol], kekulize: bool = True, confId: int = -1):
    """
    Write molecules to sdf file
    """
    w = Chem.SDWriter(str(sdf_path))
    if not kekulize:
        w.SetKekulize(False)
    for m in molecules:
        if m is not None:
            w.write(m, confId=confId)


def flatten_for_pyg(samples: list[torch.tensor]) -> list[torch.tensor]:
    """
    Flatten data for pytorch geometric
    """
    return_dict = {}
    for k in samples[0]:
        return_dict[k] = []
    return_dict["indices"] = []
    return_dict["natoms"] = []
    for i, sample in enumerate(samples):
        atomic_numbers = sample["atomic_numbers"]
        return_dict["atomic_numbers"].append(atomic_numbers)
        return_dict["coordinates"].append(sample["coordinates"])
        return_dict["indices"].append(
            torch.ones(len(atomic_numbers), dtype=torch.int64, device=device) * i
        )
        return_dict["natoms"].append(len(atomic_numbers))

    return_dict["coordinates"] = torch.cat(return_dict["coordinates"])
    return_dict["atomic_numbers"] = torch.cat(return_dict["atomic_numbers"])
    return_dict["indices"] = torch.cat(return_dict["indices"])
    return_dict["natoms"] = torch.tensor(return_dict["natoms"], device=device)

    return return_dict


def collate_fn_pytorch_geometric(samples: list[torch.tensor]) -> list[torch.tensor]:
    """
    Collate function for pytorch geometric
    """
    samples = flatten_for_pyg(samples)
    return _make_float32(samples)


class OCPDataDummy:
    """
    Wrapper for data going into OCP model
    """

    def __init__(
        self, atomic_numbers=None, coordinates=None, indices=None, natoms=None
    ):
        self.atomic_numbers = atomic_numbers
        self.pos = coordinates
        self.cell = None
        self.batch = indices
        self.natoms = natoms
        self.data_idx = torch.zeros(len(self.atomic_numbers), dtype=torch.int64, device=device)


class Batcher(torch.utils.data.Dataset):
    """
    Dataset class to batch predictions with ANI
    """

    def __init__(self, ans: list[torch.tensor], coords: list[torch.tensor]):
        self.ans = ans
        self.coords = coords

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> dict:
        samples = {"atomic_numbers": self.ans[idx], "coordinates": self.coords[idx]}
        return samples

def _make_float32(samples: list[torch.tensor]) -> list[torch.tensor]:
    """
    Function to change all dtypes of samples to 32 bit
    """
    for key, val in samples.items():
        if val.dtype == torch.float64:
            samples[key] = val.float()
        if val.dtype == torch.int32:
            samples[key] = val.long()
    return samples


def write_mol_to_pdbqt(mol, fname):
    """
    Writes rdkit mol to pdbqt file
    """
    preparator = MoleculePreparation(rigid_macrocycles=True)
    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(
        preparator.prepare(mol)[0], bad_charge_ok=True)
    if is_ok:
        with open(fname, 'w') as f:
            f.write(pdbqt_string)
    else:
        raise (RuntimeError(error_msg))

def evaluate_sa(mol):
    """
    Evaluate the SA score of a molecule
    """
    return sascorer.calculateScore(mol)

def evaluate_qed(mol):
    """
    Evaluate the QED score of a molecule
    """
    return Chem.QED.qed(mol)

def get_pretty_time(start_time):
    """
    Format time to legible form
    """
    return str(datetime.timedelta(seconds=time.time() - start_time))[:7]


def make_mol_openbabel(positions, atomic_numbers):
    """
    Build an RDKit molecule using openbabel for creating bonds
    Taken from DiffSBDD
    Args:
        positions: N x 3
        atom_types: N
    Returns:
        rdkit molecule
    """
    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name

        # Write xyz file
        write_xyz_file(positions, [PERIODIC_TABLE[i] for i in atomic_numbers], tmp_file)

        # Convert to sdf file with openbabel
        # openbabel will add bonds
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "sdf")
        ob_mol = openbabel.OBMol()
        obConversion.ReadFile(ob_mol, tmp_file)
        obConversion.WriteFile(ob_mol, tmp_file)
        # Read sdf file with RDKit
        tmp_mol = Chem.SDMolSupplier(tmp_file, sanitize=False)[0]

    # Build new molecule. This is a workaround to remove radicals.
    mol = Chem.RWMol()
    for atom in tmp_mol.GetAtoms():
        mol.AddAtom(Chem.Atom(atom.GetSymbol()))
    mol.AddConformer(tmp_mol.GetConformer(0))

    for bond in tmp_mol.GetBonds():
        mol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                    bond.GetBondType())

    return mol

def write_xyz_file(coords, atom_types, filename):
    """
    Write coordinates and atom types to xyz file.
    """
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, 'w') as f:
        f.write(out)


def batchify(iterable, n=16):
    """
    Create batches of size n out of iterable
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
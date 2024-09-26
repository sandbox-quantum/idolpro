import os
import sys
from pathlib import Path
from collections import defaultdict

from Bio.PDB import PDBParser
from Bio.PDB import PDBIO, Select
import numpy as np
from PLOpt.plopt_utils import make_mol_openbabel, write_sdf_file

cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Model0(Select):
    def accept_model(self, model):
        return model.id == 0

if __name__ == "__main__":
    data_dir= sys.argv[1]
    split_file = f"./DiffSBDD/data/moad_test.txt"
    output_dir = f"./data/binding_moad_test"
    Path(output_dir).mkdir(exist_ok=True)
    with open(split_file, 'r') as f:
        pocket_ids = f.read().split(',')
        # (ec-number, protein, molecule tuple)
    test_split = [(x.split('_')[0][:4], (x.split('_')[1],))
                    for x in pocket_ids]
    test_split_dict = defaultdict(list)
    for p, m in test_split:
        test_split_dict[p].append(m)

    for p in test_split_dict:
        pdb_successful = set()
        for pdbfile in sorted(Path(data_dir).glob(f"{p.lower()}.bio*")):
            if len(test_split_dict[p]) == len(pdb_successful):
                continue

            name = f"{p}-{pdbfile.suffix[1:]}"
            pdb_struct = PDBParser(QUIET=True).get_structure('', pdbfile)
            struct_copy = pdb_struct.copy()

            for m in test_split_dict[p]:
                if m[0] in pdb_successful:
                    continue

                ligand_name, ligand_chain, ligand_resi = m[0].split(':')
                ligand_resi = int(ligand_resi)
                try:
                    struct_copy[0][ligand_chain].detach_child((f'H_{ligand_name}', ligand_resi, ' '))
                except KeyError:
                    continue

                try:
                    try:
                        residues = {obj.id[1]: obj for obj in
                                    pdb_struct[0][ligand_chain].get_residues()}
                    except KeyError as e:
                        raise KeyError(f'Chain {e} not found ({pdbfile}, '
                                    f'{ligand_name}:{ligand_chain}:{ligand_resi})')
                    ligand = residues[ligand_resi]
                    assert ligand.get_resname() == ligand_name, \
                        f"{ligand.get_resname()} != {ligand_name}"
                except (KeyError, AssertionError, FileNotFoundError,
                        IndexError, ValueError) as e:
                    continue

                # Create SDF file
                lig_atoms = [a for a in ligand.get_atoms()
                             if (a.element.capitalize() in [5, 6, 7, 8, 9, 15, 16, 17, 35, 53])]
                lig_coords = np.array([a.get_coord() for a in lig_atoms])

                ligand_mol = make_mol_openbabel(lig_coords, lig_atoms)
                write_sdf_file(f"{output_dir}/{name}_{m[0]}.sdf", [ligand_mol])
                pdb_successful.add(m[0])

            io = PDBIO()
            io.set_structure(struct_copy)
            io.save(f"{output_dir}/{name}.pdb", select=Model0())
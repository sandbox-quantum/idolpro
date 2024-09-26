import os
import sys
import pymol

protein_file = sys.argv[1]
protein_file_w_Hs = protein_file.replace(".pdb", "_Hs.pdb")
if not os.path.exists(protein_file_w_Hs):
    with open(protein_file):
        pymol.cmd.load(protein_file, "protein")
        pymol.cmd.h_add()
        pymol.cmd.save(protein_file_w_Hs)
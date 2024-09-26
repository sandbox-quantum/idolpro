mambapath=$1
protein=$2

${mambapath}/envs/idolpro/bin/python scripts/add_hydrogens.py ${protein}

${mambapath}/envs/mgltools/bin/python DiffSBDD/analysis/prepare_receptor4.py -r ${protein} -o ${protein}qt -A checkhydrogens -e

${mambapath}/envs/mgltools/bin/python DiffSBDD/analysis/prepare_receptor4.py -r ${protein::-4}_Hs.pdb -o ${protein::-4}_Hs.pdbqt -A checkhydrogens -e

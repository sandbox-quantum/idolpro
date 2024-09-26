import os
import sys
import csv
import glob


# def pdbs_to_pdbqts(pdb_dir, pdbqt_dir, dataset):
#     with open('/shared/akadan/protein-ligand/data/DiffSBDD_test.tsv') as fd:
#         rd = csv.reader(fd, delimiter="\t", quotechar='"')
#         for i, row in enumerate(rd):
#             if i == 0: continue
#             file, _ = row
#             name = file.split('/')[-1].split('_')[0]
#             file = os.path.join(pdb_dir, file)
#             outfile = os.path.join(pdbqt_dir, name + '.pdbqt')
#             pdb_to_pdbqt(file, outfile, dataset)
#             print('Wrote converted file to {}'.format(outfile))

def pdbs_to_pdbqts(pdb_dir, dataset='crossdocked'):
    dock_dirs = os.listdir(pdb_dir)
    for dock_dir in dock_dirs:
        files = glob.glob(os.path.join(pdb_dir, dock_dir, '*Hs.pdb'))
        assert len(files) == 1
        for file in files:
            outfile = os.path.join(pdb_dir, dock_dir, file.split('/')[-1] + 'qt')
            pdb_to_pdbqt(file, outfile, dataset)
            print('Wrote converted file to {}'.format(outfile))


def pdb_to_pdbqt(pdb_file, pdbqt_file, dataset='crossdocked'):
    if os.path.exists(pdbqt_file):
        return pdbqt_file
    if dataset == 'crossdocked':
        os.system('prepare_receptor4.py -r {} -o {}'.format(pdb_file, pdbqt_file))
    elif dataset == 'bindingmoad':
        os.system('prepare_receptor4.py -r {} -o {} -A checkhydrogens -e'.format(pdb_file, pdbqt_file))
    else:
        raise NotImplementedError
    return pdbqt_file


if __name__ == '__main__':
    print(os.getcwd())
    pdbs_to_pdbqts(sys.argv[1])

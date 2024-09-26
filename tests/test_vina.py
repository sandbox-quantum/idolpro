import unittest
import pathlib
import os
import torch
import pytorch_lightning as pl
from rdkit import Chem
import numpy as np
from pathlib import Path

from PLOpt.torchvina import TorchVinaScore
from PLOpt.ligand import Ligand
from PLOpt.protein import Protein


CWD = pathlib.Path(__file__).parent.resolve()
torch.use_deterministic_algorithms(True, warn_only=True)
pl.seed_everything(123456)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestVina(unittest.TestCase):

    def setUp(self):
        data_dir = f'{CWD}/14gs_cbd'
        self.vina_scorer = TorchVinaScore(Protein(str(list(Path(data_dir).rglob('*pocket10.pdb'))[0]),
                                                  str(list(Path(data_dir).rglob('*.sdf'))[0])))
        self.lig = Ligand(rdkit_mol=Chem.SDMolSupplier(str(list(Path(data_dir).rglob('*.sdf'))[0]))[0])

    def test_gauss1(self):
        ligand_features = self.vina_scorer._get_lig_vectors([self.lig])
        vina_features = self.vina_scorer._get_vina_feature_vectors(ligand_features)
        g1 = self.vina_scorer._calc_gauss(vina_features,
                                          o=self.vina_scorer.params["cutoffs"]["o1"],
                                          s=self.vina_scorer.params["cutoffs"]["s1"])
        self.assertAlmostEqual(g1.item(), 50.19061, places=2)

    def test_gauss2(self):
        ligand_features = self.vina_scorer._get_lig_vectors([self.lig])
        vina_features = self.vina_scorer._get_vina_feature_vectors(ligand_features)
        g2 = self.vina_scorer._calc_gauss(vina_features,
                                          o=self.vina_scorer.params["cutoffs"]["o2"],
                                           s=self.vina_scorer.params["cutoffs"]["s2"])
        self.assertAlmostEqual(g2.item(), 844.90643, places=2)

    def test_repulsion(self):
        ligand_features = self.vina_scorer._get_lig_vectors([self.lig])
        vina_features = self.vina_scorer._get_vina_feature_vectors(ligand_features)
        rep = self.vina_scorer._calc_repulsion(vina_features)
        self.assertAlmostEqual(rep.item(), 1.29313, places=2)

    def test_hydrophobic(self):
        ligand_features = self.vina_scorer._get_lig_vectors([self.lig])
        vina_features = self.vina_scorer._get_vina_feature_vectors(ligand_features)
        hydro = self.vina_scorer._calc_hydrophobic(vina_features)
        self.assertAlmostEqual(hydro.item(), 33.03773, places=2)

    def test_hbonding(self):
        ligand_features = self.vina_scorer._get_lig_vectors([self.lig])
        vina_features = self.vina_scorer._get_vina_feature_vectors(ligand_features)
        hbond = self.vina_scorer._calc_hbonding(vina_features)
        self.assertAlmostEqual(hbond.item(), 1.60891, places=2)

    def test_inter_score_1(self):
        vina_score = self.vina_scorer.score([self.lig], intra=False)
        self.assertAlmostEqual(vina_score.item(),  -6.76385, places=2)

    def test_inter_score_2(self):
        data_dir = f'{CWD}/2azy_chd'
        vina_scorer = TorchVinaScore(Protein(str(list(Path(data_dir).rglob('*pocket10.pdb'))[0]),
                                                  str(list(Path(data_dir).rglob('*.sdf'))[0])))
        lig = Ligand(rdkit_mol=Chem.SDMolSupplier(str(list(Path(data_dir).rglob('*.sdf'))[0]))[0])
        vina_score = vina_scorer.score([lig], intra=False)
        self.assertAlmostEqual(vina_score.item(), -10.90484, places=2)
    
    def test_inter_score_3(self):
        data_dir = f'{CWD}/3af2_gcp'
        vina_scorer = TorchVinaScore(Protein(str(list(Path(data_dir).rglob('*pocket10.pdb'))[0]),
                                                  str(list(Path(data_dir).rglob('*.sdf'))[0])))
        lig = Ligand(rdkit_mol=Chem.SDMolSupplier(str(list(Path(data_dir).rglob('*.sdf'))[0]))[0])
        vina_score = vina_scorer.score([lig], intra=False)
        self.assertAlmostEqual(vina_score.item(), -7.74129, places=2)

    def test_inter_score_4(self):
        data_dir = f'{CWD}/5w2g_adp'
        vina_scorer = TorchVinaScore(Protein(str(list(Path(data_dir).rglob('*pocket10.pdb'))[0]),
                                                  str(list(Path(data_dir).rglob('*.sdf'))[0])))
        lig = Ligand(rdkit_mol=Chem.SDMolSupplier(str(list(Path(data_dir).rglob('*.sdf'))[0]))[0])
        vina_score = vina_scorer.score([lig], intra=False)
        self.assertAlmostEqual(vina_score.item(), -5.68741, places=2)

    def test_intra_score_1(self):
        vina_score = self.vina_scorer.score([self.lig], inter=False)
        self.assertAlmostEqual(vina_score.item(), -0.54302, places=2)

    def test_intra_score_2(self):
        data_dir = f'{CWD}/2azy_chd'
        vina_scorer = TorchVinaScore(Protein(str(list(Path(data_dir).rglob('*pocket10.pdb'))[0]),
                                                  str(list(Path(data_dir).rglob('*.sdf'))[0])))
        lig = Ligand(rdkit_mol=Chem.SDMolSupplier(str(list(Path(data_dir).rglob('*.sdf'))[0]))[0])
        vina_score = vina_scorer.score([lig], inter=False)
        self.assertAlmostEqual(vina_score.item(), -0.38838, places=2)
    
    def test_intra_score_3(self):
        data_dir = f'{CWD}/3af2_gcp'
        vina_scorer = TorchVinaScore(Protein(str(list(Path(data_dir).rglob('*pocket10.pdb'))[0]),
                                                  str(list(Path(data_dir).rglob('*.sdf'))[0])))
        lig = Ligand(rdkit_mol=Chem.SDMolSupplier(str(list(Path(data_dir).rglob('*.sdf'))[0]))[0])
        vina_score = vina_scorer.score([lig], inter=False)
        self.assertAlmostEqual(vina_score.item(), -1.07873, places=2)

    def test_intra_score_4(self):
        data_dir = f'{CWD}/5w2g_adp'
        vina_scorer = TorchVinaScore(Protein(str(list(Path(data_dir).rglob('*pocket10.pdb'))[0]),
                                                  str(list(Path(data_dir).rglob('*.sdf'))[0])))
        lig = Ligand(rdkit_mol=Chem.SDMolSupplier(str(list(Path(data_dir).rglob('*.sdf'))[0]))[0])
        vina_score = vina_scorer.score([lig], inter=False)
        self.assertAlmostEqual(vina_score.item(), -1.27643, places=2)

    def test_vinardo_inter_score_1(self):
        data_dir = f'{CWD}/14gs_cbd'
        vina_scorer = TorchVinaScore(Protein(str(list(Path(data_dir).rglob('*pocket10.pdb'))[0]), 
                                             str(list(Path(data_dir).rglob('*.sdf'))[0])), vinardo=True)
        lig = Ligand(rdkit_mol=Chem.SDMolSupplier(str(list(Path(data_dir).rglob('*.sdf'))[0]))[0])
        vina_score = vina_scorer.score([lig], intra=False)
        self.assertAlmostEqual(vina_score.item(),  -6.09505, places=2)

    def test_vinardo_inter_score_2(self):
        data_dir = f'{CWD}/2azy_chd'
        vina_scorer = TorchVinaScore(Protein(str(list(Path(data_dir).rglob('*pocket10.pdb'))[0]),
                                             str(list(Path(data_dir).rglob('*.sdf'))[0])), vinardo=True)
        lig = Ligand(rdkit_mol=Chem.SDMolSupplier(str(list(Path(data_dir).rglob('*.sdf'))[0]))[0])
        vina_score = vina_scorer.score([lig], intra=False)
        self.assertAlmostEqual(vina_score.item(), -8.95190, places=2)
    
    def test_vinardo_inter_score_3(self):
        data_dir = f'{CWD}/3af2_gcp'
        vina_scorer = TorchVinaScore(Protein(str(list(Path(data_dir).rglob('*pocket10.pdb'))[0]),
                                             str(list(Path(data_dir).rglob('*.sdf'))[0])), vinardo=True)
        lig = Ligand(rdkit_mol=Chem.SDMolSupplier(str(list(Path(data_dir).rglob('*.sdf'))[0]))[0])
        vina_score = vina_scorer.score([lig], intra=False)
        self.assertAlmostEqual(vina_score.item(), -7.20317, places=2)

    def test_vinardo_inter_score_4(self):
        data_dir = f'{CWD}/5w2g_adp'
        vina_scorer = TorchVinaScore(Protein(str(list(Path(data_dir).rglob('*pocket10.pdb'))[0]),
                                             str(list(Path(data_dir).rglob('*.sdf'))[0])), vinardo=True)
        lig = Ligand(rdkit_mol=Chem.SDMolSupplier(str(list(Path(data_dir).rglob('*.sdf'))[0]))[0])
        vina_score = vina_scorer.score([lig], intra=False)
        self.assertAlmostEqual(vina_score.item(), -5.01322, places=2)

    def test_vinardo_intra_score_1(self):
        data_dir = f'{CWD}/14gs_cbd'
        vina_scorer = TorchVinaScore(Protein(str(list(Path(data_dir).rglob('*pocket10.pdb'))[0]),
                                             str(list(Path(data_dir).rglob('*.sdf'))[0])), vinardo=True)
        lig = Ligand(rdkit_mol=Chem.SDMolSupplier(str(list(Path(data_dir).rglob('*.sdf'))[0]))[0])
        vina_score = vina_scorer.score([lig], inter=False)
        self.assertAlmostEqual(vina_score.item(), -0.65706, places=2)

    def test_vinardo_intra_score_2(self):
        data_dir = f'{CWD}/2azy_chd'
        vina_scorer = TorchVinaScore(Protein(str(list(Path(data_dir).rglob('*pocket10.pdb'))[0]),
                                             str(list(Path(data_dir).rglob('*.sdf'))[0])), vinardo=True)
        lig = Ligand(rdkit_mol=Chem.SDMolSupplier(str(list(Path(data_dir).rglob('*.sdf'))[0]))[0])
        vina_score = vina_scorer.score([lig],  inter=False)
        self.assertAlmostEqual(vina_score.item(), -0.22233, places=2)
    
    def test_vinardo_intra_score_3(self):
        data_dir = f'{CWD}/3af2_gcp'
        vina_scorer = TorchVinaScore(Protein(str(list(Path(data_dir).rglob('*pocket10.pdb'))[0]),
                                             str(list(Path(data_dir).rglob('*.sdf'))[0])), vinardo=True)
        lig = Ligand(rdkit_mol=Chem.SDMolSupplier(str(list(Path(data_dir).rglob('*.sdf'))[0]))[0])
        vina_score = vina_scorer.score([lig], inter=False)
        self.assertAlmostEqual(vina_score.item(), -0.75400, places=2)

    def test_vinardo_intra_score_4(self):
        data_dir = f'{CWD}/5w2g_adp'
        vina_scorer = TorchVinaScore(Protein(str(list(Path(data_dir).rglob('*pocket10.pdb'))[0]),
                                             str(list(Path(data_dir).rglob('*.sdf'))[0])), vinardo=True)
        lig = Ligand(rdkit_mol=Chem.SDMolSupplier(str(list(Path(data_dir).rglob('*.sdf'))[0]))[0])
        vina_score = vina_scorer.score([lig], inter=False)
        self.assertAlmostEqual(vina_score.item(), -1.18311, places=2)

    def test_full_score_batch(self):
        vina_score = self.vina_scorer.score([self.lig for _ in range(2)], intra=False)
        vina_score = vina_score.cpu().numpy()
        assert len(vina_score) == 2
        np.testing.assert_almost_equal(vina_score, -6.76385, decimal=2)
import os

import torch
from torch_scatter import segment_coo
import numpy as np
from rdkit import Chem
from rdkit.Chem import Lipinski
import MDAnalysis as mda

from PLOpt.scoring_model import ScoringModel
from PLOpt.plopt_utils import PERIODIC_TABLE, rdkitmol_to_pdbqt_string
from PLOpt.protein import Protein
from PLOpt.ligand import Ligand

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# from https://github.com/ccsb-scripps/AutoDock-Vina/blob/develop/src/lib/atom_constants.h
vina_params = {
    "vdw_radii": {
        "C": 1.9,
        "A": 1.9,
        "N": 1.8,
        "O": 1.7,
        "F": 1.5,
        "P": 2.1,
        "S": 2.0,
        "Cl": 1.8,
        "Br": 2.0,
        "I": 2.2,
    },
    "weights": {
        "w1": -0.035579,
        "w2": -0.005156,
        "w3": 0.840245,
        "w4": -0.035069,
        "w5": -0.587439,
        "w6": 0.05846,
    },
    "cutoffs": {
        "s1": 0.5,
        "o1": 0.0,
        "o2": 3.0,
        "s2": 2.0,
        "p1": 0.5,
        "p2": 1.5,
        "h1": -0.7,
    },
}
vinardo_params = {
    "vdw_radii": {"C": 2.0, "N": 1.7, "O": 1.6},
    "weights": {
        "w1": -0.045,
        "w2": 0.000,
        "w3": 0.800,
        "w4": -0.035,
        "w5": -0.600,
        "w6": 0.02,
    },
    "cutoffs": {"s1": 0.8, "p1": 0.0, "p2": 2.5, "h1": -0.6},
}

covalent_radii = {
    "H": 0.407,
    "C": 0.847,
    "N": 0.825,
    "O": 0.803,
    "F": 0.781,
    "P": 1.166,
    "S": 1.122,
    "Cl": 1.089,
    "Br": 1.254,
    "I": 1.463,
}  # includes bond allowance factor
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchVinaScore(ScoringModel):
    """
    Model for calculating torchvina. Re-implements Vina score
    with pytorch operations to allow for automatic differentiation.
    Args:
        protein: protein class containing different files for protein
        vinardo: whether to score instead with Vinardo
        verbose: whether to print out individual contributions to vina score
    """

    def __init__(
        self, protein: Protein, vinardo: bool = False, verbose: bool = False
    ) -> None:
        self.rec_pdbqt_file = protein.get_pdbqt_file_w_Hs()
        self.verbose = verbose
        self.params = dict(vina_params)
        # use parameters for vinardo
        self.vinardo = vinardo
        if vinardo:
            for key in self.params:
                self.params[key].update(vinardo_params[key])
        # get attributes of receptor (since this will be constant throughout, and does not require differentiability)
        self._get_rec_vectors()

    def score(
        self, ligands: list[Ligand], inter: bool = True, intra: bool = True
    ) -> torch.tensor:
        """
        Score ligands according to the Vina score
        Args:
            ligands: list of ligands to score
            inter: whether to include intermolecular component
            intra: whether to include intramolecule component
        Returns:
            torch.Tensor: Vina scores of ligands
        """
        print_string = ""
        total_energy = 0
        ligands_features = self._get_lig_vectors(ligands)

        if intra:
            vina_features = self._get_vina_feature_vectors(
                ligands_features=ligands_features, ligands=ligands, inter=False
            )
            intra_energy = self._score(vina_features)
            total_energy += intra_energy
            print_string += f"\t mean intra: {np.mean(intra_energy.detach().cpu().numpy())}\n"

        if inter:
            vina_features = self._get_vina_feature_vectors(
                ligands_features=ligands_features, inter=True
            )
            inter_energy = self._score(vina_features)
            inter_energy = inter_energy / (
                1 + self.params["weights"]["w6"] * ligands_features["torsions"]
            )
            total_energy += inter_energy
            print_string += f"\t mean inter: {np.mean(inter_energy.detach().cpu().numpy())}\n"

        if self.verbose:
            print_string += f"\t mean total: {np.mean(total_energy.detach().cpu().numpy())}\n"
            print(print_string)

        return total_energy

    def _score(self, vina_features: dict) -> torch.tensor:
        """
        Utiliy function for score. Allows calling same scoring algorithm
        with both intermolecular and intramolecular feature vectors.
        """
        # Gauss 1
        gauss_1 = self._calc_gauss(
            vina_features,
            o=self.params["cutoffs"]["o1"],
            s=self.params["cutoffs"]["s1"],
        )
        # Gauss 2
        gauss_2 = self._calc_gauss(
            vina_features,
            o=self.params["cutoffs"]["o2"],
            s=self.params["cutoffs"]["s2"],
        )
        # Repulsion
        repulsion = self._calc_repulsion(vina_features)
        # Hydrophobic
        hydrophobic = self._calc_hydrophobic(vina_features)
        # HBonding
        hbonding = self._calc_hbonding(vina_features)

        if self.verbose:
            print(
                f"\t mean gauss1: {np.mean(gauss_1.detach().cpu().numpy())}\n"
                + f"\t mean gauss2: {np.mean(gauss_2.detach().cpu().numpy())}\n"
                + f"\t mean repulsion: {np.mean(repulsion.detach().cpu().numpy())}\n"
                + f"\t mean hydrophobic: {np.mean(hydrophobic.detach().cpu().numpy())}\n"
                + f"\t mean hydrogen: {np.mean(hbonding.detach().cpu().numpy())}\n"
            )

        energy = (
            self.params["weights"]["w1"] * gauss_1
            + self.params["weights"]["w2"] * gauss_2
            + self.params["weights"]["w3"] * repulsion
            + self.params["weights"]["w4"] * hydrophobic
            + self.params["weights"]["w5"] * hbonding
        )
        return energy

    def _get_vina_feature_vectors(
        self, ligands_features: dict, ligands: list[Ligand] = None, inter: bool = True
    ) -> dict:
        """
        Generate feature vectors needed for Vina score
        """

        # get indices of contacts
        if inter:
            # define s1 and s2
            s2_hydro = self.rec_hydro
            s2_donor = self.rec_donor
            s2_acceptor = self.rec_acceptor
            s2_vdw = self.rec_vdw
            # distance calculation
            pl_dist = torch.cdist(
                ligands_features["coords"][None, :, :], self.rec_coords[None, :, :]
            )[0]
            dist_mask = pl_dist <= 8

            s1_inds, s2_inds = torch.where(dist_mask)
        else:
            # define s1 and s2
            s2_hydro = ligands_features["hydrophobic"]
            s2_donor = ligands_features["donor"]
            s2_acceptor = ligands_features["acceptor"]
            s2_vdw = ligands_features["vdw"]
            # topological distance
            max_ats = int(torch.max(ligands_features["natoms"]).item())
            topo_dist = torch.zeros((len(ligands), max_ats, max_ats), device=device)
            movable_matrix = torch.zeros_like(topo_dist)
            for i, lig in enumerate(ligands):
                mol = Chem.RemoveHs(lig.mol())
                n_ats = ligands_features["movable"][i].shape[0]
                topo_dist_i = torch.tensor(
                    Chem.rdmolops.GetDistanceMatrix(mol)
                )
                topo_dist[i][:n_ats, :n_ats] = topo_dist_i.triu()[:n_ats, :n_ats]
                movable_matrix[i][:n_ats, :n_ats] = ligands_features[
                    "movable"
                ][i]
            # regular disance
            masked_coords = torch.zeros([len(ligands) * max_ats, 3], device=device)
            new_indeces = (
                torch.cat(
                    [torch.arange(natoms, device=device) for natoms in ligands_features["natoms"]]
                )
                + ligands_features["inds"] * max_ats
            )
            masked_coords.index_copy_(
                dim=0, index=new_indeces, source=ligands_features["coords"]
            )
            masked_coords = masked_coords.reshape(-1, max_ats, 3)

            pl_dist = torch.cdist(masked_coords, masked_coords)
            dist_mask = torch.logical_and(
                torch.logical_and(topo_dist > 3, pl_dist <= 8), movable_matrix
            )
            natom_ind, s1_inds, s2_inds = torch.where(dist_mask)
            offset = torch.index_select(
                torch.cat(
                    (
                        torch.tensor([0], device=device),
                        torch.cumsum(ligands_features["natoms"], dim=0)[:-1],
                    )
                ),
                0,
                natom_ind,
            )

            s1_inds += offset
            s2_inds += offset

        vina_features = {}
        vina_features["segment_inds"] = torch.index_select(
            ligands_features["inds"], 0, s1_inds
        )
        vina_features["num_ligands"] = len(ligands_features["natoms"])
        # combine rec and lig vectors
        vina_features["hydrophobic"] = torch.index_select(
            ligands_features["hydrophobic"], 0, s1_inds
        ) * torch.index_select(s2_hydro, 0, s2_inds)
        vina_features["h-bonding"] = torch.index_select(
            ligands_features["donor"], 0, s1_inds
        ) * torch.index_select(s2_acceptor, 0, s2_inds) + torch.index_select(
            ligands_features["acceptor"], 0, s1_inds
        ) * torch.index_select(
            s2_donor, 0, s2_inds
        )
        vina_features["distance"] = (
            pl_dist[dist_mask]
            - torch.index_select(ligands_features["vdw"], 0, s1_inds)
            - torch.index_select(s2_vdw, 0, s2_inds)
        )
        return vina_features

    def _get_rec_vectors(self):
        """
        Generate feature vectors for receptor.
        """
        rec_ans_H, rec_coords_H, rec_atom_codes = self._parse_receptor()
        self.rec_coords = rec_coords_H[rec_ans_H != 1]
        # get atom codes and connectivity
        rec_connectivity = self._get_connectivity(rec_ans_H, rec_coords_H)
        assert len(rec_atom_codes) == self.rec_coords.shape[0]
        self.rec_hydro = self._get_hydro_vector(
            rec_ans_H, rec_connectivity, rec_atom_codes
        )
        rec_donor, rec_acceptor = self._get_hbond_vectors(
            rec_atom_codes, rec_ans_H, rec_connectivity
        )
        self.rec_donor = rec_donor
        self.rec_acceptor = rec_acceptor
        self.rec_vdw = self._get_vdw(rec_atom_codes)

    def _get_lig_vectors(self, ligands: list[Ligand]) -> dict:
        """
        Generate feature vectors for ligand.
        """
        # create hbond vectors
        (
            all_lig_coords,
            all_lig_hydro,
            all_lig_donor,
            all_lig_acceptor,
            all_atom_codes,
            all_lig_torsions,
            all_movable,
        ) = ([], [], [], [], [], [], [])
        for ligand in ligands:
            # ligand ans and coords
            lig_ans_H = ligand.ans(add_hydrogens=True)
            lig_coords_H = ligand.coords(add_hydrogens=True)
            lig_coords = lig_coords_H[lig_ans_H != 1]

            # ligand hbond vector
            atom_codes, movable_matrix = self._get_lig_pdbqt_info(
                rdkitmol_to_pdbqt_string(ligand.mol(add_hydrogens=True)),
                lig_coords,
            )
            assert len(atom_codes) == lig_coords.shape[0]
            torsions = self._get_num_torsions(ligand)

            # ligand hydrophobic vector
            lig_connectivity = self._get_connectivity(lig_ans_H, lig_coords_H)
            lig_hydro = self._get_hydro_vector(lig_ans_H, lig_connectivity, atom_codes)

            lig_donor, lig_acceptor = self._get_hbond_vectors(
                atom_codes, lig_ans_H, lig_connectivity
            )
            all_lig_coords.append(lig_coords)
            all_lig_hydro.append(lig_hydro)
            all_lig_donor.append(lig_donor)
            all_lig_acceptor.append(lig_acceptor)
            all_lig_torsions.append(torsions)
            all_movable.append(movable_matrix)
            all_atom_codes += atom_codes

        ligands_features = {
            "coords": torch.cat(all_lig_coords),
            "donor": torch.cat(all_lig_donor),
            "acceptor": torch.cat(all_lig_acceptor),
            "hydrophobic": torch.cat(all_lig_hydro),
            "vdw": self._get_vdw(all_atom_codes),
            "movable": all_movable,
            "natoms": torch.tensor([coords.shape[0] for coords in all_lig_coords], device=device),
            "torsions": torch.tensor(all_lig_torsions, device=device),
        }
        ligands_features["inds"] = torch.repeat_interleave(
            torch.arange(len(ligands), device=device), ligands_features["natoms"]
        )

        return ligands_features

    def _parse_receptor(
        self,
    ) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Parse the recpetor, get its atom types and coordinates
        """
        rec = mda.Universe(self.rec_pdbqt_file)
        ans_H = []
        coords_H = []
        atom_codes = []
        for res in rec.residues:
            for atom in res.atoms:
                if atom.name[0].isnumeric():
                    ans_H.append(1)
                else:
                    ans_H.append(PERIODIC_TABLE.index(atom.name[0]))
                coords_H.append(atom.position)
                if atom.type not in ['H', 'HD']:
                    atom_codes.append(atom.type)
        return torch.from_numpy(np.array(ans_H)).to(device), torch.from_numpy(np.array(coords_H)).to(device), atom_codes

    def _get_connectivity(
        self, ans: torch.tensor, coords: torch.tensor
    ) -> torch.tensor:
        """
        Connectivity matrix for atoms defined by ans and coords
        """
        dist = torch.cdist(coords[None, :, :], coords[None, :, :])[0]
        ans_covalent_radii = torch.tensor(
            [covalent_radii[PERIODIC_TABLE[at]] for at in ans], device=device
        )
        covalent_cutoffs = ans_covalent_radii[:, None].expand(
            -1, len(ans_covalent_radii)
        )
        covalent_cutoffs = covalent_cutoffs + torch.transpose(covalent_cutoffs, 1, 0)
        connect = dist <= covalent_cutoffs
        connect.fill_diagonal_(0)
        return connect

    def _get_hbond_vectors(
        self, atom_codes: list[str], ans: torch.tensor, connectivity: torch.tensor
    ) -> torch.tensor:
        """
        Get binary vector encoding which atoms are hbond donors/acceptors
        """
        connectivity_N_O = (
            connectivity
            * torch.isin(ans, torch.tensor([7, 8], device=device))[:, None].expand(
                -1, len(ans)
            )
            * ans[None, :].expand(len(ans), -1)
        )
        donor_vect = (
            torch.sum(connectivity_N_O == 1, dim=-1) > 0
        )
        donor_vect = torch.masked_select(donor_vect, torch.ne(ans, 1))

        acceptor_vect = torch.tensor(
            [
                True if code[:2] == "NA" or code[:2] == "OA" else False
                for code in atom_codes
            ],
            device=device,
        )

        return (donor_vect, acceptor_vect)

    def _get_hydro_vector(
        self,
        ans: torch.tensor,
        connectivity: torch.tensor,
        atom_types: list[str] = None,
    ) -> torch.tensor:
        """
        Get binary vector encoding which atoms are hydrophobic
        """
        hydro_vect_no_C = torch.isin(
            ans, torch.tensor([9, 17, 35, 53], device=device)
        )  # if F, Cl, Br, I, then hydrophobic

        connectivity_C_ans = (
            connectivity
            * torch.eq(ans, 6)[:, None].expand(-1, len(ans))
            * ans[None, :].expand(len(ans), -1)
        )
        not_hydro_C = (
            torch.sum(
                torch.logical_not(
                    torch.isin(
                        connectivity_C_ans, torch.tensor([0, 1, 6], device=device)
                    )
                ),
                axis=-1,
            )
            > 0
        )
        hydro_C = torch.logical_and(torch.eq(ans, 6), torch.logical_not(not_hydro_C))
        hydro_atoms = torch.masked_select(
            torch.logical_or(hydro_vect_no_C, hydro_C), torch.ne(ans, 1)
        )
        # in vinardo, all aromatic carbons and sulfur acceptors are hydrophobic
        if atom_types is not None and self.vinardo:
            aromatic_Cs = torch.tensor(
                [atom_type in ["A", "SA"] for atom_type in atom_types]
            , device=device)
            hydro_atoms = torch.logical_or(hydro_atoms, aromatic_Cs)

        return hydro_atoms

    def _get_lig_pdbqt_info(
        self, pdbqt_string: str, lig_coords: torch.tensor
    ) -> tuple[list[str], torch.tensor]:
        """
        Get information for ligand from pdbqt data. Specifcally, AD4 atom codes and movable atoms.
        """
        atom_codes, coords = [], []
        pdbqt_ind_to_atom_ind = {}

        # first get atom codes
        atom_count = 0
        for line in pdbqt_string.split("\n"):
            if len(line) > 4:
                if line[:4] == "ATOM":
                    line_split = line.split()
                    if line_split[-1] not in ["H", "HD"]:
                        atom_codes.append(line_split[-1])
                        pdbqt_ind_to_atom_ind[int(line_split[1])] = atom_count
                        coords.append(
                            [
                                float(line[30:38].strip()),
                                float(line[38:46].strip()),
                                float(line[46:54].strip()),
                            ]
                        )
                        atom_count += 1

        moveable_matrix = torch.zeros(
            (len(lig_coords), len(lig_coords)), dtype=torch.bool, device=device
        )

        lines = pdbqt_string.split("\n")
        atom_ind = -1
        for i in range(len(lines)):
            if lines[i][:4] == "ATOM" and lines[i][77:79].strip() not in ["H", "HD"]:
                atom_ind += 1
            if lines[i][:6] == "BRANCH":
                _, branch_start, branch_end = lines[i].split()
                atom_ind_j = int(atom_ind)
                branch_inds = []
                j = i + 1
                while True:
                    curr_line = lines[j].split()
                    if (
                        curr_line[0] == "ENDBRANCH"
                        and curr_line[1] == branch_start
                        and curr_line[2] == branch_end
                    ):
                        break
                    elif curr_line[0] == "ATOM" and curr_line[-1] not in ["H", "HD"]:
                        atom_ind_j += 1
                        branch_inds.append(atom_ind_j)
                    j += 1

                if len(branch_inds) > 1:
                    branch_inds = torch.tensor(branch_inds, dtype=torch.int64, device=device)

                    branch_mask = torch.zeros(len(lig_coords), dtype=torch.bool, device=device)
                    branch_mask[branch_inds] = 1
                    branch_mask[pdbqt_ind_to_atom_ind[int(branch_end)]] = 0

                    not_branch_mask = torch.ones(len(lig_coords), dtype=torch.bool, device=device)
                    not_branch_mask[branch_inds] = 0
                    not_branch_mask[pdbqt_ind_to_atom_ind[int(branch_start)]] = 0

                    moveable_matrix[branch_mask] = torch.logical_or(
                        moveable_matrix[branch_mask], not_branch_mask
                    )
                    moveable_matrix[not_branch_mask] = torch.logical_or(
                        moveable_matrix[not_branch_mask], branch_mask
                    )

        coords = torch.tensor(coords, device=device)
        pdbqt_to_sdf_map = torch.argmin(
            torch.cdist(lig_coords[None, :, :], coords[None, :, :])[0], dim=-1
        )
        moveable_matrix = moveable_matrix[pdbqt_to_sdf_map]
        moveable_matrix = moveable_matrix[:, pdbqt_to_sdf_map]
        return [atom_codes[i] for i in pdbqt_to_sdf_map], moveable_matrix

    def _get_num_torsions(self, ligand: Ligand) -> int:
        """
        Get number of torsions in ligand
        """
        return Lipinski.NumRotatableBonds(ligand.mol())

    def _get_vdw(self, atom_codes: list[str]) -> torch.tensor:
        """
        Get vdw radii for atoms based on atom codes
        """
        return torch.tensor(
            [self.params["vdw_radii"][atom_code[0]] for atom_code in atom_codes],
            device=device,
        )

    def _calc_gauss(self, features: dict, o: float, s: float) -> torch.tensor:
        """
        Calculate Gauss
        """
        guass_ij = torch.exp(-torch.pow((features["distance"] - o) / s, 2))
        return segment_coo(
            guass_ij,
            features["segment_inds"],
            torch.zeros(features["num_ligands"], dtype=torch.float32, device=device),
        )

    def _calc_repulsion(self, features: dict) -> torch.tensor:
        """
        Calculate repulsion
        """
        d_ij = features["distance"]
        return segment_coo(
            torch.pow(((d_ij < 0) * d_ij), 2),
            features["segment_inds"],
            torch.zeros(features["num_ligands"], dtype=torch.float32, device=device),
        )

    def _calc_hydrophobic(self, features: dict) -> torch.tensor:
        """
        Calculate hydrophobic interactions contribution to score
        """
        d_ij = features["distance"]
        hydro_1 = features["hydrophobic"] * (d_ij <= self.params["cutoffs"]["p1"])
        hydro_2_cond = (
            features["hydrophobic"]
            * (d_ij > self.params["cutoffs"]["p1"])
            * (d_ij < self.params["cutoffs"]["p2"])
        )
        hydro_2 = (
            hydro_2_cond
            * (self.params["cutoffs"]["p2"] - d_ij)
            / (self.params["cutoffs"]["p2"] - self.params["cutoffs"]["p1"])
        )
        return segment_coo(
            hydro_1 + hydro_2,
            features["segment_inds"],
            torch.zeros(features["num_ligands"], dtype=torch.float32, device=device),
        )

    def _calc_hbonding(self, features: dict) -> torch.tensor:
        """
        Calculate h-bonding interractions contribution to score
        """
        d_ij = features["distance"]
        hbond_1 = features["h-bonding"] * (d_ij <= self.params["cutoffs"]["h1"])
        hbond_2 = (
            features["h-bonding"]
            * (d_ij < 0)
            * (d_ij > self.params["cutoffs"]["h1"])
            * (d_ij)
            / self.params["cutoffs"]["h1"]
        )

        return segment_coo(
            hbond_1 + hbond_2,
            features["segment_inds"],
            torch.zeros(features["num_ligands"], dtype=torch.float32, device=device),
        )

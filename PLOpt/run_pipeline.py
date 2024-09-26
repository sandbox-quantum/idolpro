from argparse import ArgumentParser
import os
import sys
import time
from pathlib import Path
import copy
import math
import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs

from DiffSBDD.analysis.metrics import MoleculeProperties, rdmol_to_smiles
from PLOpt.ligand_generator import LigandGenerator
from PLOpt.ligand import Ligand
from PLOpt.protein import Protein
from PLOpt.plopt_utils import (
    eval_qvina,
    write_sdf_file,
    evaluate_sa,
    get_pretty_time,
    batchify
)
from PLOpt.ani import ANIScore
from PLOpt.torchvina import TorchVinaScore
from PLOpt.torchsa import TorchSAScore
from PLOpt.bonded import BondedScore
from PLOpt.validate_molecules import MoleculeValidater

cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PipeLine(object):
    """
    Object to run entire pipeline. Uses a set of differentiable scores to guide
    generation of ligands by modifying latent vectors of generative model via
    automatic differentiation.
    """

    def __init__(self, args: list[str]):
        self._parse_args(args)
        self._initialize()

    def _parse_args(self, args: list[str]):
        """
        Parse arguments required to run pipeline
        """
        parser = ArgumentParser()
        # i/o
        parser.add_argument("--protein", "-p", dest="protein_file", type=str, default=None, required=True)
        parser.add_argument("--ligand", "-l", dest="ligand_file", type=str, default=None, required=True)
        parser.add_argument("--num-generate", "-ng", type=int, default=100)
        parser.add_argument("--out-dir", "-o", type=str, default=None)
        parser.add_argument(
            "--overwrite",
            "-ow",
            action="store_true",
            help="whether to overwrite an existing results directory",
        )
        parser.add_argument("--verbose", "-v", type=int, default=2, choices=[0, 1, 2, 3])
        # latent optimization
        parser.add_argument("--batch-size", "-bs", type=int, default=None)
        parser.add_argument(
            "--scores",
            type=str,
            nargs="+",
            default=["torchvina", "torchsa"],
            choices=["ani", "torchvina", "torchvinardo", "torchsa"],
        )
        parser.add_argument(
            "--weights",
            type=int,
            nargs="+",
            default=None,
        )
        parser.add_argument(
            "--geo-opt-scores",
            type=str,
            nargs="+",
            default=["torchvina", "ani"],
            choices=["ani", "torchvina", "torchvinardo"],
        )
        parser.add_argument("--lig-fixed", "-lg",
                        nargs='+',
                        help="List of fixed atom indices, or True to fix Murcko Scaffold",
                        default=None)
        parser.add_argument("--learning-rate", "-lr", type=float, default=1e-1)
        parser.add_argument("--max-traj-length", "-maxt", type=int, default=200)
        parser.add_argument("--min-traj-length", "-mint", type=int, default=10)
        parser.add_argument("--max_attempts", "-ma", type=int, default=10)
        # generator parameters
        parser.add_argument(
            "--horizon",
            "-hz",
            type=int,
            default=20,
            help="optimization horizon (which step in generative diffusion to change latent parameters).",
        )
        parser.add_argument("--diffusion-steps", "-ds", type=int, default=50)
        parser.add_argument(
            "--num-lig-atoms",
            "-nl",
            type=int,
            default=None,
            help="Number of atoms in each ligand (if None samples randomly).",
        )
        parser.add_argument(
            "--num-atom-bias",
            "-nb",
            type=int,
            default=0,
            help="How much to bias each ligand size sample by.",
        )
        # geom opt parameters
        parser.add_argument(
            "--post-process",
            "-pp",
            type=str,
            default="geom-opt",
            choices=["geom-opt", "qvina", "none"],
        )

        parsed_args = parser.parse_args(args)
        self.__dict__.update(vars(parsed_args))
        if self.batch_size is None: # set batch size automatically
            self.batch_size = math.ceil(torch.cuda.get_device_properties(0).total_memory / 10 ** 9)

    def _make_results_dir(self):
        """
        Make directory where results will be written
        """
        if self.out_dir is None:
            self.out_dir = "results"

        self.results_dir = Path(f"{self.out_dir}/{self.pdb_id}_{self.lig_id}")
        print(f"Will write results into {self.results_dir}")
        # make all output directories
        os.makedirs(self.results_dir, exist_ok=True)

    def _initialize(self):
        """
        Initialize the pipeline
        1. Make results directory
        2. Stop run if already completed, or if ligand incompatible with ANI
        3. Load generative model and scoring models
        """
        if not os.path.isabs(self.protein_file):
            self.protein_file = f"{cwd}/{self.protein_file}"

        if not os.path.isabs(self.ligand_file):
            self.ligand_file = f"{cwd}/{self.ligand_file}"

        self.pdb_id = os.path.basename(self.protein_file)[:-4]
        self.lig_id = os.path.basename(self.ligand_file)[:-4]

        self._make_results_dir()

        supported_atom_types = [6, 7, 8, 9, 16, 17]
        if "ani" not in self.scores and "ani" not in self.geo_opt_scores:
            supported_atom_types += [15, 35, 53]

        self.protein = Protein(self.protein_file, self.ligand_file)

        if os.path.exists(Path(self.results_dir, "data.pt")) and not self.overwrite:
            raise ValueError(
                f"Run for {self.pdb_id}_{self.lig_id} already completed. If you want to overwrite this directory, call script with --overwrite"
            )

        # Identify atoms to fix, either by identifying a scaffold or using input list
        if self.lig_fixed:
            if self.lig_fixed[0] == 'True':
                scaffold = self.protein.get_reference_scaffold()
                if scaffold:
                    self.lig_fixed = scaffold
                else:
                    raise ValueError(
                        f"Cannot find scaffold using RDKit for {self.pdb_id}_{self.lig_id}"
                    )
            else:
                self.lig_fixed = list(map(int, self.lig_fixed))

        # If optimizing reference or fixing atoms, check if they are compatiable with ANI2x
        if self.lig_fixed is not None:
            reference_ligand = self.protein.get_reference_ligand()
            reference_atoms = reference_ligand.ans()  # Assuming `ans()` returns atoms or similar
            if self.lig_fixed:
                can_run = all(a in supported_atom_types for a in reference_atoms[self.lig_fixed])
            else:
                can_run = all(a in supported_atom_types for a in reference_atoms)
            if not can_run:
                raise ValueError(
                    f"Cannot run reference optimization on {self.pdb_id}_{self.lig_id} because has atoms incompatible with ani2x"
                )

        self.molecule_validater = MoleculeValidater(self.protein, supported_atom_types)
        # load necessary models
        if "ani" in self.scores or "ani" in self.geo_opt_scores:
            # load energy model
            self.ani_scorer = ANIScore(self.protein)

        if "torchvina" in self.scores or "torchvina" in self.geo_opt_scores:
            # load VINA scorer
            self.vina_scorer = TorchVinaScore(self.protein)

        if "torchvinardo" in self.scores or "torchvinardo" in self.geo_opt_scores:
            self.vinardo_scorer = TorchVinaScore(self.protein, vinardo=True)

        if "torchsa" in self.scores:
            self.synth_scorer = TorchSAScore()

        # load generative model (DiffSBDD)
        self.gen_model = LigandGenerator(
            protein = self.protein,
            batch_size = self.batch_size,
            molecule_validater = self.molecule_validater,
            num_lig_atoms = self.num_lig_atoms,
            num_atom_bias = self.num_atom_bias,
            lig_fixed = self.lig_fixed,
            timesteps = self.diffusion_steps,
            horizon = self.horizon
        )

        # set weights on scores for latent and geometry optimization
        self.latent_ws = {
            'ani': 1,
            'torchvina': 1,
            'torchvinardo': 1,
            'torchsa': 1,
        }
        if self.weights is not None:
            for score, weight in zip(self.scores, self.weights):
                self.latent_ws[score] = weight
        for score in self.scores:
            print(f"Optimizing {score} with a weight of {self.latent_ws[score]}")

        self.geo_ws = {
            'ani': 1,
            'torchvina': 1,
            'torchvinardo': 1,
            'bonded': 0.01,
        }

        # initialize ligands
        self.all_ligands = []
        self.ligands = [Ligand(scores=self.scores) for _ in range(self.batch_size)]

        # global optimization scores
        self.start_time = time.time()

        # initialize individual stats
        self.time_stamps = []

    def _setup(self):
        """
        Set initialize latent vectors and pass to optimizer
        """
        # initial steps before optimization
        with torch.no_grad():
            if self.verbose > 1:
                print("\tSampling new initial latents")
                latents_lig = self.gen_model.initialize_latents()

        self.latents_lig = latents_lig

        # set up resampling parameters
        self.valid_mask = torch.zeros(self.batch_size, dtype=torch.bool)
        self.attempts = torch.zeros(self.batch_size, device=device)
        self.latents_counter = torch.zeros(
            self.batch_size, dtype=torch.int64, device=device
        )
        self.latents_cache = [latents_lig]

        # load optimizer
        self.optimizer = torch.optim.Adam(
            [
                {"params": latent_lig, "lr": self.learning_rate, "name": f"latents_{i}"}
                for i, latent_lig in enumerate(latents_lig)
            ],
            betas=(0.5, 0.999),
        )

    def _get_valid_ligands(self):
        """
        Get valid ligands from current ligands
        """
        valid_ligands = []
        for i, valid_flag in enumerate(self.valid_mask):
            if valid_flag:
                valid_ligands.append(self.ligands[i])
        return valid_ligands

    def _closure_geo_opt(self):
        """
        Objective for geometry optimization
        """
        torch.cuda.empty_cache()
        self.geo_optimizer.zero_grad()
        loss = 0

        for ligand in self.geo_opt_ligands:
            ligand.update_coords(ligand.coords())

        if "ani" in self.geo_opt_scores:
            energies = self.ani_scorer.score(self.geo_opt_ligands, intra_only=True)
            loss += self.geo_ws["ani"] * torch.mean(energies)

        if "torchvina" in self.geo_opt_scores:
            vina_scores = self.vina_scorer.score(self.geo_opt_ligands, intra=False)
            loss += self.geo_ws["torchvina"] * torch.mean(vina_scores)

        if "torchvinardo" in self.geo_opt_scores:
            vinardo_scores = self.vinardo_scorer.score(self.geo_opt_ligands, intra=False)
            loss += self.geo_ws["torchvinardo"] * torch.mean(vinardo_scores)

        bonded_scores = BondedScore.score(self.geo_opt_ligands)
        loss += self.geo_ws["bonded"] * torch.mean(bonded_scores)

        loss.backward()

        return loss

    def _closure_latents(self):
        """
        Objective for latent optimization
        """
        self.optimizer.zero_grad()
        # generate ligands from horizon
        self.ligands, self.valid_mask, _ = (
            self.gen_model.generate_ligands(self.ligands, torch.cat(self.latents_lig))
        )
        torch.cuda.empty_cache()

        # check validity of generated ligands
        self.attempts += torch.logical_not(self.valid_mask.type(torch.int64))
        if not torch.any(self.valid_mask):  # no valid ligands generated...
            return torch.zeros(1, requires_grad=True, device=device)

        # perform inference with docknscore (either just confidence, or also docking)
        valid_ligands = self._get_valid_ligands()

        loss = 0
        scores = {}

        if "ani" in self.scores:
            energies = self.ani_scorer.score(valid_ligands)
            loss += self.latent_ws["ani"] * torch.mean(energies)
            scores["ani"] = energies

        if "torchvina" in self.scores:
            vina_scores = self.vina_scorer.score(valid_ligands)
            loss += self.latent_ws["torchvina"] * torch.mean(vina_scores)
            scores["torchvina"] = vina_scores

        if "torchvinardo" in self.scores:
            vinardo_scores = self.vinardo_scorer.score(valid_ligands)
            loss += self.latent_ws["torchvinardo"] * torch.mean(vinardo_scores)
            scores["torchvinardo"] = vinardo_scores

        if "torchsa" in self.scores:
            sa_scores = self.synth_scorer.score(valid_ligands)
            loss += self.latent_ws["torchsa"] * torch.mean(sa_scores)
            scores["torchsa"] = sa_scores

        for i, ligand in enumerate(valid_ligands):
            ligand.append_to_trajectory(
                dict([(key, scores[key][i].item()) for key in scores])
            )

        loss.backward()
        if self.lig_fixed:
            self._apply_mask_to_latents()

        return loss

    def _apply_mask_to_latents(self):
        for idx, latent_lig in enumerate(self.latents_lig):
            mask = torch.ones_like(latent_lig)
            mask[:len(self.lig_fixed)] = 0
            with torch.no_grad():
                self.latents_lig[idx].grad *= mask

    @torch.no_grad()
    def _resample_latents(self):
        """
        Resample or reduce learning rate for latent codes for
        ligands that have failed self.max_attempts times
        """
        # create mask for latents
        invalid_mask = self.attempts == self.max_attempts

        param_lrs_dict = dict(
            [
                (param_group["name"], param_group["lr"])
                for param_group in self.optimizer.param_groups
            ]
        )
        param_lrs = [param_lrs_dict[f"latents_{idx}"] for idx in range(self.batch_size)]
        lower_lr_mask = torch.logical_and(
            torch.tensor(
                [param_lr != 1e-3 for param_lr in param_lrs],
                dtype=torch.bool,
                device=device,
            ),
            torch.tensor(
                [len(ligand.latents) > 0 for ligand in self.ligands],
                dtype=torch.bool,
                device=device,
            ),
        )

        lower_lr_mask = torch.logical_and(invalid_mask, lower_lr_mask)

        max_iter_mask = torch.tensor(
            [
                len(ligand.get_trajectory()) >= self.max_traj_length
                for ligand in self.ligands
            ],
            dtype=torch.bool,
            device=device,
        )

        resample_mask = torch.logical_or(
            torch.logical_and(invalid_mask, torch.logical_not(lower_lr_mask)),
            max_iter_mask,
        )

        if torch.any(torch.logical_or(lower_lr_mask, resample_mask)):
            self.latents_counter += resample_mask.type(torch.int64)
            self.attempts = self.attempts.masked_fill(
                torch.logical_or(lower_lr_mask, resample_mask), 0
            )
            if torch.any(
                torch.masked_select(
                    self.latents_counter == len(self.latents_cache), resample_mask
                )
            ):
                if self.verbose > 1:
                    print("\tSampling new initial latents")
                    latents_lig_sample = self.gen_model.initialize_latents()
                self.latents_cache.append(latents_lig_sample)

            for idx in torch.nonzero(torch.logical_or(lower_lr_mask, resample_mask)):
                idx = idx.item()
                param_names = [
                    param_group["name"] for param_group in self.optimizer.param_groups
                ]
                param_idx = param_names.index(f"latents_{idx}")
                if resample_mask[idx]:
                    self._store_trajectory(idx)
                    self.latents_lig[idx] = self.latents_cache[
                        self.latents_counter[idx]
                    ][idx]
                    new_lr = self.learning_rate
                else:
                    self.latents_lig[idx] = (
                        self.ligands[idx].latents[-1].requires_grad_()
                    )
                    new_lr = self.optimizer.param_groups[param_idx]["lr"] / 10
                del self.optimizer.param_groups[param_idx]
                self.optimizer.add_param_group(
                    {
                        "params": self.latents_lig[idx],
                        "lr": new_lr,
                        "name": f"latents_{idx}",
                    }
                )

    def _optimize(self):
        """
        Optimization loop for latent optimization
        """
        self._setup()
        curr_iter = 0

        while len(self.all_ligands) < self.num_generate:
            self.optimizer.step(self._closure_latents)
            torch.cuda.empty_cache()
            valid_ligands = self._get_valid_ligands()

            total_time = get_pretty_time(self.start_time)
            print_msg = f"{total_time} Iteration {curr_iter:>4}\n"

            if len(valid_ligands) > 0:
                print_msg += f"\tNumber of valid ligands: {len(valid_ligands)}\n"
                for score in self.scores:
                    mean_score = np.nanmean([ligand.get_score(score) for ligand in self.ligands])
                    min_score = np.nanmin([ligand.get_score(score) for ligand in self.ligands])
                    print_msg += f"\tMean {score} : {mean_score:2.4f}, Min {score} : {min_score:2.4f}\n"
            else:
                print_msg += f"\tNo valid ligands generated...\n"
            # print stats
            if self.verbose > 1:
                print(print_msg)

            curr_iter += 1
            self._resample_latents()
            torch.cuda.empty_cache()

    def _store_trajectory(self, idx: int):
        """
        Store trajectory of a ligand
        """
        trajectory = self.ligands[idx].get_trajectory()

        if len(trajectory) >= self.min_traj_length:
            if self.verbose > 0:
                print(f"{get_pretty_time(self.start_time)}: saving trajectory number {len(self.all_ligands)}")

            self.ligands[idx].atomic_probs = None
            self.ligands[idx].reset_to(strategy="best")
            self.all_ligands.append(copy.deepcopy(self.ligands[idx]))
            self.time_stamps.append(time.time() - self.start_time)

        self.ligands[idx] = Ligand(scores=self.scores)

    def _post_process(self):
        """
        Post process generated ligands according to method, involves
        both structural refinement and metric calculation.
        """
        pdbqt_file = self.protein.get_pdbqt_file()
        main_metrics = ["vina", "smiles", "sa"]
        end_of_run_metrics = ["qed", "logp", "lipinski"]
        dict_stats = dict(
            [("rank", [])]
            + [("tanimoto_ref", [])]
            + [("tanimoto_traj", [])]
            + [("index", [])]
            + [("traj_length", [])]
            + [("time", [])]
            + [("vina_raw", [])]
            + [("name", [])]
            + [
                (metric, [])
                for metric in main_metrics + end_of_run_metrics
            ]
            + [
                (f"{metric}_0", [])
                for metric in main_metrics + end_of_run_metrics
            ]
        )

        # geometry optimization method
        if self.post_process == "geom-opt":
            opt_ligands = []
            for ligand in self.all_ligands:
                mol_hs = ligand.mol(add_hydrogens=True)
                opt_ligand = Ligand(mol_hs, protonated=True)
                opt_ligand.update_coords(opt_ligand.coords().clone().detach().requires_grad_())
                opt_ligand.set_connectivity_and_cutoffs()
                opt_ligands.append(opt_ligand)

            for batch in batchify(opt_ligands, n=120):
                self.geo_opt_ligands = batch
                self.geo_optimizer = torch.optim.LBFGS(
                    [elem.coordinates for elem in self.geo_opt_ligands],
                    max_iter=100,
                    line_search_fn="strong_wolfe",
                    tolerance_grad=1e-3
                )
                self.geo_optimizer.step(self._closure_geo_opt)

            for ligand, opt_ligand in zip(self.all_ligands, opt_ligands):
                opt_coords = opt_ligand.coords()
                opt_ans = opt_ligand.ans()
                ligand.update_coords(opt_coords[opt_ans != 1])

            valid_mask = torch.tensor(
                [self.molecule_validater.validate_molecule(ligand) for ligand in self.all_ligands],
                device=device,
                dtype=torch.bool)
        else:  # all ligands are valid
            valid_mask = torch.ones(len(self.all_ligands), dtype=torch.bool)

        valid_indices = torch.where(valid_mask)[0]
        self.all_ligands = [self.all_ligands[i] for i in valid_indices]
        # write out structs based on rank
        mp = MoleculeProperties()
        mp_outs = mp.evaluate(
            [
                [ligand.mol() for ligand in self.all_ligands],
                [ligand.trajectory[0] for ligand in self.all_ligands],
            ]
        )
        per_pocket_diversity = mp_outs[-1]
        mp_dict = dict(
            [
                (metric_name, metric)
                for metric_name, metric in zip(end_of_run_metrics, mp_outs[:-1])
            ]
        )

        print()
        # master sdf file of post-processed generated ligands
        master_ligand_list = []
        for i, (ind, ligand) in enumerate(zip(valid_indices, self.all_ligands)):
            ind = ind.item()
            dict_stats["index"].append(ind)
            # 1. evaluate final vina score
            if self.post_process == "qvina":
                eval_qvina(pdbqt_file, ligand, local=True)
            vina_processed = eval_qvina(pdbqt_file, ligand)
            dict_stats["vina"].append(vina_processed)
            ligand_name = f"idolpro-{self.pdb_id}-{self.lig_id}-{ind}"
            ligand.mol().SetProp("Name", ligand_name)
            dict_stats["name"].append(ligand_name)
            master_ligand_list.append(ligand.mol())
            # 2. write raw (unprocessed) and initial structure and evaluate vina
            vina_0 = eval_qvina(pdbqt_file, Ligand(ligand.trajectory[0]))
            dict_stats["vina_0"].append(vina_0)
            vina_raw = eval_qvina(pdbqt_file, Ligand(ligand.trajectory[-1]))
            dict_stats["vina_raw"].append(vina_raw)
            # 3. write SMILES for first and final ligand
            dict_stats["smiles_0"].append(rdmol_to_smiles(ligand.trajectory[0]))
            dict_stats["smiles"].append(rdmol_to_smiles(ligand.mol()))
            dict_stats["sa_0"].append(evaluate_sa(ligand.trajectory[0]))
            dict_stats["sa"].append(evaluate_sa(ligand.mol()))
            # 4. add rest of metrics to dict
            dict_stats["tanimoto_traj"].append(
                DataStructs.TanimotoSimilarity(
                    Chem.RDKFingerprint(ligand.trajectory[0]),
                    Chem.RDKFingerprint(ligand.mol()),
                )
            )
            dict_stats["tanimoto_ref"].append(
                DataStructs.TanimotoSimilarity(
                    Chem.RDKFingerprint(self.protein.get_reference_ligand().mol()),
                    Chem.RDKFingerprint(ligand.mol()),
                )
            )
            dict_stats["traj_length"].append(len(ligand.trajectory) - 1)
            dict_stats["time"].append(self.time_stamps[ind])
            for metric in end_of_run_metrics:
                dict_stats[metric].append(mp_dict[metric][0][i])
                dict_stats[f"{metric}_0"].append(mp_dict[metric][1][i])

            print(f"\tLigand {i} - vina: {vina_processed:0.3f}")
            print(
                f'\tQED: {dict_stats["qed"][-1]:0.3f} SA: {dict_stats["sa"][-1]:0.3f} logp: {dict_stats["logp"][-1]:0.3f} lipinksi: {dict_stats["lipinski"][-1]:0.3f}'
            )
            print(
                f'\ttraj_length: {dict_stats["traj_length"][-1]:d} tanimoto: {dict_stats["tanimoto_ref"][-1]:0.3f} vina_0: {vina_0:0.3f} SA_0: {dict_stats["sa_0"][-1]:0.3f}\n'
            )
        master_sdf_file = os.path.join(self.results_dir, "generated_ligands.sdf")
        write_sdf_file(master_sdf_file, master_ligand_list)

        dict_stats["rank"] = np.argsort(np.argsort(dict_stats["vina"])).tolist()

        return dict_stats, per_pocket_diversity

    def _write_results(self):
        """
        Write results of optimization run
        """
        dict_stats, pocket_diversity = self._post_process()
        df_stats = pd.DataFrame(data=dict_stats, index=dict_stats["index"])
        df_stats = df_stats.drop("index", axis=1)
        df_stats.to_csv(os.path.join(self.results_dir, "stats.csv"))
        # finally, dump all stats into .pt file
        dict_stats["total_time"] = time.time() - self.start_time
        dict_stats["pocket_diversity"] = pocket_diversity[0]
        dict_stats["pocket_diversity_0"] = pocket_diversity[1]
        torch.save(dict_stats, Path(self.results_dir, "data.pt"))

    def run(self):
        self._optimize()
        self._write_results()


def main():
    pipeline = PipeLine(sys.argv[1:])
    pipeline.run()


if __name__ == "__main__":
    main()
import copy
import os

import torch
import torch.nn.functional as F
from torch.utils import checkpoint
from torch_scatter import scatter_mean

from DiffSBDD.lightning_modules import LigandPocketDDPM
from DiffSBDD.utils import num_nodes_to_batch_mask, batch_to_list
from DiffSBDD.constants import FLOAT_TYPE, INT_TYPE
from PLOpt.plopt_utils import PERIODIC_TABLE, make_mol_openbabel
from PLOpt.protein import Protein
from PLOpt.ligand import Ligand
from PLOpt.validate_molecules import MoleculeValidater

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LigandGenerator:
    """
    Model for generating ligands. Wrapper around DiffSBDD functionality
    Args:
        protein: the protein within which the ligand will be generated
        batch_size: how many ligands to generate per batch
        num_lig_atoms: can be used to specify how many heavy atoms ligand should have
        num_atom_bias: number of atoms to bias random ligand size sampling (i.e. if
            num_atom_bias = 5, will add 5 heavy atoms to size drawn from random sampling)
        timesteps: how many diffusion steps to discretize reverse diffusion into
        horizon: optimization horizon -- latent vectors from this time step will be
            optimizied.
        pocket_by_resi: whether the pocket should be defined by adjacent residues, if
            False, will be defined by adajcent atoms.
    """

    def __init__(
        self,
        protein: Protein,
        batch_size: int,
        molecule_validater: MoleculeValidater,
        num_lig_atoms: int = None,
        num_atom_bias: int = 5,
        lig_fixed: int = None,
        timesteps: int = 100,
        horizon: int = 10,
        pocket_by_resi: bool = True
    ):
        generator_checkpoint = os.path.join(
            cwd, "./DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
        )
        model = LigandPocketDDPM.load_from_checkpoint(
            generator_checkpoint, map_location=device
        )
        for param in model.parameters():
            param.requires_grad_(False)

        self.model = model
        self.timesteps = timesteps
        self.horizon = horizon
        self.device = self.model.device
        self.batch_size = batch_size
        self.z_pocket = None
        self.lig_fixed = lig_fixed
        self.molecule_validater = molecule_validater

        # create conv tensor (for some reason self.device is cpu in __init__...)
        conv_tensor = torch.zeros(
            len(self.model.dataset_info["atom_decoder"]), dtype=torch.int64, device=device
        )
        for i, atom in enumerate(self.model.dataset_info["atom_decoder"]):
            conv_tensor[i] = PERIODIC_TABLE.index(atom)
        self.conv_tensor = conv_tensor
        self._prepare_pocket_and_ligand(
            protein,
            batch_size,
            num_lig_atoms,
            num_atom_bias,
            lig_fixed,
            pocket_by_resi,
        )

    def _prepare_pocket_and_ligand(
        self,
        protein: Protein,
        batch_size: int,
        num_nodes_lig: int = None,
        num_nodes_bias: int = 0,
        lig_fixed: int = None,
        pocket_by_resi: bool = True,
    ):
        """
        Prepares the pocket and ligand for pocket-guided ligand generation
        """

        # Load PDB

        # define pocket with reference ligand
        pocket_atoms, pocket_coords = protein.get_pocket_ans_and_coords(by_resi=pocket_by_resi)
        conv_list = self.conv_tensor.tolist()
        pocket_types = torch.tensor(
            [conv_list.index(at.item()) for at in pocket_atoms],
            device=device,
        )

        pocket_one_hot = F.one_hot(
            pocket_types, num_classes=len(self.model.pocket_type_encoder)
        )

        pocket_size = torch.tensor(
            [len(pocket_coords)] * batch_size, device=self.device, dtype=INT_TYPE
        )
        pocket_mask = torch.repeat_interleave(
            torch.arange(batch_size, device=self.device, dtype=INT_TYPE),
            len(pocket_coords),
        )

        self.pocket = {
            "x": pocket_coords.repeat(batch_size, 1),
            "one_hot": pocket_one_hot.repeat(batch_size, 1),
            "size": pocket_size,
            "mask": pocket_mask,
        }

        # Create dummy ligands
        if lig_fixed is not None:

            ref_types = torch.tensor(
                [conv_list.index(at.item()) for at in protein.ref_ligand.ans()],
                device=device,
            )
            ref_one_hot = F.one_hot(
                ref_types, num_classes=len(self.model.dataset_info["atom_decoder"])
            )
            ref_coords = protein.ref_ligand.coords()

            # retrieve coordinates and one_hot_encodings of fixed atom froms reference
            x_fixed = ref_coords[lig_fixed]
            one_hot_fixed = ref_one_hot[lig_fixed]
            n_atoms_reference = len(ref_one_hot)
            n_fixed = len(lig_fixed)
            # If number of atoms in ligands are not specified, sample ligand size
            if num_nodes_lig is None:
                std = 0.5 * (n_atoms_reference - n_fixed)
                # When scaffold is large compared to reference, set std to user defined bias
                if std < num_nodes_bias:
                    std = num_nodes_bias
                # Generate normally distributed integers and ensure minimum value
                num_nodes_lig = torch.round(torch.normal(mean=n_atoms_reference, std=std, size=(batch_size,))).int()
                num_nodes_lig = torch.clamp(num_nodes_lig, min=n_fixed)
            else:
                num_nodes_lig = torch.tensor(
                    [num_nodes_lig for _ in range(batch_size)], device=self.device
            )
                num_nodes_lig = num_nodes_lig + num_nodes_bias

            # Create ligand mask
            lig_mask = num_nodes_to_batch_mask(
                len(num_nodes_lig), num_nodes_lig, self.model.device)

            # Create dummy ligand to fill with coordinates
            ligand = {
                'x': torch.zeros((len(lig_mask), self.model.x_dims),
                                device=self.model.device, dtype=FLOAT_TYPE),
                'one_hot': torch.zeros((len(lig_mask), self.model.atom_nf),
                                    device=self.device, dtype=FLOAT_TYPE),
                'size': num_nodes_lig,
                'mask': lig_mask
            }

            # create dummy mask for the fixed atoms in ligand, needed for inpainting
            lig_fixed_mask = torch.zeros_like(lig_mask)

            # fill ligand mask, and dummy ligand
            for i in range(batch_size):
                # select batch item
                sele = (lig_mask == i)

                # fill first n_fixed entries of dummy ligand in batch item with fixed atom coordinates
                x_new = ligand['x'][sele]
                x_new[:n_fixed] = x_fixed
                ligand['x'][sele] = x_new

                # fill first n_fixed entries of dummy ligand in batch item with fixed atom type
                h_new = ligand['one_hot'][sele]
                h_new[:n_fixed] = one_hot_fixed
                ligand['one_hot'][sele] = h_new

                # create mask for fixed atoms in batch item
                fixed_new = lig_fixed_mask[sele]
                fixed_new[:n_fixed] = 1
                lig_fixed_mask[sele] = fixed_new
            self.ligand = ligand
            self.lig_fixed = lig_fixed_mask

        else:
            if num_nodes_lig is None:
                num_nodes_lig = self.model.ddpm.size_distribution.sample_conditional(
                    n1=None, n2=self.pocket["size"]
                )
                num_nodes_lig = torch.clamp(num_nodes_lig, max=30)
            else:
                num_nodes_lig = torch.tensor(
                    [num_nodes_lig for _ in range(batch_size)], device=self.device
                )
            num_nodes_lig = num_nodes_lig + num_nodes_bias

            lig_mask = num_nodes_to_batch_mask(
                len(num_nodes_lig), num_nodes_lig, self.device
            )

            self.ligand = {
                "x": torch.zeros(
                    (len(lig_mask), self.model.x_dims), device=self.device, dtype=FLOAT_TYPE
                ),
                "one_hot": torch.zeros(
                    (len(lig_mask), self.model.atom_nf),
                    device=self.device,
                    dtype=FLOAT_TYPE,
                ),
                "size": num_nodes_lig,
                "mask": lig_mask,
            }

    def initialize_latents(self) -> list[torch.tensor]:
        """
        Used to initialize the latent vectors for optimization. This involves
        reverse diffusing them from t=T to t=self.horizon.
        Returns:
            list of latent vectors from timestep self.horizon
        """
        latents_lig = self.generate_ligands(
            [Ligand() for _ in range(self.batch_size)],
            None,
            (self.timesteps * (self.model.T // self.timesteps), # translate to DiffSBDD's timesteps
             self.horizon * (self.model.T // self.timesteps)))[-1]
        return latents_lig

    def generate_ligands(
        self, ligands, z_lig_0: list[torch.tensor] = None, t_interval: tuple[int, int] = None
    ) -> tuple[
        list[Ligand], torch.tensor
    ]:
        """
        Reverse diffuse latent vectors from t_interval[1] to t_interval[0]. If z_lig_0 is None, will be drawn
        from random Gaussian noise.
        Args:
            z_lig_0: initial latent vector (None if sampling from t=self.timesteps)
            t_interval: the interval for which to reverse diffuse the latent vector
        Returns:
            tuple of the form (ligands, validity mask, latents)
        """
        # if this isn't being called from initliaze_latents, this will be the case
        if t_interval is None:
            t_interval = (self.horizon, 0)

        pocket = copy.deepcopy(self.pocket)
        ligand = copy.deepcopy(self.ligand)
        z_lig = z_lig_0.clone() if z_lig_0 is not None else None
        calc_grad = z_lig_0 is not None

        # Pocket's center of mass
        pocket_com_before = scatter_mean(self.pocket["x"], self.pocket["mask"], dim=0)

        # If fixing atoms, use inpaint method
        if self.lig_fixed is not None:
            method = self.model.ddpm.inpaint
            custom_method = self.custom_inpaint(self.model.ddpm)
            inputs = (ligand, pocket, self.lig_fixed)
        else:
            method = self.model.ddpm.sample_given_pocket
            custom_method = self.custom_sample_given_pocket(self.model.ddpm)
            inputs = (pocket, ligand['mask'])

        if calc_grad:
            for t in reversed(range(t_interval[1] + 2, t_interval[0])):
                out = checkpoint.checkpoint(
                    custom_method, *inputs, self.timesteps, z_lig, (t, t - 1))

                xh_lig, xh_pocket, z_lig = out
                pocket_com_after = scatter_mean(
                    xh_pocket[:, : self.model.x_dims], pocket["mask"], dim=0
                )
                z_lig = z_lig + torch.cat(
                    (
                        (pocket_com_before - pocket_com_after)[ligand["mask"]],
                        torch.zeros((len(z_lig), self.model.atom_nf), device=device),
                    ),
                    dim=1,
                )
        xh_lig, xh_pocket, z_lig = method(*inputs,
                                            timesteps = max(self.timesteps, t_interval[0]),
                                            z_lig_0 = z_lig,
                                            t_interval = (t_interval[1] + 2, t_interval[1]) if calc_grad else t_interval)

        # Move generated molecule back to the original pocket position
        pocket_com_after = scatter_mean(
            xh_pocket[:, : self.model.x_dims], pocket["mask"], dim=0
        )
        xh_lig[:, : self.model.x_dims] += (pocket_com_before - pocket_com_after)[ligand["mask"]]
        z_lig = z_lig + torch.cat(
            (
                (pocket_com_before - pocket_com_after)[ligand["mask"]],
                torch.zeros((len(z_lig), self.model.atom_nf), device=self.device),
            ),
            dim=1,
        )

        # Build mol objects
        x = xh_lig[:, : self.model.x_dims]
        atom_probs = xh_lig[:, self.model.x_dims :]
        atom_type = atom_probs.argmax(1)

        z_lig_list = batch_to_list(z_lig, ligand["mask"])
        if not calc_grad:
            z_lig_list = [z_lig.detach().requires_grad_() for z_lig in z_lig_list]
            valid_mask = None
        else:
            ligands, valid_mask = self.make_ligands(
                batch_to_list(x, ligand["mask"]),
                batch_to_list(atom_type, ligand["mask"]),
                batch_to_list(atom_probs, ligand["mask"]),
                ligands
            )

        return ligands, valid_mask, z_lig_list

    def custom_sample_given_pocket(self, module):
        """
        Function to allow for gradient checkpointing when running reverse diffusion from scratch.
        """

        def custom_forward(*inputs):
            output = module.sample_given_pocket(
                *inputs[:2],
                timesteps=inputs[2],
                z_lig_0=inputs[3],
                t_interval=inputs[4]
            )
            return output

        return custom_forward


    def custom_inpaint(self, module):
        """
        Function to allow for gradient checkpointing when running reverse diffusion with fixed atoms.
        """
        def custom_forward(*inputs):
            output = module.inpaint(
                *inputs[:3],
                timesteps=inputs[3],
                z_lig_0=inputs[4],
                t_interval=inputs[5]
            )
            return output

        return custom_forward

    def make_ligands(self, coordinates, model_atomic_numbers, model_atomic_probabilities, ligands):
        """
        Build ligands out of outputs of DiffSBDD
        """
        valid_mask = torch.zeros(len(ligands), device=device, dtype=torch.bool)
        for i, ligand in enumerate(ligands):
            ligand_ans = self.conv_tensor[model_atomic_numbers[i]]
            ligand.set_ligand(make_mol_openbabel(coordinates[i], ligand_ans),
                              coordinates[i],
                              ligand_ans,
                              model_atomic_probabilities[i])

            if self.molecule_validater.validate_molecule(ligand):
                valid_mask[i] = True
            else:
                ligand.set_ligand()
                valid_mask[i] = False

        return ligands, valid_mask

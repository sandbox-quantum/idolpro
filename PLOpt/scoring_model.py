import os

import torch

from PLOpt.plopt_utils import Batcher, collate_fn_pytorch_geometric, OCPDataDummy
from ocpmodels.models import PaiNN

cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScoringModel:
    """
    Base Scoring Model which all scoring models should inherit from.
    """

    def __init__(self):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError


class PaiNNScoringModel(ScoringModel):
    """
    Scoring model using PaiNN architecture
    """

    def __init__(self, model_ckpt: str, params: dict = {}):
        self.model_ckpt = model_ckpt
        self._load_model(params)

    def _load_model(self, params: dict):
        """
        Load the PaiNN model.
        """
        default_params = {
            "num_atoms": None,
            "bond_feat_dim": None,
            "hidden_channels": 256,
            "num_layers": 4,
            "num_rbf": 64,
            "cutoff": 8.0,
            "max_neighbors": 30,
            "regress_forces": False,
            "use_pbc": False,
            "atom_encoding": "one-hot",
            "num_elements": 10,
        }
        default_params.update(params)

        model = PaiNN(**default_params)
        model.load_state_dict(torch.load(self.model_ckpt, map_location=device))
        model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        self.model = model
    
    def score(self, ans: torch.Tensor, coords: torch.Tensor):
        """
        Score ans and coords with PaiNN
        """
        data = Batcher(ans, coords)
        dataloader = torch.utils.data.DataLoader(
            data, batch_size=len(ans), collate_fn=collate_fn_pytorch_geometric
        )
        batch = next(iter(dataloader))
        pred = self.model(
            OCPDataDummy(
                batch["atomic_numbers"],
                batch["coordinates"],
                batch["indices"],
                batch["natoms"],
            )
        )
        return pred['energy']
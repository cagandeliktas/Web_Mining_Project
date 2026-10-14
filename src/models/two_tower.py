"""Two-tower rating model with a bias head.

In plain words: The neural network itself: one small network for users, one for products, and a
score made from how well their outputs match.

    y_hat = mu + b_user + b_item + <tower_u(x_user), tower_i(x_item)>

``mu`` is the train mean rating. The biases are looked up by id; slot 0 is the
cold slot (user/item unseen in train) and is pinned to zero via
``padding_idx``, so an unseen id contributes exactly 0 and the prediction
falls back to content features alone. No target rescaling and no cosine
squashing: the output lives on the 1-5 rating scale directly.
"""

import torch
from torch import nn


def _mlp(n_in: int, hidden: list[int], n_out: int, dropout: float) -> nn.Sequential:
    """Build a small neural network: linear layers with ReLU and dropout that turn n_in numbers into
    n_out numbers.
    """
    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    last = nn.Linear(prev, n_out)
    nn.init.normal_(last.weight, std=0.02)  # start with a near-zero interaction term
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class TwoTower(nn.Module):
    """Predicts a rating as: overall average + user offset + product offset + how well the user vector
    matches the product vector. Each vector comes from its own "tower" (a small network).
    """
    def __init__(
        self,
        n_user_features: int,
        n_item_features: int,
        n_users: int,
        n_items: int,
        mu: float,
        embed_dim: int = 32,
        hidden: tuple = (256, 128),
        dropout: float = 0.1,
        use_biases: bool = True,
    ):
        """Build the two towers and the offset lookup tables. n_users / n_items count the known ids
        plus slot 0 for "unknown"; mu is the training average rating.
        """
        super().__init__()
        self.init_args = dict(
            n_user_features=n_user_features, n_item_features=n_item_features,
            n_users=n_users, n_items=n_items, mu=float(mu), embed_dim=embed_dim,
            hidden=list(hidden), dropout=dropout, use_biases=use_biases,
        )
        self.mu = float(mu)
        self.use_biases = use_biases
        self.user_tower = _mlp(n_user_features, list(hidden), embed_dim, dropout)
        self.item_tower = _mlp(n_item_features, list(hidden), embed_dim, dropout)
        if use_biases:
            self.user_bias = nn.Embedding(n_users, 1, padding_idx=0)
            self.item_bias = nn.Embedding(n_items, 1, padding_idx=0)
            nn.init.zeros_(self.user_bias.weight)
            nn.init.zeros_(self.item_bias.weight)

    def biases(self, user_idx: torch.Tensor, item_idx: torch.Tensor):
        """Look up the learned offset of each user and product (always 0 for slot 0, i.e. unknown ids).
        """
        return self.user_bias(user_idx).squeeze(-1), self.item_bias(item_idx).squeeze(-1)

    def forward(self, x_user, x_item, user_idx, item_idx) -> torch.Tensor:
        """Predict ratings for a batch of (user, product) pairs."""
        score = (self.user_tower(x_user) * self.item_tower(x_item)).sum(dim=-1)
        out = self.mu + score
        if self.use_biases:
            b_u, b_i = self.biases(user_idx, item_idx)
            out = out + b_u + b_i
        return out


def save_model(model: TwoTower, path) -> None:
    """Save the model's settings and learned weights to one file."""
    torch.save({"init_args": model.init_args, "state_dict": model.state_dict()}, path)


def load_model(path, map_location="cpu") -> TwoTower:
    """Rebuild a model from a saved file and put it in evaluation mode."""
    blob = torch.load(path, map_location=map_location, weights_only=True)
    model = TwoTower(**blob["init_args"])
    model.load_state_dict(blob["state_dict"])
    return model.eval()

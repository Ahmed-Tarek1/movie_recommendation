"""
Factorization Machine model definition.
Shared between training notebooks and backend inference —
import THIS class in both places to avoid load/train mismatches.
"""
import torch
import torch.nn as nn


class FactorizationMachine(nn.Module):
    """
    Second-order FM: y = w0 + sum(w_i * x_i) + sum_i<j <v_i, v_j> x_i x_j

    field_dims: list of cardinalities for each categorical field
                e.g. [n_users, n_items, n_genres] for a basic setup.
    embed_dim: latent factor dimension (k)
    """

    def __init__(self, field_dims: list[int], embed_dim: int = 16):
        super().__init__()
        self.field_dims = field_dims
        self.embed_dim = embed_dim

        total_dims = sum(field_dims)
        # offsets to map per-field indices into a single flat embedding table
        self.offsets = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(field_dims[:-1]), dim=0))
        )

        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Embedding(total_dims, 1)
        self.embedding = nn.Embedding(total_dims, embed_dim)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, num_fields) long tensor of per-field categorical indices
           (already offset-adjusted is NOT required — done internally)
        """
        x = x + self.offsets.to(x.device)

        linear_term = self.linear(x).sum(dim=1).squeeze(-1) + self.bias

        emb = self.embedding(x)  # (batch, num_fields, embed_dim)
        sum_sq = emb.sum(dim=1).pow(2)
        sq_sum = emb.pow(2).sum(dim=1)
        interaction_term = 0.5 * (sum_sq - sq_sum).sum(dim=1)

        return linear_term + interaction_term

    def get_item_embeddings(self, item_field_index: int, item_offset_within_field: range) -> torch.Tensor:
        """
        Pull out learned item latent vectors for similarity computation.
        item_field_index: which field in field_dims corresponds to items
        item_offset_within_field: range of raw item indices (0..n_items-1)
        """
        base_offset = int(self.offsets[item_field_index].item())
        idxs = torch.tensor([base_offset + i for i in item_offset_within_field])
        return self.embedding(idxs).detach()


class DeepFM(nn.Module):
    """
    FM component (linear + 2nd-order interactions) + a deep MLP component,
    sharing the same embedding table. Extends FactorizationMachine rather than
    duplicating it, so the FM half stays identical and battle-tested.

    mlp_dims: hidden layer sizes, e.g. [128, 64, 32]
    dropout: applied after each hidden layer
    """

    def __init__(
        self,
        field_dims: list[int],
        embed_dim: int = 16,
        mlp_dims: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if mlp_dims is None:
            mlp_dims = [128, 64, 32]

        self.fm = FactorizationMachine(field_dims, embed_dim)

        num_fields = len(field_dims)
        input_dim = num_fields * embed_dim

        layers = []
        prev_dim = input_dim
        for dim in mlp_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_offset = x + self.fm.offsets.to(x.device)

        # FM component (linear + pairwise interaction terms)
        linear_term = self.fm.linear(x_offset).sum(dim=1).squeeze(-1) + self.fm.bias
        emb = self.fm.embedding(x_offset)  # (batch, num_fields, embed_dim)
        sum_sq = emb.sum(dim=1).pow(2)
        sq_sum = emb.pow(2).sum(dim=1)
        fm_interaction = 0.5 * (sum_sq - sq_sum).sum(dim=1)
        fm_out = linear_term + fm_interaction

        # Deep component — same embeddings, flattened into the MLP
        deep_input = emb.view(emb.size(0), -1)
        deep_out = self.mlp(deep_input).squeeze(-1)

        return fm_out + deep_out

    def get_item_embeddings(self, item_field_index: int, item_offset_within_field: range) -> torch.Tensor:
        """Reuses the FM half's embedding table — same vectors power both components."""
        return self.fm.get_item_embeddings(item_field_index, item_offset_within_field)

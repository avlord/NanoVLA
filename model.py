import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SimpleTransformer(nn.Module):
    """
    Per-step token = [img1 (| img2) | state] -> d_model.
    Task embedding (MiniLM) is prepended as an additional token.
    Small Transformer encoder over the sequence; the last obs hidden state
    is decoded into (action_horizon, action_dim).
    """

    def __init__(
        self,
        img_dim: int,
        state_dim: int,
        task_dim: int,
        action_dim: int = 7,
        seq_len: int = 1024,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        num_cameras: int = 2,
    ):
        super().__init__()
        self.action_dim = action_dim
        obs_input_dim = img_dim * num_cameras + state_dim
        self.obs_proj = nn.Linear(obs_input_dim, d_model)
        self.task_proj = nn.Linear(task_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
        )

    def forward(self, image_left, image_gripper, instruction_emb, gripper_state):
        obs = torch.cat((image_left, image_gripper, gripper_state), dim=-1)
        obs_tok = self.obs_proj(obs)

        task_tok = self.task_proj(instruction_emb)
        x = torch.cat([task_tok, obs_tok], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x)

        return self.head(x)[:, 1:]

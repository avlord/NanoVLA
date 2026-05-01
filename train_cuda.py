import os

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from dataset import ParquetEpisodeImageDataset
from embedding_models import TextEmbeddingModel, VideoEmbeddingModel
from model import SimpleTransformer


def _load_tasks(dataset_root: str) -> pd.DataFrame:
    path = os.path.join(dataset_root, "meta", "tasks.parquet")
    df = pd.read_parquet(path).reset_index()
    df = df.rename(columns={df.columns[0]: "task"})
    df["task_index"] = df["task_index"].astype(np.int64)
    df["task"] = df["task"].astype(str)
    return df.sort_values("task_index").reset_index(drop=True)


def encode_text(dataset_root: str) -> np.ndarray:
    """Encode all task descriptions into embeddings. Returns (num_tasks, embed_dim)."""
    tasks_df = _load_tasks(dataset_root)
    text_encoder = TextEmbeddingModel()
    return text_encoder(tasks_df["task"].tolist())


def main():
    # --- Configuration -------------------------------------------------
    DATASET_ROOT = "dataset"
    CHECKPOINT_DIR = "checkpoints"
    ENCODER_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"

    NUM_WORKERS = 8
    BATCH_SIZE = 8

    DEVICE = "cuda"

    MODEL_CONFIG = dict(
        img_dim=384,
        task_dim=384,
        state_dim=8,
        action_dim=7,
        seq_len=1024,
        d_model=256,
        n_heads=8,
        n_layers=4,
        dropout=0.1,
        num_cameras=2,
    )
    # -------------------------------------------------------------------

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_dataset = ParquetEpisodeImageDataset(DATASET_ROOT)
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=True,
    )

    video_encoder = VideoEmbeddingModel(device=DEVICE)
    text_embeddings = torch.from_numpy(encode_text(DATASET_ROOT)).to(
        DEVICE
    )  # (num_tasks, 384)

    policy = SimpleTransformer(**MODEL_CONFIG).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    for step, batch in enumerate(loader):

        if step % 100 == 0:
            print(f"saving checkpoint at step {step}")
            torch.save(
                {
                    "model": policy.state_dict(),
                    "config": MODEL_CONFIG,
                    "encoder_model": ENCODER_MODEL,
                    "step": step,
                },
                os.path.join(CHECKPOINT_DIR, f"model_{step}.pt"),
            )

        optimizer.zero_grad()

        image_left = video_encoder(batch["images_left"].to(DEVICE))  # (B, T, 384)
        image_gripper = video_encoder(batch["images_gripper"].to(DEVICE))  # (B, T, 384)
        instruction_emb = text_embeddings[batch["task_index"]][
            :, None, :
        ]  # (B, 1, 384)
        actions = batch["action"].to(DEVICE)  # (B, T, 7)
        gripper_state = batch["state"].to(DEVICE)  # (B, T, 8)

        out = policy(
            image_left=image_left,
            image_gripper=image_gripper,
            instruction_emb=instruction_emb,
            gripper_state=gripper_state,
        )

        loss = torch.nn.functional.smooth_l1_loss(out, actions, beta=0.1)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()

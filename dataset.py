from torch.utils.data import Dataset, DataLoader
import pyarrow.dataset as ds
import os
import pandas as pd
import numpy as np
import torch
from utils import decode_image


class ParquetEpisodeImageDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        episodes_file="meta/episodes/chunk-000/file-000.parquet",
    ):
        self.dataset_root = dataset_root

        self.episodes = pd.read_parquet(
            os.path.join(dataset_root, episodes_file)
        ).reset_index(drop=True)

    def __len__(self):
        return len(self.episodes)

    def _get_chunk_dataset(self, chunk_idx):
        path = os.path.join(
            self.dataset_root,
            "data",
            f"chunk-{chunk_idx:03d}",
        )
        return ds.dataset(path, format="parquet")

    def __getitem__(self, idx):
        row = self.episodes.iloc[idx]

        chunk_idx = int(row["data/chunk_index"])
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])

        columns = [
            "index",
            "episode_index",
            "frame_index",
            "task_index",
            "observation.images.image",  # 3x256x256 LEFT-VIEW
            "observation.images.image2",  # 3x256x256  GRIPPER-VIEW
            "observation.state",  # 8-dim GRIPPER-STATE
            "action",  # 7-dim action
        ]

        dataset = self._get_chunk_dataset(chunk_idx)

        table = dataset.to_table(
            filter=(ds.field("index") >= start) & (ds.field("index") < end),
            columns=columns,
        )

        df = table.to_pandas().sort_values("index").reset_index(drop=True)

        images_left = torch.stack(
            [decode_image(x) for x in df["observation.images.image"].to_numpy()]
        )
        images_gripper = torch.stack(
            [decode_image(x) for x in df["observation.images.image2"].to_numpy()],
        )
        state = torch.tensor(
            np.stack(df["observation.state"].to_numpy()), dtype=torch.float32
        )
        action = torch.tensor(np.stack(df["action"].to_numpy()), dtype=torch.float32)

        # Sample a random 64-frame window. Fixed length keeps XLA from recompiling.

        sequence_length = images_left.shape[0]
        low = min(32, sequence_length // 2)
        high = max(sequence_length - 32, low + 1)
        window_center = np.random.randint(low, high)

        out = {
            "images_left": images_left[window_center - 32 : window_center + 32],
            "images_gripper": images_gripper[window_center - 32 : window_center + 32],
            "state": state[window_center - 32 : window_center + 32],
            "action": action[window_center - 32 : window_center + 32],
            "task_index": int(df["task_index"].iloc[0]),
        }

        return out

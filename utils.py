import numpy as np
from PIL import Image
import torch
from io import BytesIO
import imageio.v2 as imageio

# import torch_xla.runtime as xr
# from torch_xla.distributed.spmd import Mesh


# def build_mesh():
# n = xr.global_runtime_device_count()
# return Mesh(np.arange(n), (n,), ("data",)), n


def save_gif(obs_list, path="rollout.gif", camera_key="agentview_image", duration=50):
    frames = []

    for obs in obs_list:
        frame = obs[camera_key]

        frame = frame.astype(np.uint8)
        frames.append(frame)

    imageio.mimsave(path, frames, duration=duration)
    print(f"Saved gif to {path}")


def decode_image(x):
    # x is bytes containing a PNG/JPEG
    img = Image.open(BytesIO(x["bytes"])).convert("RGB")
    arr = np.array(img)  # [H, W, C], uint8
    return torch.from_numpy(arr).permute(2, 0, 1)  # [C, H, W]

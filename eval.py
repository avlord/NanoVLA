import os

import numpy as np
import torch

# XLA must be configured before any other torch imports
os.environ["PJRT_DEVICE"] = "TPU"
import torch_xla
import torch_xla.core.xla_model as xm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from embedding_models import TextEmbeddingModel, VideoEmbeddingModel
from model import SimpleTransformer
from utils import save_gif

DEVICE = torch_xla.device()
CHECKPOINT = "checkpoints/model_100.pt"

img_model = VideoEmbeddingModel(device=DEVICE)
txt_model = TextEmbeddingModel()


def setup_libero(task_suite, task_id=0):
    suite = benchmark.get_benchmark_dict()[task_suite]()
    task = suite.get_task(task_id)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.reset()
    env.set_init_state(suite.get_task_init_states(task_id)[0])

    obs, _, _, _ = env.step([0.0] * 7)

    return obs, env, task.language


def evaluate(ckpt_path, task_suite="libero_object", n_inits=5, max_steps=300):
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = raw["config"]

    policy = SimpleTransformer(**cfg).to(DEVICE).eval()
    policy.load_state_dict(raw["model"])

    obs, env, task_language = setup_libero(task_suite)

    task_emb = (
        torch.from_numpy(txt_model(task_language)).to(DEVICE).unsqueeze(0)
    )  # (1, 1, 384)

    # Fixed-size rolling window matching the 64-frame training context.
    # Constant shapes mean XLA compiles the graph once and reuses it every step.
    context_len = 64
    img_dim = cfg["img_dim"]
    state_dim = cfg["state_dim"]
    buf_left = torch.zeros(1, context_len, img_dim, device=DEVICE)
    buf_grip = torch.zeros(1, context_len, img_dim, device=DEVICE)
    buf_state = torch.zeros(1, context_len, state_dim, device=DEVICE)

    obs_list = []
    for _ in range(max_steps):
        image_emb = img_model(
            torch.stack(
                [
                    torch.from_numpy(obs["agentview_image"]),
                    torch.from_numpy(obs["robot0_eye_in_hand_image"]),
                ]
            )
            .permute(0, -1, 1, 2)[:, None, :]
            .to(DEVICE)  # (2, 1, C, H, W)
        )  # (2, 1, img_dim)

        new_state = torch.tensor(
            obs["robot0_proprio-state"][:state_dim],
            dtype=torch.float32,
            device=DEVICE,
        )[
            None, None
        ]  # (1, 1, state_dim)

        # Drop oldest frame, append newest — shapes stay (1, context_len, D)
        buf_left = torch.cat([buf_left[:, 1:], image_emb[0].unsqueeze(1)], dim=1)
        buf_grip = torch.cat([buf_grip[:, 1:], image_emb[1].unsqueeze(1)], dim=1)
        buf_state = torch.cat([buf_state[:, 1:], new_state], dim=1)

        with torch.no_grad():
            action = (
                policy(buf_left, buf_grip, task_emb, buf_state)[0, -1].cpu().numpy()
            )

        xm.mark_step()
        obs, _, done, _ = env.step(action.tolist())
        obs_list.append(obs)

    env.close()
    return obs_list


if __name__ == "__main__":
    obs_list = evaluate(CHECKPOINT)
    save_gif(obs_list)

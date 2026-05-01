"""
Download the HuggingFaceVLA/libero dataset from Hugging Face Hub.

Dataset: HuggingFaceVLA/libero
  - Robot:      Panda
  - Episodes:   1,693  |  Frames: 273,465  |  Tasks: 40
  - FPS:        10
  - Cameras:    observation.images.image, observation.images.image2  (256x256, AV1)
  - State dim:  8  (proprio)
  - Action dim: 7
  - Dataset size is roughly 30GB

Run:
    pip install lerobot huggingface_hub
    python download_dataset.py
"""

from huggingface_hub import snapshot_download

REPO_ID = "HuggingFaceVLA/libero"
LOCAL_DIR = "/dataset"


def main():
    print(f"Downloading {REPO_ID} -> {LOCAL_DIR}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        force_download=False,
        max_workers=4,
    )
    print("Done.")


if __name__ == "__main__":
    main()

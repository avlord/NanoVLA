# VLA Minimal Tutorial: Training a Robot Policy from Scratch

This tutorial walks you through training a **Vision-Language-Action (VLA)** policy on the LIBERO tabletop manipulation dataset. The policy learns to map camera images + a text instruction → robot arm actions, using behavior cloning on human demonstrations.

**Pipeline overview:**
```
1. download_dataset.py  →  raw parquet data (~30 GB)
2. dataset.py           →  loads episodes from disk
3. embedding_models.py  →  encodes images (DINOv3) and text (MiniLM)
4. model.py             →  SimpleTransformer policy
5. train.py             →  training loop with checkpointing
6. eval.py              →  run a rollout and save a GIF
```

---

### 🛠️ System Setup

```bash
sudo apt update
sudo apt install -y cmake build-essential libegl1-mesa-dev libgl1-mesa-dev

# 🚀 Step 0 — Create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# 🔧 Fix for egl-probe + modern CMake
export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"

# 📦 Install dependencies
uv pip install "lerobot[libero]" \
  huggingface_hub \
  torch \
  sentence-transformers \
  transformers \
  pyarrow \
  pandas \
  Pillow \
  libtpu==0.0.17 \
  torch==2.8.0 torch_xla==2.8.0 torchvision \
  -f https://storage.googleapis.com/libtpu-releases/index.html

```
---

### Step 1 — Download the Dataset

**File:** [`download_dataset.py`](download_dataset.py)

```python
from huggingface_hub import snapshot_download

REPO_ID  = "HuggingFaceVLA/libero"
LOCAL_DIR = "./dataset"

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=LOCAL_DIR,
    force_download=False,
    max_workers=4,        # download 4 shards in parallel
)
```

**Run it:**
```bash
python download_dataset.py
```

**What you get — the LIBERO dataset:**

| Property  | Value |
|-----------|-------|
| Robot     | Franka Panda arm |
| Episodes  | 1,693 human demos |
| Frames    | 273,465 total steps |
| Tasks     | 40 tabletop manipulation tasks |
| FPS       | 10 Hz |
| Cameras   | `image` (left-side, 256×256) and `image2` (wrist/gripper, 256×256) |
| State dim | 8 (proprioception: EEF position + quaternion + gripper) |
| Action dim | 7 (Cartesian delta: 3 pos + 3 rot + 1 gripper) |
| Size      | ~30 GB |

Data lands in `./dataset/` as partitioned **Parquet** files — a columnar binary format that's much faster to read than CSV and stores images as compressed bytes.

```
dataset/
├── meta/
│   ├── episodes/chunk-000/file-000.parquet   ← episode metadata
│   └── tasks.parquet                          ← 40 task descriptions
└── data/
    ├── chunk-000/   ← observation rows (images, state, action)
    ├── chunk-001/
    └── ...
```

---

## Step 2 — Understanding the Dataset Loader

**File:** [`dataset.py`](dataset.py)

The dataset is split across **chunks** (groups of episodes). Each episode lives in one chunk but spans a contiguous range of rows. The loader must:
1. Look up which chunk an episode is in (from the episode metadata).
2. Read only the rows for that episode (using a filter — no full scan).
3. Decode compressed image bytes into tensors.
4. Sample a fixed-length 64-frame window for XLA compatibility.

```python
class ParquetEpisodeImageDataset(Dataset):
    def __init__(self, dataset_root, ...):
        # Load the index: one row per episode, tells us chunk + row range
        self.episodes = pd.read_parquet(
            os.path.join(dataset_root, "meta/episodes/chunk-000/file-000.parquet")
        ).reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.episodes.iloc[idx]

        chunk_idx = int(row["data/chunk_index"])   # which parquet folder
        start     = int(row["dataset_from_index"]) # first row of this episode
        end       = int(row["dataset_to_index"])   # last row (exclusive)

        # Open the chunk as a lazy Arrow dataset (no data loaded yet)
        dataset = ds.dataset(
            os.path.join(self.dataset_root, "data", f"chunk-{chunk_idx:03d}"),
            format="parquet",
        )

        # Push the filter down to Parquet — only deserializes the matching rows
        table = dataset.to_table(
            filter=(ds.field("index") >= start) & (ds.field("index") < end),
            columns=[...],
        )

        df = table.to_pandas().sort_values("index").reset_index(drop=True)

        # Each image column is a dict {"bytes": <compressed PNG/JPEG bytes>}
        # decode_image converts those bytes → CHW uint8 tensor
        images_left    = torch.stack([decode_image(x) for x in df["observation.images.image"]])
        images_gripper = torch.stack([decode_image(x) for x in df["observation.images.image2"]])
        state          = torch.tensor(np.stack(df["observation.state"]), dtype=torch.float32)
        action         = torch.tensor(np.stack(df["action"]),            dtype=torch.float32)

        # Sample a random 64-frame window. Fixed length keeps XLA from recompiling.
        sequence_length = images_left.shape[0]
        low = min(32, sequence_length // 2)
        high = max(sequence_length - 32, low + 1)
        window_center = np.random.randint(low, high)

        return {
            "images_left":    images_left[window_center - 32 : window_center + 32],
            "images_gripper": images_gripper[window_center - 32 : window_center + 32],
            "state":          state[window_center - 32 : window_center + 32],
            "action":         action[window_center - 32 : window_center + 32],
            "task_index":     int(df["task_index"].iloc[0]),
        }
        # shapes returned (always 64 frames):
        #   images_left    : (64, 3, 256, 256)
        #   images_gripper : (64, 3, 256, 256)
        #   state          : (64, 8)
        #   action         : (64, 7)
```

**Why 64 frames?** XLA (TPU) compiles a computation graph per unique tensor shape. Variable-length episodes would trigger a recompile for every batch. Sampling a fixed-size window avoids this entirely.

**Image decoding** — [`utils.py`](utils.py):
```python
def decode_image(x):
    # x["bytes"] is a compressed PNG/JPEG blob stored in Parquet
    img = Image.open(BytesIO(x["bytes"])).convert("RGB")
    arr = np.array(img)                             # (H, W, 3) uint8
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W) uint8
```

---

## Step 3 — Embedding Models

**File:** [`embedding_models.py`](embedding_models.py)

Raw pixels and raw text strings are high-dimensional and noisy. Before passing them to the policy, we compress them into compact **embedding vectors** using pre-trained encoders.

### 3a — Text encoder: MiniLM (384-dim)

```python
from transformers import AutoTokenizer, AutoModel

class TextEmbeddingModel:
    def __init__(self, device=None):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(device).eval()
        self.embed_dim = self.model.config.hidden_size  # 384

    @torch.no_grad()
    def __call__(self, texts):
        # texts: list of strings, e.g. ["put the bowl on the plate", ...]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)  # mean pool over tokens
        return emb.cpu().numpy()                     # (N, 384)
```

There are only **40 unique task descriptions** in LIBERO, so we embed them all once at training startup and keep the 40×384 matrix in device memory (see `encode_text()` in `train.py`).

### 3b — Image encoder: DINOv3 (384-dim)

```python
from transformers import AutoImageProcessor, AutoModel

class VideoEmbeddingModel:
    def __init__(self, device=None):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        self.model = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m").to(device).eval()
        self.embed_dim = self.model.config.hidden_size  # 384

    @torch.no_grad()
    def __call__(self, x):
        # x: (B, T, 3, H, W) uint8 pixel tensor
        B, T, C, H, W = x.shape
        x = einops.rearrange(x, "B T C H W -> (B T) C H W")
        inputs = self.processor(images=x, return_tensors="pt", do_rescale=True).to(self.device)
        out = self.model(**inputs).pooler_output          # ((B*T), 384)
        return einops.rearrange(out, "(B T) D -> B T D", B=B)  # (B, T, 384)
```

**Why DINOv3?** It was trained with self-supervised learning on a massive image corpus, so its embeddings are semantically rich without any task-specific fine-tuning. The ViT-S/16 variant is small enough to run on a single TPU but still gives excellent representations.

---

## Step 4 — The Policy Model

**File:** [`model.py`](model.py)

The policy is a small **Transformer encoder** that takes a *sequence* of past observations and predicts the next chunk of actions.

### Architecture walkthrough

```
Inputs per timestep t:
  image_left[t]    : (384,)  ← DINOv3 embedding of the left camera
  image_gripper[t] : (384,)  ← DINOv3 embedding of the gripper camera
  state[t]         : (8,)    ← proprioception (EEF pos + quat + gripper)

  Concatenate → (768+8 = 776,) raw observation token
  Linear(776 → 256)           → obs_token[t]  (d_model = 256)

Task embedding (once per episode):
  instruction_emb  : (384,)  ← MiniLM embedding of the task text
  Linear(384 → 256)           → task_token    (d_model = 256)

Sequence fed to Transformer:
  [task_token, obs_token[0], obs_token[1], ..., obs_token[T-1]]
  length = T + 1

Output:
  head(hidden[1:])  → (T, action_dim) predictions
```


### Positional encoding

```python
class PositionalEncoding(nn.Module):
    # Standard sinusoidal encoding — injects temporal order into the sequence.
    # Without it the Transformer treats the sequence as a bag of tokens.
    def __init__(self, d_model, max_len=64):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-log(10000) / d_model))
        pe[:, 0::2] = sin(pos * div)
        pe[:, 1::2] = cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]   # broadcast over batch dim
```

### Full forward pass

```python
class SimpleTransformer(nn.Module):
    def forward(self, image_left, image_gripper, instruction_emb, gripper_state):
        # image_left      : (B, T, 384)
        # image_gripper   : (B, T, 384)
        # gripper_state   : (B, T, 8)
        # instruction_emb : (B, 1, 384)

        # 1. Build per-step observation token
        obs     = torch.cat([image_left, image_gripper, gripper_state], dim=-1)  # (B, T, 776)
        obs_tok = self.obs_proj(obs)   # (B, T, 256)

        # 2. Build task token
        task_tok = self.task_proj(instruction_emb)   # (B, 1, 256)

        # 3. Prepend task token → sequence length T+1
        x = torch.cat([task_tok, obs_tok], dim=1)    # (B, T+1, 256)
        x = self.pos_enc(x)

        # 4. Run Transformer encoder (all tokens attend to all other tokens)
        x = self.encoder(x)   # (B, T+1, 256)

        # 5. Decode each observation position into action_dim predictions
        #    Skip index 0 (the task token)
        return self.head(x[:, 1:])   # (B, T, action_dim)
```

**Why prepend the task token?** It gives the Transformer a global context token that every observation token can attend to, effectively conditioning all predictions on the language instruction.

---

## Step 5 — Training Loop

**File:** [`train.py`](train.py)

```python
def main():
    DEVICE = torch_xla.device()   # TPU
    DATASET_ROOT = "dataset"

    # --- 1. Load dataset ---
    train_dataset = ParquetEpisodeImageDataset(DATASET_ROOT)
    loader = DataLoader(train_dataset, batch_size=8, num_workers=8, drop_last=True)

    # --- 2. Pre-compute text embeddings (40 tasks × 384-dim) ---
    text_embeddings = torch.from_numpy(encode_text(DATASET_ROOT)).to(DEVICE)
    # shape: (40, 384)  — one vector per unique LIBERO task

    # --- 3. Build model + optimizer ---
    policy    = SimpleTransformer(**MODEL_CONFIG).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    # --- 4. Encode images on-the-fly and train ---
    video_encoder = VideoEmbeddingModel(device=DEVICE)

    for step, batch in enumerate(loader):
        optimizer.zero_grad()

        image_left    = video_encoder(batch["images_left"].to(DEVICE))    # (B, T, 384)
        image_gripper = video_encoder(batch["images_gripper"].to(DEVICE)) # (B, T, 384)
        instruction_emb = text_embeddings[batch["task_index"]][:, None, :] # (B, 1, 384)
        actions       = batch["action"].to(DEVICE)                        # (B, T, 7)
        gripper_state = batch["state"].to(DEVICE)                         # (B, T, 8)

        out = policy(
            image_left=image_left,
            image_gripper=image_gripper,
            instruction_emb=instruction_emb,
            gripper_state=gripper_state,
        )

        # Smooth-L1 (Huber) loss: like MSE but less sensitive to outlier actions
        loss = F.smooth_l1_loss(out, actions, beta=0.1)
        loss.backward()
        optimizer.step()
        xm.mark_step()   # flush TPU graph

        # Save a checkpoint every 100 steps
        if step % 100 == 0:
            torch.save(
                {"model": policy.state_dict(), "config": MODEL_CONFIG, "step": step},
                f"checkpoints/model_{step}.pt",
            )
```

**Run it:**
```bash
python train.py
```

Checkpoints land in `checkpoints/model_0.pt`, `model_100.pt`, etc. Each `.pt` file stores the model weights, the architecture config dict, and the step number so you can resume or evaluate later.

**Training notes:**
- Image encoding (DINOv3 forward pass) happens inside the training loop per batch — this is the bottleneck. For faster iteration you could pre-encode all images to disk first.
- `drop_last=True` ensures every batch is exactly `batch_size=8`, which is required on TPU to avoid shape-triggered recompilation.
- `smooth_l1_loss` with `beta=0.1` behaves like L1 for errors > 0.1 and like L2 for smaller errors, which is robust to the occasional large action outlier.
- No learning rate schedule is used here — a good next step would be adding a cosine decay.

---

## Step 6 — Evaluation in the LIBERO Simulator

**Files:** [`eval.py`](eval.py), [`utils.py`](utils.py)

After training, run the policy in the LIBERO simulator and save a GIF of the rollout.

```bash
python eval.py
# or change CHECKPOINT at the top of the file
```

### 6a — Setting up the environment

```python
def setup_libero(task_suite, task_id=0):
    suite = benchmark.get_benchmark_dict()[task_suite]()
    task  = suite.get_task(task_id)
    bddl  = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"

    # OffScreenRenderEnv renders cameras off-screen (no display required)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.reset()
    env.set_init_state(suite.get_task_init_states(task_id)[0])

    # Send a no-op first to get the initial observation
    obs, _, _, _ = env.step([0.0] * 7)

    return obs, env, task.language  # task.language is the text instruction
```

### 6b — Rollout loop

```python
def evaluate(ckpt_path, task_suite="libero_object", n_inits=5, max_steps=300):
    raw    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg    = raw["config"]
    policy = SimpleTransformer(**cfg).to(DEVICE).eval()
    policy.load_state_dict(raw["model"])

    obs, env, task_language = setup_libero(task_suite)

    # Encode the task text once — shape (1, 1, 384): batch × seq × dim
    task_emb = torch.from_numpy(txt_model(task_language)).to(DEVICE).unsqueeze(0)

    # Pre-allocate fixed-size rolling buffers matching the 64-frame training context.
    # Constant shapes mean XLA compiles the graph once and reuses it every step.
    context_len = 64
    img_dim, state_dim = cfg["img_dim"], cfg["state_dim"]
    buf_left  = torch.zeros(1, context_len, img_dim,   device=DEVICE)
    buf_grip  = torch.zeros(1, context_len, img_dim,   device=DEVICE)
    buf_state = torch.zeros(1, context_len, state_dim, device=DEVICE)

    obs_list = []
    for _ in range(max_steps):
        # LIBERO returns images as (H, W, C) numpy arrays — stack and convert to (2, 1, C, H, W)
        image_emb = img_model(
            torch.stack([
                torch.from_numpy(obs["agentview_image"]),
                torch.from_numpy(obs["robot0_eye_in_hand_image"]),
            ])
            .permute(0, -1, 1, 2)[:, None, :]  # (2, 1, C, H, W)
            .to(DEVICE)
        )  # → (2, 1, img_dim)

        new_state = torch.tensor(
            obs["robot0_proprio-state"][:state_dim], dtype=torch.float32, device=DEVICE
        )[None, None]  # (1, 1, state_dim)

        # Drop oldest frame, append newest — shapes stay (1, context_len, D)
        buf_left  = torch.cat([buf_left[:,  1:], image_emb[0].unsqueeze(1)], dim=1)
        buf_grip  = torch.cat([buf_grip[:,  1:], image_emb[1].unsqueeze(1)], dim=1)
        buf_state = torch.cat([buf_state[:, 1:], new_state],                 dim=1)

        with torch.no_grad():
            # policy returns (1, T, action_dim); take the last timestep prediction
            action = policy(buf_left, buf_grip, task_emb, buf_state)[0, -1].cpu().numpy()

        xm.mark_step()
        obs, _, done, _ = env.step(action.tolist())
        obs_list.append(obs)

    env.close()
    return obs_list
```

**Key details:**
- `obs["agentview_image"]` is `(256, 256, 3)` uint8 — `.permute(0, -1, 1, 2)` converts the stacked pair from `(2, H, W, C)` to `(2, C, H, W)`, then `[:, None, :]` adds the time dimension the video encoder expects.
- `robot0_proprio-state` is a 39-dim full proprio vector from LIBERO; we slice `[:cfg["state_dim"]]` (default 8) to match training.
- The rolling window is always exactly 64 frames — the oldest frame is dropped and the newest appended each step. This matches the training window size and keeps tensor shapes constant so XLA doesn't recompile.

### 6c — Saving the GIF

```python
# utils.py
def save_gif(obs_list, path="rollout.gif", camera_key="agentview_image", duration=50):
    frames = [obs[camera_key].astype(np.uint8) for obs in obs_list]
    imageio.mimsave(path, frames, duration=duration)   # duration = ms per frame
```

```python
# at the end of eval.py
obs_list = evaluate(CHECKPOINT)
save_gif(obs_list)   # → rollout.gif
```

The GIF gives you a quick visual sanity check — if the arm is moving randomly the embeddings or proprio slice probably don't match training.

---

## Architecture Summary

```
Task instruction text
       │
  MiniLM (384-dim)
       │ task_token (1, 1, 256)
       │
Camera left (64 × 256×256×3)   Camera gripper (64 × 256×256×3)
       │                                │
  DINOv3 (384-dim)               DINOv3 (384-dim)
       │                                │
       └──────────┬─────────────────────┘
              concat + state (8-dim)
                  │  obs_token (1, 64, 256)
                  │
       prepend task_token → (1, 65, 256)
                  │
         Sinusoidal pos enc
                  │
       TransformerEncoder
       (4 layers, 8 heads, d_ff=1024)
                  │
        head (LayerNorm + 2-layer MLP)
                  │
        Predicted actions (1, 64, 7)
```

---

## Key Concepts Recap

| Concept | What it is | Why it matters here |
|---|---|---|
| **Behavior cloning** | Learn policy by imitating demonstrations (supervised learning on (obs, action) pairs) | Simple, stable training; no RL needed |
| **DINOv3** | Self-supervised Vision Transformer image encoder | Strong visual features without task-specific labels |
| **MiniLM** | Small sentence encoder (384-dim) | Compact, fast text embeddings for task conditioning |
| **Task token prepend** | Language embedding as the first Transformer token | Conditions all attention layers on the instruction |
| **Fixed rolling window** | Always pass the last 64 frames, drop the oldest each step | Keeps tensor shapes constant so XLA compiles the graph once |
| **Smooth-L1 loss** | Huber loss with β=0.1 | Robust to outlier actions in the demonstration data |

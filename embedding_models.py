import torch
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
import einops


class TextEmbeddingModel:
    def __init__(self, device=None):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        )
        self.model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        ).to(device).eval()
        self.embed_dim = self.model.config.hidden_size

    @torch.no_grad()
    def __call__(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
        return emb.cpu().numpy()


class VideoEmbeddingModel:
    def __init__(self, device=None):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov3-vits16-pretrain-lvd1689m"
        )
        self.model = AutoModel.from_pretrained(
            "facebook/dinov3-vits16-pretrain-lvd1689m"
        ).to(device).eval()
        self.embed_dim = self.model.config.hidden_size

    @torch.no_grad()
    def __call__(self, x):
        B, T, C, H, W = x.shape
        x = einops.rearrange(x, "B T C H W -> (B T) C H W")
        inputs = self.processor(images=x, return_tensors="pt", do_rescale=True).to(self.device)
        out = self.model(**inputs).pooler_output
        return einops.rearrange(out, "(B T) D -> B T D", B=B)

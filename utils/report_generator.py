import torch
import torch.nn as nn
from torchvision import models
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image
import torchvision.transforms as transforms


# ============================
# SAME TRANSFORM AS TRAINING
# ============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================
# SAME CNN + GPT2 ARCHITECTURE
# ============================
class CNN_GPT2(nn.Module):
    def __init__(self, gpt2_model="gpt2"):
        super().__init__()

        # CNN = ResNet18 WITHOUT classifier
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # GPT-2 LM Head
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model)

        # Combined image feature size → GPT2 embedding size
        self.linear = nn.Linear(backbone.fc.in_features * 2, self.gpt2.config.n_embd)
        self.dropout = nn.Dropout(0.1)

    def forward(self, img_tensor, input_ids, attention_mask):
        img1 = img_tensor[:, :3, :, :]
        img2 = img_tensor[:, 3:, :, :]

        feat1 = self.pool(self.cnn(img1)).squeeze(-1).squeeze(-1)
        feat2 = self.pool(self.cnn(img2)).squeeze(-1).squeeze(-1)

        # Combine features
        cnn_feat = torch.cat([feat1, feat2], dim=1)
        prefix_emb = self.dropout(self.linear(cnn_feat)).unsqueeze(1)

        # GPT2 embeddings
        gpt_emb = self.gpt2.transformer.wte(input_ids)

        # Prepend prefix
        gpt_emb = torch.cat([prefix_emb, gpt_emb], dim=1)

        # Attention mask
        prefix_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Labels shifted
        prefix_token = self.gpt2.config.pad_token_id or self.gpt2.config.eos_token_id
        labels = torch.cat([
            torch.full((input_ids.size(0), 1), prefix_token, dtype=torch.long, device=input_ids.device),
            input_ids
        ], dim=1)

        return self.gpt2(inputs_embeds=gpt_emb, attention_mask=attention_mask, labels=labels)


# ============================
# REPORT GENERATOR
# ============================
class ReportGenerator:
    def __init__(self, ckpt_path, device="cpu", max_length=128):
        self.device = device
        self.max_length = max_length

        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", revision="main")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build model
        self.model = CNN_GPT2("gpt2").to(device)

        # Load checkpoint (auto-detect type)
        raw = torch.load(ckpt_path, map_location=device)

        if "model_state_dict" in raw:
            print("🔍 Loading from wrapped checkpoint")
            self.model.load_state_dict(raw["model_state_dict"])
        else:
            print("🔍 Loading raw state_dict")
            self.model.load_state_dict(raw)

        self.model.eval()

    # -------------------------
    # Preprocessing for Streamlit
    # -------------------------
    def preprocess(self, front_img, lateral_img):
        front = transform(front_img.convert("RGB"))
        lat = transform(lateral_img.convert("RGB"))
        return torch.cat([front, lat], dim=0).unsqueeze(0).to(self.device)

    # -------------------------
    # Autoregressive text generation
    # -------------------------
    @torch.no_grad()
    def generate(self, front_img, lateral_img, prompt="Findings: "):
        img_tensor = self.preprocess(front_img, lateral_img)

        # Encode prompt text
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        )
        input_ids = enc["input_ids"].to(self.device)          # shape (1, L)
        attention_mask = enc["attention_mask"].to(self.device)

        generated = input_ids

        for _ in range(self.max_length):
            out = self.model(img_tensor, generated, attention_mask)
            logits = out.logits[0, -1, :]

            # Greedy (for now)
            next_token = torch.argmax(logits).item()

            generated = torch.cat(
                [generated,
                 torch.tensor([[next_token]], device=self.device)],
                dim=1
            )
            attention_mask = torch.ones_like(generated)

            if next_token == self.tokenizer.eos_token_id:
                break

        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text.strip()

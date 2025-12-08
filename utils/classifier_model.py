import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights
from PIL import Image

# ---------------------------------------------
# Model Architecture (same as your training)
# ---------------------------------------------
class XRayClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Load DenseNet121 with ImageNet weights
        self.base = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        # Number of features before the classification head
        num_ftrs = self.base.classifier.in_features

        # Remove DenseNet classification head
        self.base.classifier = nn.Identity()

        # Custom classification head
        # You trained with 3 outputs → adjust as needed
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 3)     # 3 classes
        )

    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------
# Wrapper for preprocessing + prediction
# ---------------------------------------------
class TBClassifier:
    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)

        self.model = XRayClassifier().to(self.device)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        # ✅ Match notebook preprocessing: resize + [0,1], no mean/std
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # already scales to [0,1]
        ])

        # ✅ Real labels from notebook
        self.class_names = ["health", "tb" ,"sick"]
        self.tb_index = 1

    def preprocess(self, pil_img):
        img = pil_img.convert("RGB")
        tensor = self.transform(img)
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, pil_img):
        x = self.preprocess(pil_img)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]

        tb_prob = float(probs[self.tb_index])
        return tb_prob, probs.cpu().numpy()

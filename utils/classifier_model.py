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

        # 1. Rebuild the model architecture
        self.model = XRayClassifier().to(self.device)

        # 2. Load state_dict (your converted PyTorch weights)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)

        self.model.eval()

        # 3. Preprocessing (matches DenseNet training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Map classes: adjust to your labels
        # Example: ["Normal", "Pneumonia", "TB"]
        self.class_names = ["Class0", "Class1", "Class2"]

        # If TB is class 2 → TB probability = softmax[:, 2]
        self.tb_index = 2

    def preprocess(self, pil_img):
        img = pil_img.convert("RGB")
        tensor = self.transform(img)
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, pil_img):
        """
        Returns TB probability as a float between 0 and 1.
        """
        x = self.preprocess(pil_img)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]

        tb_prob = float(probs[self.tb_index])
        return tb_prob, probs.cpu().numpy()

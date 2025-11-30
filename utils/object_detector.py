import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import cv2
import numpy as np


class TBObjectDetector:
    def __init__(self, model_path, device="cpu"):
        self.device = device

        # -----------------------------
        # 1. Create EXACT SAME MODEL used in training
        # -----------------------------
        backbone = torchvision.models.mobilenet_v3_large(pretrained=False).features
        backbone.out_channels = 960

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )

        self.model = FasterRCNN(
            backbone,
            num_classes=2,  # TB / background
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=512,
            max_size=512
        )

        # -----------------------------
        # 2. Load checkpoint
        # -----------------------------
        checkpoint = torch.load(model_path, map_location=device)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        print("⚠ Missing keys:", missing)
        print("⚠ Unexpected keys:", unexpected)

        self.model.to(device)
        self.model.eval()

        # -----------------------------
        # 3. Preprocessing
        # -----------------------------
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((512, 512)),
        ])

    # ----------------------------------------------------
    # Run detection on a NumPy or PIL image
    # ----------------------------------------------------
    def predict(self, image):
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_tensor = self.transform(image).to(self.device)

        with torch.no_grad():
            outputs = self.model([image_tensor])[0]

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()

        final = []
        for b, s in zip(boxes, scores):
            if s < 0.4:  # threshold
                continue
            x1, y1, x2, y2 = b
            final.append([float(x1), float(y1), float(x2), float(y2), float(s)])

        return final

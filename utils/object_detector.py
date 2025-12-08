import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from PIL import Image
import numpy as np


class TBObjectDetector:
    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)

        # 1. Recreate the *exact* baseline model from the notebook
        backbone = torchvision.models.mobilenet_v3_large(pretrained=True).features
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
            num_classes=2,  # background + TB
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

        # 2. Load the weights saved from the strategy comparison
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print("⚠ Missing keys:", missing)
        if unexpected:
            print("⚠ Unexpected keys:", unexpected)

        self.model.to(self.device)
        self.model.eval()

        # 3. Match UltraSafe dataset preprocessing:
        #    - ToTensor() → [0,1]
        #    - Then map to [-1,1] with (x - 0.5) / 0.5
        self.transform = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: (x - 0.5) / 0.5),
        ])

        # Confidence threshold (you used 0.4 in your deployment pipeline)
        self.score_thresh = 0.4

    def _to_pil(self, image):
        """Convert different image types to PIL.Image."""
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            # assume HWC, RGB or BGR — if your X-rays are grayscale, this still works
            if image.ndim == 2:
                return Image.fromarray(image)
            return Image.fromarray(image)
        raise TypeError(f"Unsupported image type: {type(image)}")

    def predict(self, image):
        """
        image: PIL.Image.Image or numpy array
        returns: list of [x1, y1, x2, y2, score]
        """
        pil_img = self._to_pil(image).convert("RGB")

        # Apply same preprocessing as in training
        img_tensor = self.transform(pil_img).to(self.device)

        with torch.no_grad():
            outputs = self.model([img_tensor])[0]

        boxes = outputs["boxes"].cpu().numpy()
        scores = outputs["scores"].cpu().numpy()

        results = []
        for b, s in zip(boxes, scores):
            if s < self.score_thresh:
                continue
            x1, y1, x2, y2 = b
            results.append([float(x1), float(y1), float(x2), float(y2), float(s)])

        return results

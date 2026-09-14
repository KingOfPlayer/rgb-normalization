"""
It is one of the preprocessing components in which the image is rotated.
"""

import os
import cv2
import sys
import numpy as np
from PIL import Image as PILImage

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from capsules.EmbeddingExtraction.src.utils.response import (
    build_clip_comparison_response,
)
from capsules.EmbeddingExtraction.src.models.PackageModel import PackageModel
from sdks.novavision.src.base.application import Application


class ClipComparison(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        # 1. Extract Inputs
        self.input_image = self.request.get_param("inputImage")

        # 2. Extract Configs
        self.normalize = self.request.get_param("Normalize") or False
        # Parse text list classes from widget
        raw_classes = self.request.get_param("Classes") or ""
        if isinstance(raw_classes, str):
            self.classes = [c.strip() for c in raw_classes.split(",") if c.strip()]
        elif isinstance(raw_classes, list):
            self.classes = raw_classes
        else:
            self.classes = []

        # Model
        self.model = self.bootstrap.get("model")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        bootstrap = {}
        application = Application()

        model_name = (
            application.get_param(config=config, name="ModelName") or "ViT-B-16"
        )
        device = application.get_param(config=config, name="Device") or "gpu"

        # Model initializationclasses
        from capsules.EmbeddingExtraction.src.classes.ModelFactory import ModelFactory

        bootstrap["model"] = ModelFactory.get_model(model_name, device)

        return bootstrap

    def compute_similarity(self, input_image, text_classes):

        input_image = input_image.value.astype(np.uint8)
        input_image = PILImage.fromarray(input_image)

        # Compute Comparison Embedding
        image_embedding = self.model.encode_image(input_image, normalize=self.normalize)
        text_embeddings = self.model.encode_text(text_classes, normalize=self.normalize)

        # 1. Compute raw similarities

        # 1. Compute raw similarities
        raw_similarities = np.dot(text_embeddings, image_embedding.squeeze())
        similarities_0_to_1 = np.clip(raw_similarities, 0.0, 1.0).tolist()

        # 2. Indices for extreme values
        max_idx = int(np.argmax(raw_similarities))
        min_idx = int(np.argmin(raw_similarities))

        # 3. Target values
        similarities = similarities_0_to_1
        max_similarity = float(similarities_0_to_1[max_idx])
        most_similar_class = str(text_classes[max_idx])
        min_similarity = float(similarities_0_to_1[min_idx])
        least_similar_class = str(text_classes[min_idx])

        # 4. Classification predictions (ranked by highest similarity)
        classification_predictions = [
            {"class": text_classes[i], "confidence": similarities_0_to_1[i]}
            for i in np.argsort(-raw_similarities)
        ]

        self.metaData = {
            "similarities": similarities,
            "max_similarity": max_similarity,
            "most_similar_class": most_similar_class,
            "min_similarity": min_similarity,
            "least_similar_class": least_similar_class,
            "classification_predictions": classification_predictions,
        }

    def run(self):
        input_image = Image.get_frame(img=self.input_image, redis_db=self.redis_db)

        self.compute_similarity(input_image=input_image, text_classes=self.classes)

        packageModel = build_clip_comparison_response(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()

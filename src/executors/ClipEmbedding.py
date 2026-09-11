import os
import cv2
import sys
import numpy as np
from PIL import Image as PILImage

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from capsules.EmbeddingExtraction.src.utils.response import build_clip_embedding_response
from capsules.EmbeddingExtraction.src.models.PackageModel import PackageModel
from sdks.novavision.src.base.application import Application


class ClipEmbedding(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        # Inputs
        self.input_data = self.request.get_param("inputData")

        # Configs
        self.normalize = self.request.get_param("Normalize") or False

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

    def run(self):
        if bool(isinstance(self.input_data, dict)):
            input_image = Image.get_frame(img=self.input_data, redis_db=self.redis_db)
            input_image = input_image.value.astype(np.uint8)
            input_image = PILImage.fromarray(input_image)
            self.embedding = self.model.encode_image(
                input_image, normalize=self.normalize
            )
        else:
            self.embedding = self.model.encode_text(
                self.input_data, normalize=self.normalize
            )
        packageModel = build_clip_embedding_response(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()

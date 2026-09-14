import os
import cv2
import sys
import numpy as np
from PIL import Image as PILImage

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from capsules.EmbeddingExtraction.src.utils.response import build_perception_encoder_response
from capsules.EmbeddingExtraction.src.utils.utils import build_bootstrap
from capsules.EmbeddingExtraction.src.models.PackageModel import PackageModel


class PerceptionEncoderEmbedding(Component):
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
        return build_bootstrap(config=config, default_model_name="PerceptionEncoderEmbedding_PE-Core-B16-224")

    def rotation(self, image):
        if self.keep_side == True:
            height, width = image.shape[:2]
            image_center = (width / 2, height / 2)
            rotation_arr = cv2.getRotationMatrix2D(image_center, self.rotation_degree, 1)
            abs_cos = abs(rotation_arr[0, 0])
            abs_sin = abs(rotation_arr[0, 1])
            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)
            rotation_arr[0, 2] += bound_w / 2 - image_center[0]
            rotation_arr[1, 2] += bound_h / 2 - image_center[1]
            img_rotation = cv2.warpAffine(image, rotation_arr, (bound_w, bound_h))

            return img_rotation

        elif self.keep_side == False:
            height, width = image.shape[:2]
            rotation_arr = cv2.getRotationMatrix2D((height / 2, width / 2), self.rotation_degree, 1)
            img_rotation = cv2.warpAffine(image, rotation_arr, (height, width))

            return img_rotation

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

        packageModel = build_perception_encoder_response(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
"""
    It is one of the preprocessing components in which the image is rotated.
"""

import os
import cv2
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from capsules.EmbeddingExtraction.src.utils.response import build_response
from capsules.EmbeddingExtraction.src.models.PackageModel import PackageModel
from sdks.novavision.src.base.application import Application


class ClipComparison(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        # 1. Extract Inputs
        self.input_image = self.request.get_param("inputImage")

        # 2. Extract Configs
        self.advance = self.request.get_param("Advance")
        self.is_advance = bool(self.advance) if self.advance is not None else False

        self.version = self.request.get_param("Version") or "ViT-B-16"
        self.device = self.request.get_param("Device") or "CPU"

        # Parse text list classes from widget
        raw_classes = self.request.get_param("Classes") or ""
        if isinstance(raw_classes, str):
            self.classes = [c.strip() for c in raw_classes.split(",") if c.strip()]
        elif isinstance(raw_classes, list):
            self.classes = raw_classes
        else:
            self.classes = []

        # 3. Pull Preloaded Assets from Bootstrap
        self.model = self.bootstrap.get("model")
        self.processor = self.bootstrap.get("processor")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        bootstrap = {}
        application = Application()

        version = application.get_param(config=config, name="Version") or "ViT-B-16"
        device = application.get_param(config=config, name="Device") or "CPU"

        # Model initialization logic
        # bootstrap["model"] = load_clip_model(version=version, device=device)
        # bootstrap["processor"] = load_clip_processor(version=version)

    def run(self):
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        img.value = self.rotation(img.value)
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        packageModel = build_response(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
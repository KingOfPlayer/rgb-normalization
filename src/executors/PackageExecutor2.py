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
from components.RgbNormalization.src.utils.response import build_response_executor_2
from components.RgbNormalization.src.models.PackageModel import PackageModel


class PackageExecutor2(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.demo_option = self.request.get_param("demoOption")
        self.image = self.request.get_param("inputImage")
        self.image2 = self.request.get_param("inputImage2")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}


    def run(self):
        # Image1
        img = Image.get_frame(img=self.image, redis_db=self.redis_db)
        self.image = Image.set_frame(img=img, package_uID=self.uID, redis_db=self.redis_db)
        # Image2
        img2 = Image.get_frame(img=self.image2, redis_db=self.redis_db)
        self.image2 = Image.set_frame(img=img2, package_uID=self.uID, redis_db=self.redis_db)
        packageModel = build_response_executor_2(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
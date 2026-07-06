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
from components.Package.src.utils.response import build_response_executor_2
from components.Package.src.models.PackageModel import PackageModel


class PackageExecutor2(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.demo_option = self.request.get_param("PackageExecutor2DemoOption")
        self.image = self.request.get_param("inputImage")
        self.image2 = self.request.get_param("inputImage2")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}


    def run(self):
        # Do Someting
        packageModel = build_response_executor_2(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
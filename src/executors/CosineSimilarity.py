import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.CosineSimilarity.src.utils.response import build_response
from components.CosineSimilarity.src.models.PackageModel import PackageModel

import numpy as np
import traceback


class CosineSimilarity(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.embedding_1 = self.request.get_param("embedding_1")
        self.embedding_2 = self.request.get_param("embedding_2")
        self.similarity = 0.0

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def calculate_similarity(self, emb1, emb2):
        vec1 = np.array(emb1, dtype=np.float32)
        vec2 = np.array(emb2, dtype=np.float32)

        if vec1.shape != vec2.shape:
            raise ValueError(f"Embeddings must have the same dimensionality. Got {vec1.shape} and {vec2.shape}")

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        sim = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(np.clip(sim, -1.0, 1.0))

    def run(self):
        self.similarity = self.calculate_similarity(self.embedding_1, self.embedding_2)
        packageModel = build_response(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()

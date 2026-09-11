from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class BaseEmbeddingModel(ABC):
    def __init__(self, model_name: str, device: str, cache_dir:str, **kwargs):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.kwargs = kwargs


        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._load()

    @abstractmethod
    def _load(self):
        pass

    @abstractmethod
    def encode_image(self, input_image: Image.Image | list[Image.Image], normalize:bool = False) -> list[float] | list[list[float]]:
        pass

    @abstractmethod
    def encode_text(self, input_text: str | list[str], normalize:bool = False) -> list[float] | list[list[float]]:
        pass


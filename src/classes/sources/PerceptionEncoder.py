from capsules.EmbeddingExtraction.src.classes.ModelSource import BaseEmbeddingSoruce
import torch
from sdks.novavision.src.base.logger import LoggerManager
from huggingface_hub import hf_hub_download

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../lib/PerceptionModels/"))

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

logger = LoggerManager()

class PerceptionEncoderSource(BaseEmbeddingSoruce):
    def _load(self):
        self.model_cache = str(self.cache_dir)

        print(
            f"(EmbeddingExtractor) Downloading PerceptionEncoder model checkpoint: {self.model_name}"
        )
        hf_hub_download(
            repo_id=f"facebook/{self.model_name}",
            filename=self.model_name + ".pt",
            local_dir=self.model_cache,
        )
        print(
            f"(EmbeddingExtractor) Loading PerceptionEncoder model: {self.model_name}"
        )
        self.model = pe.CLIP.from_config(
            self.model_name,
            pretrained=True,
            checkpoint_path=str(self.cache_dir / str(self.model_name + ".pt")),
        )
        self.model = self.model.to(
            "cuda"
            if self.device.lower() == "gpu" and torch.cuda.is_available()
            else "cpu"
        )
        self.model.eval()

        self.preprocess = transforms.get_image_transform(self.model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)
        print(f"(EmbeddingExtractor) Loaded PerceptionEncoder model: {self.model_name}")

    def encode_image(self, input_image, normalize: bool = False):
        if isinstance(input_image, list):
            image_tensor = torch.stack(
                [self.preprocess(img) for img in input_image]
            ).to(self.device)
        else:
            image_tensor = self.preprocess(input_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_tensor)
        if normalize:
            image_embedding = image_embedding / image_embedding.norm(
                dim=-1, keepdim=True
            )

        image_embedding = image_embedding.cpu().numpy().astype(float)
        return image_embedding

    def encode_text(self, input_text, normalize: bool = False):
        if isinstance(input_text, str):
            input_text = [input_text]
        text_tokens = self.tokenizer(input_text).to(self.device)
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
        if normalize:
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        text_embedding = text_embedding.cpu().numpy().astype(float)
        return text_embedding

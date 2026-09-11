from capsules.EmbeddingExtraction.src.classes.ModelSource import BaseEmbeddingModel
import torch
from sdks.novavision.src.base.logger import LoggerManager

logger = LoggerManager()

class OpenClipSoruce(BaseEmbeddingModel):
    def Bootstrap(self):
        global create_model_and_transforms, tokenizer
        # Install open_clip 
        try:
            from open_clip import create_model_and_transforms, tokenizer
        except ImportError:
            
            logger.warning("(EmbeddingExtractor) OpenClip library not found. Installing open-clip-torch")
            import subprocess
            import sys
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "open-clip-torch==3.3.0"]
            )
            from open_clip import create_model_and_transforms, tokenizer
        

    def _load(self):
        self.Bootstrap();
        logger.info(f"(EmbeddingExtractor) OpenClip Model Loading: {self.model_name} ")
        self.model, _, self.preprocess = create_model_and_transforms(
            model_name=self.model_name,
            pretrained=self.kwargs.get("pretrained", "openai"),
            device=self.device,
            cache_dir=self.cache_dir
        )
        logger.info(f"(EmbeddingExtractor) OpenClip Model Loaded: {self.model_name} ")
        self.model.eval()

        self.tokenizer = tokenizer.tokenize

    def _convert_to_float_list(self, embedding_tensor):
        if isinstance(embedding_tensor, torch.Tensor):
            return embedding_tensor.cpu().numpy().tolist()
        else:
            raise TypeError("Input must be a torch.Tensor")

    def encode_image(self, input_image, normalize:bool = False):
        if isinstance(input_image, list):
            image_tensor = torch.stack([self.preprocess(img) for img in input_image]).to(self.device)
        else:
            image_tensor = self.preprocess(input_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_tensor)
        if normalize:
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        
        image_embedding = self._convert_to_float_list(image_embedding)
        return image_embedding

    def encode_text(self, input_text, normalize:bool = False):
        if isinstance(input_text, str):
            input_text = [input_text]
        text_tokens = self.tokenizer(input_text).to(self.device)
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
        if normalize:
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        text_embedding = self._convert_to_float_list(text_embedding)
        return text_embedding
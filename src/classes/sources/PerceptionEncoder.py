from src.classes.ModelSource import BaseEmbeddingSoruce
import torch


class PerceptionEncoderSource(BaseEmbeddingSoruce):
    def _bootstrap(self):
        # Install perception-encoder library
        import subprocess
        import sys
        if not self.cache_dir.exists():

            #Clone the repository depth 1 to the cache_dir
            repo_url = "https://github.com/facebookresearch/perception_models.git"
            try: 
                print(f"Cloning the repository {repo_url} into {self.cache_dir}")
                subprocess.run(["git", "clone", "--depth", "1", repo_url, str(self.cache_dir)], check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to clone the repository: {e}")
        
        libs_dir = self.cache_dir / "libs"
        """ if requirements_path.exists():
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(self.cache_dir / "requirements.txt"), "--target", str(libs_dir)], check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to install dependencies from requirements.txt: {e}")
        else:
            raise FileNotFoundError(f"requirements.txt not found") """
        
        sys.path.append(str(self.cache_dir))
        sys.path.append(str(libs_dir))
        print(f"Library ready to use.")

    def _load(self):
        self._bootstrap()
        
        from huggingface_hub import hf_hub_download
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        self.model_cache = str(self.cache_dir / "checkpoints")

        print(f"(PerceptionEncoderEmbedding) Downloading PerceptionEncoderEmbedding model checkpoint: {self.model_name}")
        hf_hub_download(
            repo_id=f"facebook/{self.model_name}",
            filename= self.model_name + ".pt",
            local_dir=self.model_cache,
        )
        print(f"(PerceptionEncoderEmbedding) Downloaded model checkpoint: {self.model_name}")
        print(f"(PerceptionEncoderEmbedding) Loading PerceptionEncoderEmbedding model: {self.model_name}")
        self.model = pe.CLIP.from_config("PE-Core-B16-224", pretrained=True, checkpoint_path=str(self.cache_dir / "checkpoints" / str(self.model_name + ".pt")))
        self.model = self.model.to("cuda" if self.device.lower() == "gpu" and torch.cuda.is_available() else "cpu")
        self.model.eval()

        self.preprocess = transforms.get_image_transform(self.model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)
        print(f"(PerceptionEncoderEmbedding) Loaded PerceptionEncoderEmbedding model: {self.model_name}")
        

    def encode_image(self, input_image, normalize:bool = False):
        if isinstance(input_image, list):
            image_tensor = torch.stack([self.preprocess(img) for img in input_image]).to(self.device)
        else:
            image_tensor = self.preprocess(input_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_tensor)
        if normalize:
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        
        return image_embedding

    def encode_text(self, input_text, normalize:bool = False):
        if isinstance(input_text, str):
            input_text = [input_text]
        text_tokens = self.tokenizer(input_text).to(self.device)
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
        if normalize:
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        return text_embedding
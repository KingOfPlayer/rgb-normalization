from pathlib import Path

class ModelFactory:
    @staticmethod
    def get_model(model_name: str, device: str, cache_dir:str="/storage/embeding/",**kwargs):
        base_cache_path = Path(cache_dir)
        factory_name, model_name = model_name.split("_", 1)
        if factory_name.startswith("OpenClip"):
            from capsules.EmbeddingExtraction.src.classes.sources.OpenClip import OpenClipSoruce
            return OpenClipSoruce(model_name=model_name, device=device, cache_dir=base_cache_path / "OpenClip",  **kwargs)
        # For feature expansion, you can add more model sources here. For example:
        #if source_name == "<name_of_another_model_source>":
        #    from classes.sources.<another_model_source> import <AnotherModelSource>
        #    return <AnotherModelSource>(model_name=model_name, device=device, cache_dir=base_cache_path / <name_of_another_model_source> **kwargs)
    
        raise ValueError(f"Unknown model source: {factory_name}")
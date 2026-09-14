
from sdks.novavision.src.base.application import Application

def build_bootstrap(config: dict, default_model_name: str ) -> dict:
    bootstrap = {}
    application = Application()

    model_name = (
        application.get_param(config=config, name="ModelName") or default_model_name
    )
    device = application.get_param(config=config, name="Device") or "gpu"

    # Model initializationclasses
    from capsules.EmbeddingExtraction.src.classes.ModelFactory import ModelFactory

    bootstrap["model"] = ModelFactory.get_model(model_name, device)

    return bootstrap
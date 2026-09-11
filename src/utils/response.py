from capsules.EmbeddingExtraction.src.models.PackageModel import (
    ClipEmbeddingExecutor,
    ClipEmbeddingOutputs,
    ClipEmbeddingResponse,
    ConfigExecutor,
    OutputEmbedding,
    PackageConfigs,
    PackageModel,
)
from sdks.novavision.src.helper.package import PackageHelper


def build_clip_embedding_response(context):
    outputEmbedding = OutputEmbedding(value=context.embedding)
    Outputs = ClipEmbeddingOutputs(outputEmbedding=outputEmbedding)
    packageResponse = ClipEmbeddingResponse(outputs=Outputs)
    packageExecutor = ClipEmbeddingExecutor(value=packageResponse)
    executor = ConfigExecutor(value=packageExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

from sdks.novavision.src.helper.package import PackageHelper

from capsules.EmbeddingExtraction.src.models.PackageModel import (
    ClipEmbeddingExecutor,
    ClipEmbeddingOutputs,
    ClipEmbeddingResponse,
    ClipComparisonExecutor,
    ClipComparisonOutputs,
    ClipComparisonResponse,
    PerceptionEncoderExecutor,
    PerceptionEncoderOutputs,
    PerceptionEncoderResponse,
    ConfigExecutor,
    OutputEmbedding,
    OutputMetadata,
    PackageConfigs,
    PackageModel,
)


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


def build_clip_comparison_response(context):
    outputMetadata = OutputMetadata(value=context.similarity_results)
    Outputs = ClipComparisonOutputs(outputMetadata=outputMetadata)
    packageResponse = ClipComparisonResponse(outputs=Outputs)
    packageExecutor = ClipComparisonExecutor(value=packageResponse)
    executor = ConfigExecutor(value=packageExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel


def build_perception_encoder_response(context):
    outputEmbedding = OutputEmbedding(value=context.embedding)
    Outputs = PerceptionEncoderOutputs(outputEmbedding=outputEmbedding)
    packageResponse = PerceptionEncoderResponse(outputs=Outputs)
    packageExecutor = PerceptionEncoderExecutor(value=packageResponse)
    executor = ConfigExecutor(value=packageExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

from sdks.novavision.src.helper.package import PackageHelper
from ..models.PackageModel import (
    PackageModel, 
    PackageConfigs, 
    ConfigExecutor, 
    CosineSimilarityOutputs, 
    CosineSimilarityResponse, 
    CosineSimilarityExecutor, 
    OutputSimilarity
)

def build_response(context):
    outputSimilarity = OutputSimilarity(value=context.similarity)
    Outputs = CosineSimilarityOutputs(similarity=outputSimilarity)
    packageResponse = CosineSimilarityResponse(outputs=Outputs)
    packageExecutor = CosineSimilarityExecutor(value=packageResponse)
    executor = ConfigExecutor(value=packageExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

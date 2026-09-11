
from sdks.novavision.src.helper.package import PackageHelper
from capsules.EmbeddingExtraction.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, PackageOutputs, PackageResponse, PackageExecutor, OutputImage


def build_response(context):
    outputImage = OutputImage(value=context.image)
    Outputs = PackageOutputs(outputImage=outputImage)
    packageResponse = PackageResponse(outputs=Outputs)
    packageExecutor = PackageExecutor(value=packageResponse)
    executor = ConfigExecutor(value=packageExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel
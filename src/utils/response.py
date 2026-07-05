
from models.PackageModel import PackageExecutor1, PackageExecutor1Outputs, PackageExecutor1Response, PackageExecutor2Outputs
from sdks.novavision.src.helper.package import PackageHelper
from components.Package.src.models.PackageModel import PackageModel, PackageConfigs, ConfigExecutor, PackageOutputs, PackageResponse, PackageExecutor, OutputImage

def build_response_executor_1(context):
    outputImage = OutputImage(value=context.image)
    Outputs = PackageExecutor1Outputs(outputImage=outputImage)

    packageResponse = PackageExecutor1Response(outputs=Outputs)
    packageExecutor = PackageExecutor1(value=packageResponse)
    executor = ConfigExecutor(value=packageExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel

def build_response_executor_2(context):
    outputImage = OutputImage(value=context.image)
    outputImage2 = OutputImage(value=context.image2)
    Outputs = PackageExecutor2Outputs(outputImage=outputImage,outputImage2=outputImage2)

    packageResponse = PackageExecutor1Response(outputs=Outputs)
    packageExecutor = PackageExecutor1(value=packageResponse)
    executor = ConfigExecutor(value=packageExecutor)
    packageConfigs = PackageConfigs(executor=executor)
    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    packageModel = package.build_model(context)
    return packageModel
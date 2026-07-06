
from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, Config

# Image Class

class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image],Image]
    type: str = "object"

    class Config:
        title  = "Image"

class OutputImage(Output):
    name: Literal["outputImage"] = "outputImage"
    value: Union[List[Image],Image]
    type: str = "object"

    class Config:
        title  = "Image"

class InputImage2(Input):
    name: Literal["inputImage2"] = "inputImage2"
    value: Union[List[Image],Image]
    type: str = "object"

    class Config:
        title  = "Image"

class OutputImage2(Output):
    name: Literal["outputImage2"] = "outputImage2"
    value: Union[List[Image],Image]
    type: str = "object"

    class Config:
        title  = "Image"

# Config Field
class PackageExecutor1DemoOption1(Config):
    name: Literal["True"] = "True"
    value: Literal["False"] = "False"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Executor 1 Option False"


class PackageExecutor1DemoOption2(Config):
    name: Literal["True"] = "True"
    value: Literal["True"] = "True"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Executor 1 Option True"
    

class PackageExecutor1DemoOption(Config):
    """
        Executor 1 Demo Option
    """
    name: Literal["PackageExecutor1DemoOption"] = "PackageExecutor1DemoOption"
    value: Union[PackageExecutor1DemoOption1, PackageExecutor1DemoOption2]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Demo Option 1"
        json_schema_extra = {
            "shortDescription": "Executor 1 Demo Option"
        }


class PackageExecutor2DemoOption1(Config):
    name: Literal["True"] = "True"
    value: Literal["False"] = "False"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Executor 2 Option False"


class PackageExecutor2DemoOption2(Config):
    name: Literal["True"] = "True"
    value: Literal["True"] = "True"
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Executor 2 Option True"


class PackageExecutor2DemoOption(Config):
    """
        Executor 2 Demo Option
    """
    name: Literal["PackageExecutor1DemoOption"] = "PackageExecutor1DemoOption"
    value: Union[PackageExecutor2DemoOption1, PackageExecutor2DemoOption2]
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Demo Option 2"
        json_schema_extra = {
            "shortDescription": "Executor 2 Demo Option"
        }

# Inputs
class PackageExecutor1Inputs(Inputs):
    inputImage: InputImage

class PackageExecutor2Inputs(Inputs):
    inputImage: InputImage
    inputImage2: InputImage2

# Input Configs
class PackageExecutor1Configs(Configs):
    demoOption: PackageExecutor1DemoOption

class PackageExecutor2Configs(Configs):
    demoOption: PackageExecutor2DemoOption

# Output
class PackageExecutor1Outputs(Outputs):
    outputImage: OutputImage

class PackageExecutor2Outputs(Outputs):
    outputImage: OutputImage
    outputImage2: OutputImage2

# Executor Requests
class PackageExecutor1Request(Request):
    inputs: Optional[PackageExecutor1Inputs] = None
    configs: PackageExecutor1Configs

    class Configs:
        json_schema_extra = {
            "target": "configs"
        }

class PackageExecutor2Request(Request):
    inputs: Optional[PackageExecutor2Inputs] = None
    configs: PackageExecutor2Configs

    class Configs:
        json_schema_extra = {
            "target": "configs"
        }

# Executor Responses
class PackageExecutor1Response(Response):
    outputs: PackageExecutor1Outputs

class PackageExecutor2Response(Response):
    outputs: PackageExecutor2Outputs

# Executor Configs
class PackageExecutor1(Config):
    """
        PackageExecutor1
    """
    name: Literal["PackageExecutor1"] = "PackageExecutor1"
    value: Union[PackageExecutor1Request, PackageExecutor1Response]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "PackageExecutor1"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }

class PackageExecutor2(Config):
    """
        PackageExecutor2
    """
    name: Literal["PackageExecutor2"] = "PackageExecutor2"
    value: Union[PackageExecutor2Request, PackageExecutor2Response]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "PackageExecutor2"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }

# Root Package Configs
class ConfigExecutor(Config):
    """
        ConfigExecutor
    """
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[PackageExecutor1, PackageExecutor2]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["RgbNormalization"] = "RgbNormalization"

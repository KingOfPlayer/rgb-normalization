
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

# Config Field
class PackageExecutor1DemoOption1(Config):
    name: Literal["True"] = "True"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Option False"


class PackageExecutor1DemoOption2(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Option True"
    

class PackageExecutor1DemoOption(Config):
    """
        Demo Option
    """
    name: Literal["PackageExecutor1DemoOption"] = "PackageExecutor1DemoOption"
    value: Union[PackageExecutor1DemoOption1, PackageExecutor1DemoOption2]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

# Inputs
class PackageExecutor1Inputs(Inputs):
    inputImage: InputImage

# Input Configs
class PackageExecutor1Configs(Configs):
    demoOption: PackageExecutor1DemoOption

# Requests
class PackageExecutor1Request(Request):
    inputs: Optional[PackageExecutor1Inputs]
    configs: PackageExecutor1Configs

    class Configs:
        json_schema_extra = {
            "target": "value"
        }

# Executor Configs
class PackageExecutor1(Config):
    name: Literal["Package"] = "PackageExecutor1"
    value: Union[PackageExecutor1Request, PackageExecutor1Response]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Package"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }

# Root Package Configs
class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[PackageExecutor1]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["DemoPackage"] = "DemoPackage"

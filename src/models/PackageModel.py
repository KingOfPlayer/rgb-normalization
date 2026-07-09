
from pydantic import Field, create_model
from pydantic._internal._model_construction import ModelMetaclass
from typing import List, Optional, Union, Literal, Dict, Any
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, Config

# region Dynamic
# Base definations 
class BaseInputImage(Input):
    name: Literal["baseInputImage"] = "baseInputImage"
    value: Union[List[Image], Image]
    type: str = "object"
    class Config: title = "Image"

class BaseOutputImage(Output):
    name: Literal["baseOutputImage"] = "baseOutputImage"
    value: Union[List[Image], Image]
    type: str = "object"
    class Config: title = "Image"

def _make_image_model(base_cls, field_name: str):
    """
    Create a subclass of base_cls (BaseInputImage or BaseOutputImage)
    whose `name` Literal is pinned to field_name, matching the key
    it will be assigned to on the parent model.
    """
    return create_model(
        f"{base_cls.__name__}_{field_name}",
        __base__=base_cls,
        name=(Literal[field_name], field_name),
    )

# Config
class InputOutputCount(Config):
    """
        "Input Output Count"
    """
    name: Literal["InputOutputCount"] = "InputOutputCount"
    value: int = Field(default=1, ge=1, le=5)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Input Output Count"

class DynamicFieldsMetaclass(ModelMetaclass):
    def __new__(mcs, name, bases, namespace, **kwargs):
        annotations = namespace.get("__annotations__", {})

        # Pull the default count instead of hardcoding range(1, 6)
        default_count = InputOutputCount.model_fields["value"].default  # pydantic v2
        # default_count = InputOutputCount.__fields__["value"].default  # pydantic v1

        if "Inputs" in name or name == "ExecutorDynamicInputs":
            base_input_cls = _make_image_model(BaseInputImage, "baseInputImage")
            annotations["baseInputImage"] = base_input_cls
            namespace["baseInputImage"] = Field(default_factory=base_input_cls)

            for i in range(1, default_count + 1):
                field_name = f"baseOutputImage{i}"
                cls_i = _make_image_model(BaseInputImage, field_name)
                annotations[field_name] = cls_i
                namespace[field_name] = Field(default_factory=cls_i)

        elif "Outputs" in name or name == "ExecutorDynamicOutputs":
            base_output_cls = _make_image_model(BaseOutputImage, "baseOutputImage")
            annotations["baseOutputImage"] = base_output_cls
            namespace["baseOutputImage"] = Field(default_factory=base_output_cls)

            for i in range(1, default_count + 1):
                field_name = f"baseOutputImage{i}"
                cls_i = _make_image_model(BaseOutputImage, field_name)
                annotations[field_name] = cls_i
                namespace[field_name] = Field(default_factory=cls_i)

        namespace["__annotations__"] = annotations
        return super().__new__(mcs, name, bases, namespace, **kwargs)


# Apply the dynamic compilation metaclass onto your target structures
class ExecutorDynamicInputs(Inputs, metaclass=DynamicFieldsMetaclass):
    #baseInputImage: BaseInputImage (Example, must be field name same as BaseInputImage's name field) 
    pass

class ExecutorDynamicOutputs(Outputs, metaclass=DynamicFieldsMetaclass):
    #baseOutputImage: BaseOutputImage (Example, must be field name same as BaseOutputImage's name field) 
    pass

class ExecutorDynamicConfigs(Configs):
    inputOutputCount: InputOutputCount

class ExecutorDynamicRequest(Request):
    inputs: Optional[ExecutorDynamicInputs]
    configs: ExecutorDynamicConfigs
    class Configs: json_schema_extra = {"target": "configs"}

class ExecutorDynamicResponse(Response):
    outputs: ExecutorDynamicOutputs

class ExecutorDynamic(Config):
    name: Literal["ExecutorDynamic"] = "ExecutorDynamic"
    value: Union[ExecutorDynamicRequest, ExecutorDynamicResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"
    class Config:
        title = "ExecutorDynamic"
        json_schema_extra = {"target": {"value": 0}}
# endregion Dynamic


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
    name: Literal["False"] = "False"
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
    name: Literal["False"] = "False"
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
    name: Literal["PackageExecutor2DemoOption"] = "PackageExecutor2DemoOption"
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
    value: Union[PackageExecutor1, PackageExecutor2, ExecutorDynamic]
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

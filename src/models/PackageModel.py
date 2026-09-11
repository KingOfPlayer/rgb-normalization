from typing import Literal

from pydantic import ConfigDict, field_validator, ValidationInfo
from sdks.novavision.src.base.model import (
    Config,
    Configs,
    Image,
    Input,
    Inputs,
    Output,
    Outputs,
    Package,
    Request,
    Response,
)

# region General Classes


class ConfigEnable(Config):
    name: Literal["True"] = "True"
    value: Literal[True] = True
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Enable"


class ConfigDisable(Config):
    name: Literal["False"] = "False"
    value: Literal[False] = False
    type: Literal["bool"] = "bool"
    field: Literal["option"] = "option"

    class Config:
        title = "Disable"


class BaseOption(Config):
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"


# Dynamic class generation for options


def make_option(label: str, val: str | None = None) -> type[BaseOption]:
    val = val or label
    clean_identifier = label.replace("-", "").replace(" ", "")

    return type(
        f"Option{clean_identifier}",
        (BaseOption,),
        {
            "__annotations__": {
                "name": Literal[val],
                "value": Literal[val],
            },
            "name": val,
            "value": val,
            "model_config": ConfigDict(title=label),
        },
    )


class BaseDropdown(Config):
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"


class BaseDependentDropdown(Config):
    type: Literal["object"] = "object"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"
    restart: Literal[True] = True


# endregion

# region Base Classes


# region List Of Models
# title, value: <factory_name>_<model_name>
OptionViTB16 = make_option("ViT-B-16", "OpenClip_ViT-B-16")
OptionViTB32 = make_option("ViT-B-32", "OpenClip_ViT-B-32")
OptionRN50 = make_option("RN50", "OpenClip_RN50")
# endregion


class BaseModels(BaseDropdown):
    name: Literal["ModelName"] = "ModelName"
    value: OptionViTB16 | OptionViTB32 | OptionRN50  # pyright: ignore

    class Config:
        title = "Model"


class BaseNormalize(BaseDropdown):
    name: Literal["Normalize"] = "Normalize"
    value: ConfigDisable | ConfigEnable

    class Config:
        title = "Normalize Embedding"


OptionCPU = make_option("CPU", "cpu")
OptionGPU = make_option("GPU", "gpu")


class BaseDevice(BaseDropdown):
    name: Literal["Device"] = "Device"
    value: OptionCPU | OptionGPU  # pyright: ignore

    class Config:
        title = "Device"


# endregion

# region Input-Output


class InputData(Input):
    name: Literal["inputData"] = "inputData"
    value: Image | str
    type: str = "object"

    @field_validator("type", mode="before")
    @classmethod
    def set_type_based_on_value(cls, v, info: ValidationInfo):
        val = info.data.get("value") if info.data else None
        if isinstance(val, str):
            return "str"
        return "object"  # Return the string "object", NOT the Image instance

    class Config:
        title = "Image / Text"


class OutputEmbedding(Output):
    name: Literal["outputEmbedding"] = "outputEmbedding"
    value: list[float] | list[list[float]]
    type: Literal["list"] = "list"

    class Config:
        title = "Embedding"


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Image
    type: Literal["object"] = "object"

    class Config:
        title = "Image"


class OutputMeta(Output):
    name: Literal["outputMeta"] = "outputMeta"
    value: dict
    type: Literal["object"] = "object"

    class Config:
        title = "Meta"


# endregion

# region Configs
# region ClipEmbedding


class ConfigClipEmbeddingAdvanceEnable(ConfigEnable):
    configClipEmbeddingModelName: BaseModels
    configClipEmbeddingNormalize: BaseNormalize
    configClipEmbeddingDevice: BaseDevice


class ConfigClipEmbeddingAdvance(BaseDependentDropdown):
    name: Literal["Advance"] = "Advance"
    value: ConfigClipEmbeddingAdvanceEnable | ConfigDisable

    class Config:
        title = "Advance"


# endregion
# region ClipComparison


class ConfigClipComparisonClasses(Config):
    name: Literal["Classes"] = "Classes"
    value: str
    type: Literal["string"] = "string"
    field: Literal["widget"] = "widget"

    class Config:
        title = "List of classes"
        json_schema_extra = {
            "class": "\\novavision\\app\\widgets\\TextList",
            "shortDescription": "List of text entries",
        }


class ConfigClipComparisonAdvanceEnable(ConfigEnable):
    configClipComparisonModelName: BaseModels
    configClipComparisonClasses: ConfigClipComparisonClasses
    configClipComparisonDevice: BaseDevice


class ConfigClipComparisonAdvance(BaseDependentDropdown):
    name: Literal["Advance"] = "Advance"
    value: ConfigClipComparisonAdvanceEnable | ConfigDisable

    class Config:
        title = "Advance"


# endregion
# region PerceptionEncoder


class ConfigPerceptionEncoderAdvanceEnable(ConfigEnable):
    configPerceptionEncoderModelName: BaseModels
    configPerceptionEncoderNormalize: BaseNormalize
    configPerceptionEncoderDevice: BaseDevice


class ConfigPerceptionEncoderAdvance(BaseDependentDropdown):
    name: Literal["Advance"] = "Advance"
    value: ConfigPerceptionEncoderAdvanceEnable | ConfigDisable

    class Config:
        title = "Advance"


# endregion
# endregion

# region Inputs-Ouputs-Configs


class ClipEmbeddingInputs(Inputs):
    inputData: InputData


class ClipEmbeddingOutputs(Outputs):
    outputEmbedding: OutputEmbedding


class ClipEmbeddingConfigs(Configs):
    configClipEmbeddingAdvance: ConfigClipEmbeddingAdvance


class ClipComparisonInputs(Inputs):
    inputImage: InputImage


class ClipComparisonOutputs(Outputs):
    outputMeta: OutputMeta


class ClipComparisonConfigs(Configs):
    configClipComparisonAdvance: ConfigClipComparisonAdvance


class PerceptionEncoderInputs(Inputs):
    inputData: InputData


class PerceptionEncoderOutputs(Outputs):
    outputEmbedding: OutputEmbedding


class PerceptionEncoderConfigs(Configs):
    configPerceptionEncoderAdvance: ConfigPerceptionEncoderAdvance


# endregion

# region Request-Response


class ClipEmbeddingRequest(Request):
    inputs: ClipEmbeddingInputs
    configs: ClipEmbeddingConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class ClipEmbeddingResponse(Response):
    outputs: ClipEmbeddingOutputs


class ClipComparisonRequest(Request):
    inputs: ClipComparisonInputs
    configs: ClipComparisonConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class ClipComparisonResponse(Response):
    outputs: ClipComparisonOutputs


class PerceptionEncoderRequest(Request):
    inputs: PerceptionEncoderInputs
    configs: PerceptionEncoderConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class PerceptionEncoderResponse(Response):
    outputs: PerceptionEncoderOutputs


# endregion

# region Executors


class ClipEmbeddingExecutor(Config):
    name: Literal["ClipEmbedding"] = "ClipEmbedding"
    value: ClipEmbeddingRequest | ClipEmbeddingResponse
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Clip Embedding"
        json_schema_extra = {"target": {"value": 0}}


class ClipComparisonExecutor(Config):
    name: Literal["ClipComparison"] = "ClipComparison"
    value: ClipComparisonRequest | ClipComparisonResponse
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Clip Comparison"
        json_schema_extra = {"target": {"value": 0}}


class PerceptionEncoderExecutor(Config):
    name: Literal["PerceptionEncoder"] = "PerceptionEncoder"
    value: PerceptionEncoderRequest | PerceptionEncoderResponse
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Perception Encoder"
        json_schema_extra = {"target": {"value": 0}}


# endregion

# region Root Config


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: ClipEmbeddingExecutor | ClipComparisonExecutor | PerceptionEncoderExecutor
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"
    name: Literal["EmbeddingExtraction"] = "EmbeddingExtraction"


# endregion

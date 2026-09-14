from typing import Literal

from pydantic import ConfigDict, ValidationInfo, field_validator
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


class OutputMetadata(Output):
    name: Literal["outputMetadata"] = "outputMetadata"
    value: dict
    type: Literal["object"] = "object"

    class Config:
        title = "Metadata"


# endregion

# region Configs
# region Clip Model List
# title, value: <factory_name>_<model_name>
OptionRN50 = make_option("RN50", "OpenClip_RN50")
OptionViTB16 = make_option("ViT-B-16", "OpenClip_ViT-B-16")
OptionViTB32 = make_option("ViT-B-32", "OpenClip_ViT-B-32")


class ClipModels(BaseDropdown):
    name: Literal["ModelName"] = "ModelName"
    value: OptionRN50 | OptionViTB16 | OptionViTB32  # pyright: ignore

    class Config:
        title = "Model"


# endregion

# region ClipEmbedding


class ConfigClipEmbeddingAdvanceEnable(ConfigEnable):
    configClipEmbeddingModelName: ClipModels
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
    configClipComparisonModelName: ClipModels
    configClipComparisonClasses: ConfigClipComparisonClasses
    configClipComparisonDevice: BaseDevice


class ConfigClipComparisonAdvance(BaseDependentDropdown):
    name: Literal["Advance"] = "Advance"
    value: ConfigClipComparisonAdvanceEnable | ConfigDisable

    class Config:
        title = "Advance"


# endregion
# region PerceptionEncoderEmbedding
# region Clip Model List
# title, value: <factory_name>_<model_name>
OptionPECoreS16384 = make_option(
    "PE-Core-S16-384", "PerceptionEncoderEmbedding_PE-Core-S16-384"
)
OptionPECoreB162246 = make_option(
    "PE-Core-B16-224", "PerceptionEncoderEmbedding_PE-Core-B16-224"
)
OptionPECoreL14336 = make_option(
    "PE-Core-L14-336", "PerceptionEncoderEmbedding_PE-Core-L14-336"
)


class PerceptionEncoderEmbeddingModels(BaseDropdown):
    name: Literal["ModelName"] = "ModelName"
    value: OptionPECoreS16384 | OptionPECoreB162246 | OptionPECoreL14336  # pyright: ignore

    class Config:
        title = "Model"


# endregion


class ConfigPerceptionEncoderEmbeddingAdvanceEnable(ConfigEnable):
    configPerceptionEncoderEmbeddingModelName: PerceptionEncoderEmbeddingModels
    configPerceptionEncoderEmbeddingNormalize: BaseNormalize
    configPerceptionEncoderEmbeddingDevice: BaseDevice


class ConfigPerceptionEncoderEmbeddingAdvance(BaseDependentDropdown):
    name: Literal["Advance"] = "Advance"
    value: ConfigPerceptionEncoderEmbeddingAdvanceEnable | ConfigDisable

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
    outputMetadata: OutputMetadata


class ClipComparisonConfigs(Configs):
    configClipComparisonAdvance: ConfigClipComparisonAdvance


class PerceptionEncoderEmbeddingInputs(Inputs):
    inputData: InputData


class PerceptionEncoderEmbeddingOutputs(Outputs):
    outputEmbedding: OutputEmbedding


class PerceptionEncoderEmbeddingConfigs(Configs):
    configPerceptionEncoderEmbeddingAdvance: ConfigPerceptionEncoderEmbeddingAdvance


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


class PerceptionEncoderEmbeddingRequest(Request):
    inputs: PerceptionEncoderEmbeddingInputs
    configs: PerceptionEncoderEmbeddingConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class PerceptionEncoderEmbeddingResponse(Response):
    outputs: PerceptionEncoderEmbeddingOutputs


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


class PerceptionEncoderEmbeddingExecutor(Config):
    name: Literal["PerceptionEncoderEmbedding"] = "PerceptionEncoderEmbedding"
    value: PerceptionEncoderEmbeddingRequest | PerceptionEncoderEmbeddingResponse
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Perception Encoder Embedding"
        json_schema_extra = {"target": {"value": 0}}


# endregion

# region Root Config


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: (
        ClipEmbeddingExecutor
        | ClipComparisonExecutor
        | PerceptionEncoderEmbeddingExecutor
    )
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

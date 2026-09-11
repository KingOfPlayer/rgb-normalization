from typing import Literal

from pydantic import ConfigDict
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
OptionViTB16 = make_option("ViT-B-16")
OptionViTB32 = make_option("ViT-B-32")
OptionRN50 = make_option("RN50")
# endregion


class BaseModelVersions(BaseDropdown):
    name: Literal["Version"] = "Version"
    value: OptionViTB16 | OptionViTB32 | OptionRN50  # pyright: ignore

    class Config:
        title = "Version"


class BaseNormalize(BaseDropdown):
    name: Literal["Normalize"] = "Normalize"
    value: ConfigDisable | ConfigEnable

    class Config:
        title = "Normalize Embedding"


OptionCPU = make_option("CPU")
OptionGPU = make_option("GPU")


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

    class Config:
        title = "Image / Text"


class OutputEmbedding(Output):
    name: Literal["outputEmbedding"] = "outputEmbedding"
    value: list[float]
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
# region ClipGenerate


class ConfigClipGenerateAdvanceEnable(ConfigEnable):
    configClipGenerateVersion: BaseModelVersions
    configClipGenerateNormalize: BaseNormalize
    configClipGenerateDevice: BaseDevice


class ConfigClipGenerateAdvance(BaseDependentDropdown):
    name: Literal["Advance"] = "Advance"
    value: ConfigClipGenerateAdvanceEnable | ConfigDisable

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
    configClipComparisonVersion: BaseModelVersions
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
    configPerceptionEncoderVersion: BaseModelVersions
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


class ClipGenerateInputs(Inputs):
    inputData: InputData


class ClipGenerateOutputs(Outputs):
    outputEmbedding: OutputEmbedding


class ClipGenerateConfigs(Configs):
    configClipGenerateAdvance: ConfigClipGenerateAdvance


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


class ClipGenerateRequest(Request):
    inputs: ClipGenerateInputs
    configs: ClipGenerateConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class ClipGenerateResponse(Response):
    outputs: ClipGenerateOutputs


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


class ClipGenerateExecutor(Config):
    name: Literal["ClipGenerate"] = "ClipGenerate"
    value: ClipGenerateRequest | ClipGenerateResponse
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Clip Generate"
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
    value: ClipGenerateExecutor | ClipComparisonExecutor | PerceptionEncoderExecutor
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

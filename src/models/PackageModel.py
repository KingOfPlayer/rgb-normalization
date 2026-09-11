
from pydantic import ConfigDict
from typing import List, Union, Literal
from sdks.novavision.src.base.model import Package, Image, Inputs, Configs, Outputs, Response, Request, Output, Input, Config

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

# region list of models


OptionViTB16 = make_option("ViT-B-16")
OptionViTB32 = make_option("ViT-B-32")
OptionRN50 = make_option("RN50")
# endregion


class BaseModelVersions(Config):
    name: Literal["Version"] = "Version"
    value: Union[OptionViTB16, OptionViTB32, OptionRN50]  # pyright: ignore
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Version"


class BaseNormalize(Config):
    name: Literal["Normalize"] = "Normalize"
    value: Union[ConfigDisable, ConfigEnable]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Normalize Embedding"
# endregion

# region Input-Output


class InputData(Input):
    name: Literal["inputData"] = "inputData"
    value: Union[Image, str]
    type: str = "object"

    class Config:
        title = "Image / Text"


class OutputEmbedding(Output):
    name: Literal["outputEmbedding"] = "outputEmbedding"
    value: List[float]
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


class ConfigClipGenerateVersion(BaseModelVersions):
    pass


class ConfigClipGenerateNormalize(BaseNormalize):
    pass


class ConfigClipGenerateAdvanceEnable(ConfigEnable):
    configClipGenerateVersion: ConfigClipGenerateVersion
    configClipGenerateNormalize: ConfigClipGenerateNormalize


class ConfigClipGenerateAdvance(BaseDependentDropdown):
    name: Literal["Advance"] = "Advance"
    value: Union[ConfigClipGenerateAdvanceEnable, ConfigDisable]

    class Config:
        title = "Advance"
# endregion
# region ClipComparison


class ConfigClipComparisonVersion(BaseModelVersions):
    pass


class ConfigClipComparisonClasses(Config):
    name: Literal["Classes"] = "Classes"
    value: str
    type: Literal["string"] = "string"
    field: Literal["widget"] = "widget"

    class Config:
        title = "List of classes"
        json_schema_extra = {
            "class": "\\novavision\\app\\widgets\\TextList",
            "shortDescription": "List of text entries"
        }


class ConfigClipComparisonAdvanceEnable(ConfigEnable):
    configClipComparisonVersion: ConfigClipGenerateVersion
    configClipComparisonClasses: ConfigClipComparisonClasses


class ConfigClipComparisonAdvance(BaseDependentDropdown):
    name: Literal["Advance"] = "Advance"
    value: Union[ConfigClipComparisonAdvanceEnable, ConfigDisable]

    class Config:
        title = "Advance"
# endregion
# region PerceptionEncoder


class ConfigPerceptionEncoderVersion(BaseModelVersions):
    pass


class ConfigPerceptionEncoderNormalize(BaseNormalize):
    pass


class ConfigPerceptionEncoderAdvanceEnable(ConfigEnable):
    configPerceptionEncoderVersion: ConfigPerceptionEncoderVersion
    configPerceptionEncoderNormalize: ConfigPerceptionEncoderNormalize


class ConfigPerceptionEncoderAdvance(BaseDependentDropdown):
    name: Literal["Advance"] = "Advance"
    value: Union[ConfigPerceptionEncoderAdvanceEnable, ConfigDisable]

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
        json_schema_extra = {
            "target": "configs"
        }


class ClipGenerateResponse(Response):
    outputs: ClipGenerateOutputs


class ClipComparisonRequest(Request):
    inputs: ClipComparisonInputs
    configs: ClipComparisonConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class ClipComparisonResponse(Response):
    outputs: ClipComparisonOutputs


class PerceptionEncoderRequest(Request):
    inputs: PerceptionEncoderInputs
    configs: PerceptionEncoderConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class PerceptionEncoderResponse(Response):
    outputs: PerceptionEncoderOutputs
# endregion

# region Executors


class ClipGenerateExecutor(Config):
    name: Literal["ClipGenerate"] = "ClipGenerate"
    value: Union[ClipGenerateRequest, ClipGenerateResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Clip Generate"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ClipComparisonExecutor(Config):
    name: Literal["ClipComparison"] = "ClipComparison"
    value: Union[ClipComparisonRequest, ClipComparisonResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Clip Comparison"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class PerceptionEncoderExecutor(Config):
    name: Literal["PerceptionEncoder"] = "PerceptionEncoder"
    value: Union[PerceptionEncoderRequest, PerceptionEncoderResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Perception Encoder"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }
# endregion

# region Root Config


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[
        ClipGenerateExecutor,
        ClipComparisonExecutor,
        PerceptionEncoderExecutor]
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

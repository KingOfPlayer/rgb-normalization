from pydantic import Field
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Inputs, Configs, Outputs, Response, Request, Output, Input, Config

class BaseEmbeddingInput(Input):
    value: list[float]
    type: Literal["list"] = "list"

    class Config:
        title = "Embedding"
        

class InputEmbedding1(BaseEmbeddingInput):
    name: Literal["embedding1"] = "embedding1"

class InputEmbedding2(BaseEmbeddingInput):
    name: Literal["embedding2"] = "embedding2"

class OutputSimilarity(Output):
    name: Literal["similarity"] = "similarity"
    value: float
    type: Literal["number"] = "number"

    class Config:
        title = "Similarity"


class CosineSimilarityInputs(Inputs):
    embedding1: InputEmbedding1
    embedding2: InputEmbedding2


class CosineSimilarityOutputs(Outputs):
    similarity: OutputSimilarity


class CosineSimilarityRequest(Request):
    inputs: CosineSimilarityInputs


class CosineSimilarityResponse(Response):
    outputs: CosineSimilarityOutputs


class CosineSimilarityExecutor(Config):
    name: Literal["CosineSimilarity"] = "CosineSimilarity"
    value: CosineSimilarityRequest | CosineSimilarityResponse
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Cosine Similarity"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: CosineSimilarityExecutor
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
    name: Literal["ErkanTestPackage"] = "ErkanTestPackage"

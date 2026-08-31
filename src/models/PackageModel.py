from pydantic import Field
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import Package, Inputs, Configs, Outputs, Response, Request, Output, Input, Config


class InputEmbedding1(Input):
    name: Literal["embedding_1"] = "embedding_1"
    value: List[float] = Field(..., description="Embedding vector 1", min_items=1)
    type: Literal["list"] = "list"

    class Config:
        title = "Embedding 1"


class InputEmbedding2(Input):
    name: Literal["embedding_2"] = "embedding_2"
    value: List[float] = Field(..., description="Embedding vector 2", min_items=1)
    type: Literal["list"] = "list"

    class Config:
        title = "Embedding 2"


class OutputSimilarity(Output):
    name: Literal["similarity"] = "similarity"
    value: float = Field(..., ge=-1.0, le=1.0, description="Cosine similarity score")
    type: Literal["number"] = "number"

    class Config:
        title = "Similarity"


class CosineSimilarityInputs(Inputs):
    embedding_1: InputEmbedding1
    embedding_2: InputEmbedding2


class CosineSimilarityOutputs(Outputs):
    similarity: OutputSimilarity


class CosineSimilarityRequest(Request):
    inputs: Optional[CosineSimilarityInputs]


class CosineSimilarityResponse(Response):
    outputs: CosineSimilarityOutputs


class CosineSimilarityExecutor(Config):
    name: Literal["CosineSimilarity"] = "CosineSimilarity"
    value: Union[CosineSimilarityRequest, CosineSimilarityResponse]
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
    value: Union[CosineSimilarityExecutor]
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
    name: Literal["CosineSimilarity"] = "CosineSimilarity"

# Cosine Similarity Package

A Novavision capsule (component) designed to calculate the cosine similarity between two 512-dimensional embedding vectors.

## Structure
- `src/executors/CosineSimilarity.py`: Contains the core algorithm to compute the cosine similarity between two embeddings.
- `src/models/PackageModel.py`: Defines the input/output schema according to the Novavision Package Model Specification.

## Inputs
- `embedding_1` (List[float]): The first embedding vector (e.g., from CLIP Image).
- `embedding_2` (List[float]): The second embedding vector (e.g., from CLIP String).

Both inputs are received dynamically via node connections in the flow, not manually via text configurations.

## Outputs
- `similarity` (float): The calculated cosine similarity score.

## Flow Integration
This package can act as the final block in a flow (e.g., `Input Image -> CLIP Image -> Cosine Similarity <- CLIP String <- Input String`), outputting the similarity score without requiring a visual image viewer.

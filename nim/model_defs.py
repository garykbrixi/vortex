# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pydantic import BaseModel, Field, ConfigDict, Base64Bytes
from nim_service_utils import RouteDefinition, AppMetadata, clean as c

class GenerateInputs(BaseModel):
    model_config = ConfigDict(extra='forbid')

    fasta: str = Field(...,
        title='FASTA-formatted string',
        description=c('''A string containing sequence data in FASTA format.
            The FASTA format begins with a description line starting with '>',
            followed by lines of sequence data. The initial '>' line is
            optional.
        '''),
        min_length=1,
        max_length=8192, # TODO: check
    )
    num_tokens: int | None = Field(4, ge=1, le=8192,
        title='Number of tokens to generate',
        description=c('''An integer that controls number of tokens that will
            be generated.
        '''),
    )
    temperature: float | None = Field(1.0,
        title='Temperature',
        description=c('''A float that controls the randomness of the sampling
            process. Values lower than 1.0 make the distribution sharper
            (more deterministic), while values higher than 1.0 make it more
            uniform (more random).
        '''),
    )
    top_k: int | None = Field(4, ge=0, le=20,
        title='Top K',
        description=c('''An integer that specifies the number of highest
            probability tokens to consider. When set to 1, it performs greedy
            decoding by selecting the token with the highest probability.
            Higher values allow for more diverse sampling. If set to 0, all
            tokens are considered.
        '''),
    )
    top_p: float | None = Field(1.0, ge=0.0, le=1.0,
        title='Top P',
        description=c('''A float between 0 and 1 that enables nucleus sampling.
            It filters the smallest set of tokens whose cumulative probability
            exceeds the top_p threshold. Setting this to 0.0 disables top-p
            sampling.
        '''),
    )
    enable_scores: bool = Field(False,
        title='Enable Raw Scores Reporting',
        description=c('''A boolean that if set, enables raw scores reporting in
            the output response.
        '''),
    )
    enable_sampled_probs: bool = Field(False,
        title='Enable Sampled Token Probabilities Reporting',
        description=c('''A boolean flag that, when set to True, enables the
            reporting of sampled token probabilities. When enabled, this
            feature generates a list of probability values (ranging
            from 0 to 1) corresponding to each token in the output sequence.
            These probabilities represent the model's confidence in selecting
            each token during the generation process. The resulting list has
            the same length as the output sequence, providing insight into the
            model's decision-making at each step of text generation.
        '''),
    )


class GenerateOutputs(BaseModel):
    model_config = ConfigDict(extra='forbid')

    fasta: str = Field(
        title='DNA sequence',
        description='Output DNA sequence in FASTA format',
    )
    scores: list[list[float]] | None = Field(None,
        title='Scores',
        description=c('''Output Raw Scores in [num_tokens,512] shape
            (if requested via enable_scores flag.)
        '''),
    )
    sampled_probs: list[float] | None = Field(None,
        title='Sampled Token Probabilities',
        description=c('''A list of probabilities corresponding to each token
            in the generated output sequence. Each value ranges from 0 to 1,
            representing the model's confidence in selecting that specific token
            during the generation process. The list length matches the output
            sequence length. To use this feature, set `enable_sampled_probs` to
            True. This information provides insight into the model's
            decision-making at each step of text generation.
        '''),
    )
    elapsed_ms: int = Field(
        title='Elapsed milliseconds',
        description='Elapsed milliseconds on server side',
    )


class GenerateRoute(RouteDefinition):
    API_PATH: str = '/biology/arc/evo2/generate'
    API_SUMMARY: str = 'Generate DNA sequences'
    API_DESCRIPTION: str = API_SUMMARY
    MODEL_NAME: str = 'evo2'
    ModelInputs = GenerateInputs
    ModelOutputs = GenerateOutputs
    X_NVAI_META: dict = {
        'name': API_SUMMARY,
        'returns': 'Generated DNA sequence based on the input parameters.',
        'path': 'generate',
    }


class EmbeddingsInputs(BaseModel):
    model_config = ConfigDict(extra='forbid')

    fasta: str = Field(...,
        title='FASTA-formatted string',
        description=c('''A string containing sequence data in FASTA format.
            The FASTA format begins with a description line starting with '>',
            followed by lines of sequence data. The initial '>' line is
            optional.
        '''),
        min_length=1,    # TODO: check
        max_length=8192, # TODO: check
    )
    layer_index: int | None = Field(1,
        title='Layer Index',
        description=c('''The index of the layer from which to extract
            embeddings. If set to 1, embeddings will be extracted from the
            first layer. If None, embeddings will be extracted from the
            last layer.'''
        ),
    )

class EmbeddingsOutputs(BaseModel):
    model_config = ConfigDict(extra='forbid')

    embeddings: Base64Bytes = Field(
        title='Embeddings',
        description=c('''Output Embeddings in NumPy Zipped (NPZ) format, base64
            encoded.'''
        ),
    )
    elapsed_ms: int = Field(
        title='Elapsed milliseconds on server side',
        description='Elapsed milliseconds on server side',
    )


class EmbeddingsRoute(RouteDefinition):
    API_PATH: str = '/biology/arc/evo2/embeddings'
    API_SUMMARY: str = 'Generate embeddings for DNA sequence'
    API_DESCRIPTION: str = API_SUMMARY
    MODEL_NAME: str = 'evo2'
    ModelInputs = EmbeddingsInputs
    ModelOutputs = EmbeddingsOutputs
    X_NVAI_META: dict = {
        'name': API_SUMMARY,
        'returns': 'Embeddings for the input DNA sequence at a specific layer.',
        'path': 'embeddings',
    }


class Metadata(AppMetadata):
    MODEL_NAME_PRETTY: str = 'Evo2'
    API_SEMVER: str = '1.0.0'
    MODEL_LICENSE: str = 'Apache License Version 2.0' # TODO: double check
    MODEL_LICENSE_URL: str = 'https://github.com/Zymrael/vortex/blob/b8bf0e53711a6dbf57c9351fe9fdc461e1399028/LICENSE' # TODO: need public URL


route_definitions = [GenerateRoute, EmbeddingsRoute]

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pydantic import BaseModel, Field, ConfigDict, Base64Bytes, constr
from nim_service_utils import RouteDefinition, AppMetadata, clean as c

class GenerateInputs(BaseModel):
    model_config = ConfigDict(extra='forbid')

    sequence: str = Field(...,
        title='Input DNA Sequence',
        description=c('''A string containing DNA sequence data.'''),
        min_length=1,
        max_length=8192, # TODO: check
    )
    num_tokens: int | None = Field(100, ge=1, le=8192,
        title='Number of tokens to generate',
        description=c('''An integer that controls number of tokens that will
            be generated.
        '''),
    )
    temperature: float | None = Field(0.7, gt=0.0, le=1.3,
        title='Temperature',
        description=c('''A float that controls the randomness of the sampling
            process. Values lower than 1.0 make the distribution sharper
            (more deterministic), while values higher than 1.0 make it more
            uniform (more random).
        '''),
    )
    top_k: int | None = Field(3, ge=0, le=6,
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
    random_seed: int | None = Field(None,
        title='Random Seed',
        description=("Evo2 is a generative model, its function is "
                     "to generate novel and diverse DNA sequences. Setting "
                     "random seed allows to turn Evo2 into a deterministic "
                     "model, where an input DNA and a fixed seed "
                     "would always produce the same output. This argument is "
                     "useful for development purposes, but otherwise should "
                     "be unset."
        ),
        numpy_dtype="int64",
        triton_shape=(1,)
    )
    enable_logits: bool = Field(False,
        title='Enable Logits Reporting',
        description=c('''A boolean that if set, enables logits reporting in
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
    enable_elapsed_ms_per_token: bool = Field(False,
        title='Enable Per-Token Elapsed Time Reporting',
        description=c('''A boolean flag that, when set to True, enables the
            reporting of per-token timing statistics. Used for benchmarking.
        '''),
    )


class GenerateOutputs(BaseModel):
    model_config = ConfigDict(extra='forbid')

    sequence: str = Field(
        title='DNA sequence',
        description='Output DNA sequence.',
    )
    logits: list[list[float]] | None = Field(None,
        title='Logits',
        description=c('''Output Logits in [num_tokens,512] shape
            (if requested via enable_logits flag.)
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
    elapsed_ms_per_token: list[int] | None = Field(None,
        title='Elapsed milliseconds for each generated token',
        description=c('''Elapsed milliseconds on server side for each generated
            token.
        '''),
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


class ForwardInputs(BaseModel):
    model_config = ConfigDict(extra='forbid')

    sequence: str = Field(...,
        title='Input DNA sequence',
        description=c('''A string containing DNA sequence data.'''),
        min_length=1,
        max_length=8192, # TODO: check
    )
    output_layers: list[constr(min_length=1)] = Field(...,
        title='Output capture layers.',
        description=c('''List of layer names from which to capture and save
            output tensors. For example, `["embedding_layer"]`.'''
        ),
        min_items=1,
        max_items=10,
    )

class ForwardOutputs(BaseModel):
    model_config = ConfigDict(extra='forbid')

    data: Base64Bytes = Field(
        title='outputs',
        description=c('''Tensors of requested layers in NumPy Zipped (NPZ)
            format, base64 encoded.'''
        ),
    )
    elapsed_ms: int = Field(
        title='Elapsed milliseconds on server side',
        description='Elapsed milliseconds on server side',
    )


class ForwardRoute(RouteDefinition):
    API_PATH: str = '/biology/arc/evo2/forward'
    API_SUMMARY: str = 'Run model forward pass and save layers outputs'
    API_DESCRIPTION: str = API_SUMMARY
    MODEL_NAME: str = 'evo2'
    ModelInputs = ForwardInputs
    ModelOutputs = ForwardOutputs
    X_NVAI_META: dict = {
        'name': API_SUMMARY,
        'returns': 'Tensors of requested layers.',
        'path': 'forward',
    }


class Metadata(AppMetadata):
    MODEL_NAME_PRETTY: str = 'Evo2'
    API_SEMVER: str = '1.0.0'
    MODEL_LICENSE: str = 'Apache License Version 2.0' # TODO: double check
    MODEL_LICENSE_URL: str = 'https://github.com/Zymrael/vortex/blob/b8bf0e53711a6dbf57c9351fe9fdc461e1399028/LICENSE' # TODO: need public URL


route_definitions = [GenerateRoute, ForwardRoute]

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" configuration"""



# ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
#     "roberta-base": "https://huggingface.co/roberta-base/resolve/main/config.json",
#     "roberta-large": "https://huggingface.co/roberta-large/resolve/main/config.json",
#     "roberta-large-mnli": "https://huggingface.co/roberta-large-mnli/resolve/main/config.json",
#     "distilroberta-base": "https://huggingface.co/distilroberta-base/resolve/main/config.json",
#     "roberta-base-openai-detector": "https://huggingface.co/roberta-base-openai-detector/resolve/main/config.json",
#     "roberta-large-openai-detector": "https://huggingface.co/roberta-large-openai-detector/resolve/main/config.json",
# }

"""
    This is the configuration class to store the configuration of a [`RobertaModel`] or a [`TFRobertaModel`]. It is
    used to instantiate a RoBERTa model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the RoBERTa
    [roberta-base](https://huggingface.co/roberta-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the RoBERTa model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`RobertaModel`] or [`TFRobertaModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`RobertaModel`] or [`TFRobertaModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
    """

from transformers.configuration_utils import PretrainedConfig

class WelkirConfig(PretrainedConfig):
    def __init__(
        self,
        config=None,
        vocab_size=30000,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=3,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        bb_hidden_size=768,
        num_bb_hidden_layers=6,
        num_bb_attention_heads=12,
        bb_intermediate_size=3072,
        max_bb_position_embeddings=256,
        num_cfg_flow_types=21,
        num_dfg_flow_types=22,
        add_flow_self_attn_bias=True,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

        default_values = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "intermediate_size": intermediate_size,
            "hidden_act": hidden_act,
            "hidden_dropout_prob": hidden_dropout_prob,
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "max_position_embeddings": max_position_embeddings,
            "type_vocab_size": type_vocab_size,
            "initializer_range": initializer_range,
            "layer_norm_eps": layer_norm_eps,
            "position_embedding_type": position_embedding_type,
            "use_cache": use_cache,
            "classifier_dropout": classifier_dropout,
            "bb_hidden_size": bb_hidden_size,
            "num_bb_hidden_layers": num_bb_hidden_layers,
            "num_bb_attention_heads": num_bb_attention_heads,
            "bb_intermediate_size": bb_intermediate_size,
            "max_bb_position_embeddings": max_bb_position_embeddings,
            "num_cfg_flow_types": num_cfg_flow_types,
            "num_dfg_flow_types": num_dfg_flow_types,
            "add_flow_self_attn_bias": add_flow_self_attn_bias,
        }

        for key, default in default_values.items():
            setattr(self, key, getattr(config, key, default) if config else default)
              
        if config:
            for key, value in config.__dict__.items():
                if not hasattr(self, key):
                    setattr(self, key, value)
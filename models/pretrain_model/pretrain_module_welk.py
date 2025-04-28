
import math
from typing import List, Optional, Tuple, Union
from loguru import logger
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN, gelu
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import (
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    logging,
)

from .configuration_welkir import WelkirConfig
from .pretrain_welk_for_inst import (WelkirinstEncoder, WelkirinstLMHead)




class CustomOutputWithAllInst(BaseModelOutputWithPoolingAndCrossAttentions):
    def __init__(
        self,
        last_hidden_state: torch.FloatTensor,
        pooler_output: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
        attentions: Optional[Tuple[torch.FloatTensor]] = None,
        cross_attentions: Optional[Tuple[torch.FloatTensor]] = None,
        all_inst_hidden_states: Optional[list] = None
    ):
        # 调用父类的构造函数，初始化继承的部分
        super().__init__(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
            cross_attentions=cross_attentions,
        )
        # 添加新的属性 all_inst_hidden_states
        self.all_inst_hidden_states = all_inst_hidden_states



def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=0
):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from Welkirseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (
        torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
    ) * mask
    return incremental_indices.long() + padding_idx


class WelkirLMHead(nn.Module):
    """Head for masked language modeling with customizable vocab size."""
    def __init__(self, config, vocab_size):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=float(config.layer_norm_eps))
        self.decoder = nn.Linear(config.hidden_size, vocab_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = torch.nn.functional.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

    def _tie_weights(self):
        # To tie the two weights if they get disconnected (for compatibility with certain accelerators)
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias

class WelkirPreTrainedModel(PreTrainedModel):
    config_class = WelkirConfig
    base_model_prefix = "Welkir"
    supports_gradient_checkpointing = True
    _no_split_modules = []

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, WelkirEncoder):
            module.gradient_checkpointing = value


class WelkirEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=float(config.layer_norm_eps))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length
                )
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds
                )


        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # logger.info(f"WelkirEmbeddings, position_ids.shape:{position_ids.shape}, position_ids:{position_ids}")
        

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually
        # occurs when its auto-generated, registered buffer helps users when tracing the model without passing
        # token_type_ids, solves issue #5664
        # if token_type_ids is None:
        #     if hasattr(self, "token_type_ids"):
        #         buffered_token_type_ids = self.token_type_ids[:, :seq_length]
        #         buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
        #             input_shape[0], seq_length
        #         )
        #         token_type_ids = buffered_token_type_ids_expanded
        #     else:
        #         token_type_ids = torch.zeros(
        #             input_shape, dtype=torch.long, device=self.position_ids.device
        #         )
        # logger.info(f"WelkirEmbeddings, token_type_ids.shape:{token_type_ids.shape}")

        # logger.info(f"WelkirEmbeddings, input_ids.shape:{input_ids.shape}")
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # embeddings = inputs_embeds + token_type_embeddings

        embeddings = inputs_embeds

        # logger.info(f"WelkirEmbeddings, inputs_embeds.shape:{inputs_embeds.shape}, inputs_embeds:{inputs_embeds}")
        # logger.info(f"position_ids shape: {position_ids.shape}, min value: {position_ids.min()}, max value: {position_ids.max()}")

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        #logger.info(f"after position_embeddings inputs_embeds.shape:{inputs_embeds.shape}, inputs_embeds:{inputs_embeds}")
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1,
            sequence_length + self.padding_idx + 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class WelkirSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.add_flow_self_attn_bias = config.add_flow_self_attn_bias
        if config.add_flow_self_attn_bias:
            self.cfg_embedding = nn.Embedding(
                config.num_cfg_flow_types + 1, 1, padding_idx=0
            )
            self.dfg_embedding = nn.Embedding(
                config.num_dfg_flow_types + 1, 1, padding_idx=0
            )
            self.rfg_embedding = nn.Embedding(
                config.num_rdg_flow_types + 1, 1, padding_idx=0
            )
            
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1, self.attention_head_size
            )

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        cfg_matrix: torch.Tensor=None,
        dfg_matrix: torch.Tensor=None,
        rdg_matrix: torch.Tensor=None,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(
                    key_length - 1, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            else:
                position_ids_l = torch.arange(
                    query_length, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            position_ids_r = torch.arange(
                key_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # logger.info(f"WelkirSelfAttention attention_scores.shape: {attention_scores.shape}")
        
        # logger.info(f"self.add_flow_self_attn_bias: {self.add_flow_self_attn_bias}")
        if self.add_flow_self_attn_bias:
            flow_attn_scores = self.calculate_flow_type_scores(
                size=attention_scores.size(),
                cfg_matrix=cfg_matrix,
                dfg_matrix=dfg_matrix,
                rdg_matrix=rdg_matrix,
                device=attention_scores.device
            )
            attention_scores = attention_scores + flow_attn_scores

        # logger.info(f"WelkirSelfAttention add_flow_self_attn_bias attention_scores.shape: {attention_scores.shape}")
        # logger.info(f"WelkirSelfAttention attention_mask.shape: {attention_mask.shape}")
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in WelkirModel forward() function)
            attention_scores = attention_scores + attention_mask


        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    def calculate_flow_type_scores(
        self,
        size: torch.Size,
        cfg_matrix: torch.Tensor,
        dfg_matrix: torch.Tensor,
        rdg_matrix: torch.Tensor,
        device,
    ) -> torch.Tensor:
        
        # logger.info(f"calculate_flow_type_scores size: {size}")
        # logger.info(f"calculate_flow_type_scores cfg_matrix shape: {cfg_matrix.shape}")
        # logger.info(f"calculate_flow_type_scores dfg_matrix shape: {dfg_matrix.shape}")
        # logger.info(f"calculate_flow_type_scores rdg_matrix shape: {rdg_matrix.shape}")

        # [B, num_heads, T, T]
        scores = torch.zeros(size, dtype=torch.float, device=device)
        cfg_len = cfg_matrix.size(1)
        # logger.info(f"cfg_matrix, cfg_len: {cfg_len}, scores shape: {scores.shape} ")

        cfg_scores = self.cfg_embedding(cfg_matrix).squeeze(-1)     # [B, cfg_len, cfg_len] 
        cfg_scores = cfg_scores.unsqueeze(1).repeat(1, size[1], 1, 1)
  
        dfg_scores = self.dfg_embedding(dfg_matrix).squeeze(-1)
        dfg_scores = dfg_scores.unsqueeze(1).repeat(1, size[1], 1, 1)
        
        rfg_scores = self.rfg_embedding(rdg_matrix).squeeze(-1)
        rfg_scores = rfg_scores.unsqueeze(1).repeat(1, size[1], 1, 1)
     
        scores = cfg_scores + dfg_scores + rfg_scores 
        # logger.info(f"calculate_flow_type_scores cfg_scores shape: {cfg_scores.shape} ")
        # logger.info(f"calculate_flow_type_scores dfg_scores shape: {dfg_scores.shape} ")
        # logger.info(f"calculate_flow_type_scores rfg_scores shape: {rfg_scores.shape} ")
        # logger.info(f"calculate_flow_type_scores total scores shape: {scores.shape} ")
        
        return scores


class WelkirSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=float(config.layer_norm_eps))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class WelkirAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = WelkirSelfAttention(
            config, position_embedding_type=position_embedding_type
        )
        self.output = WelkirSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        cfg_matrix: Optional[torch.Tensor] = None,
        dfg_matrix: Optional[torch.Tensor] = None,
        rdg_matrix: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            cfg_matrix=cfg_matrix,
            dfg_matrix=dfg_matrix,
            rdg_matrix=rdg_matrix
        )
        # logger.info(f"finish WelkirAttention self.self = WelkirSelfAttention ")
        #attention_output：经过自注意力机制和输出层处理后的隐藏状态，[batch_size, seq_len, hidden_size]
        #self_outputs[1:],attention_probs注意力权重， [batch_size, num_heads, seq_len, seq_len]，表示每个注意力头对于输入序列中每个位置的注意力分数。

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class WelkirIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class WelkirOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=float(config.layer_norm_eps))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class WelkirLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = WelkirAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = WelkirAttention(
                config, position_embedding_type="absolute"
            )
        self.intermediate = WelkirIntermediate(config)
        self.output = WelkirOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        cfg_matrix: Optional[torch.Tensor] = None,
        dfg_matrix: Optional[torch.Tensor] = None,
        rdg_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )

        # logger.info(f"start WelkirLayer self.attention")
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
            cfg_matrix=cfg_matrix,
            dfg_matrix=dfg_matrix,
            rdg_matrix=rdg_matrix
        )
        attention_output = self_attention_outputs[0]
        # logger.info(f" WelkirLayer self.attention")
        # logger.info(f" WelkirLayer is_decoder:{self.is_decoder} ")
        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[
                1:
            ]  # add self attentions if we output attention weights


        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:-1]
            )  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # [batch_size, seq_len, hidden_size]，该层最终的输出，即经过前馈网络处理后的隐藏状态
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        # logger.info(f" WelkirLayer layer_output shape:{layer_output.shape} ")

        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class WelkirEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [WelkirLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        cfg_matrix: Optional[torch.Tensor] = None,
        dfg_matrix: Optional[torch.Tensor] = None,
        rdg_matrix: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        
        # logger.info(f"WelkirEncoder start")

        output_hidden_states = True
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logging.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        next_decoder_cache = () if use_cache else None


        # logger.info(f"start WelkirEncoder layer_module")
        # logger.info(f"WelkirEncoder hidden_states.shape:{hidden_states.shape}")
        # logger.info(f"WelkirEncoder  head_mask:{head_mask}")
        # logger.info(f"WelkirEncoder gradient_checkpointing:{self.gradient_checkpointing}")
        

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    cfg_matrix=cfg_matrix,
                    dfg_matrix=dfg_matrix,
                    rdg_matrix=rdg_matrix
                )
            # logger.info(f"WelkirEncoder finish layer_module {i}")
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertPooler
class WelkirPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class WelkirModel(WelkirPreTrainedModel):
    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Welkir
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = WelkirEmbeddings(config)
        self.inst_encoder = WelkirinstEncoder(config)
        self.encoder = WelkirEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: torch.Tensor,
        all_inst_encoder_input_ids: torch.Tensor,
        total_group_count: torch.Tensor,
        input_inst_tokencount: torch.Tensor,
        input_instcount_list: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cfg_matrix: Optional[torch.Tensor] = None,
        dfg_matrix: Optional[torch.Tensor] = None,
        rdg_matrix: Optional[torch.Tensor] = None,
        return_inst_encoder_pooled_outputs: bool = False
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        # logger.info(f"output_attentions : {output_attentions}") 

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        # logger.info(f"output_hidden_states : {output_hidden_states}") 


        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        # logger.info(f"input_shape shape : {input_shape}") 

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        # logger.info(f"device: {device}") 

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        # logger.info(f"attention_mask shape : {attention_mask.shape}") 

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)), device=device
            )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )

        # logger.info(f"token_type_ids shape : {token_type_ids.shape}, token_type_ids:{token_type_ids}") 


        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        # logger.info(f"extended_attention_mask shape : {extended_attention_mask.shape}") 
        # logger.info(f"self.config.is_decoder:{self.config.is_decoder}")

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        all_inst_embedding_outputs = []
        all_inst_hidden_states = []
        # logger.info(f"all_inst_encoder_input_ids shape:{all_inst_encoder_input_ids.shape}")
        # logger.info(f"input_instcount_list shape:{input_instcount_list.shape}")
        # logger.info(f"input_inst_tokencount shape:{input_inst_tokencount.shape}")
            

        # 用于存储所有批次的指令嵌入
        batch_inst_embeddings = []
        
        for batch_idx, (inst_input_ids, input_instcount, input_tokencount, input_total_group_count) in enumerate(zip(all_inst_encoder_input_ids, input_instcount_list, input_inst_tokencount, total_group_count)):

            # logger.info(f"Processing batch index {batch_idx}")
            # logger.info(f"inst_input_ids shape:{inst_input_ids.shape}")
            # logger.info(f"input_instcount shape:{input_instcount.shape}, input_instcount:{input_instcount}")
            # logger.info(f"input_tokencount shape:{input_tokencount.shape}, input_tokencount{input_tokencount}")
            # logger.info(f"input_total_group_count shape:{input_total_group_count.shape}, input_total_group_count:{input_total_group_count}")

            # 获取填充的数量
            inst_input_ids[inst_input_ids  == -1] = self.config.pad_token_id
            num_pad_token = (inst_input_ids == self.config.pad_token_id).sum(dim=1)
            # logger.info(f"Number of pad_token_id in inst_input_ids: {num_pad_token}")
            
      
            # 检查 valid_inst_input_ids 的最小和最大值
            min_value = inst_input_ids.min().item()
            max_value = inst_input_ids.max().item()
            inst_encoder_attention_mask = inst_input_ids.ne(self.config.pad_token_id).int()  
            # 输出最小值和最大值
            # logger.info(f"inst_input_ids min value: {min_value}, max value: {max_value}")
            # logger.info(f"inst_input_ids shape: {inst_input_ids.shape}")
            # logger.info(f"valid_attention_mask shape: {inst_encoder_attention_mask.shape}, valid_attention_mask:{inst_encoder_attention_mask}")
            inst_embeddings = self.embeddings(inst_input_ids)
            _, inst_hidden_states = self.inst_encoder(inst_embeddings, attention_mask=inst_encoder_attention_mask)
            
            # logger.info(f"finish inst_encoder, inst_hidden_states shape:{inst_hidden_states.shape}")


            all_inst_hidden_states.append(inst_hidden_states)
            # 用于存储当前批次所有组的指令嵌入
            current_sample_inst_embeddings = []
            # 遍历当前样本中的每个组
            for group_idx, inst_count in enumerate(input_instcount):
                if group_idx >= input_total_group_count:
                    break
                inst_count = inst_count.item()  # 当前组的指令数
                #logger.info(f"Processing group {group_idx}, number of instructions: {inst_count}")
                tokencount_for_group = input_tokencount[group_idx, :inst_count]
                # logger.info(f"Token count for group {group_idx}: {tokencount_for_group}")
                # 起始位置为 1，跳过 [CLS] token
                start_pos = 1
                # 遍历当前组中的每条指令
                for inst_idx in range(inst_count):
                    token_count = tokencount_for_group[inst_idx].item()
                    if token_count == -1:
                        logger.error(f"Unexpected token count -1 at group {group_idx}, instruction {inst_idx}")
                    end_pos = start_pos + token_count
                    #logger.info(f"Instruction {inst_idx} in group {group_idx}: start_pos={start_pos}, end_pos={end_pos}, number of tokens={token_count}")
                    # 获取当前指令的 token 嵌入，去除[SEP] token，形状：[token数, hidden_size]
                    inst_token_embeddings = inst_hidden_states[group_idx, start_pos:end_pos-1, :]  
                    # 对 token 嵌入进行平均池化，得到指令嵌入
                    pooled_inst_embeddings = inst_token_embeddings.mean(dim=0)  # 形状：[hidden_size]
                    # 将指令嵌入添加到当前样本的结果列表，更新下一个指令的起始位置
                    current_sample_inst_embeddings.append(pooled_inst_embeddings)
                    start_pos = end_pos
                # logger.info(f" group {group_idx}, inst_embeddings_len: {len(current_sample_inst_embeddings)}")

            # 将当前样本的所有指令嵌入堆叠起来，形成 [指令总数, hidden_size] 形状的张量
            current_sample_inst_embeddings = torch.stack(current_sample_inst_embeddings)  # [指令总数, hidden_size]
            batch_inst_embeddings.append(current_sample_inst_embeddings)
            # logger.info(f"finish inst_encoder, current_sample_inst_embeddings.shape:{current_sample_inst_embeddings.shape}")
            # logger.info(f"finish inst_encoder, batch_inst_embeddings.len:{len(batch_inst_embeddings)}")


        # 将所有批次的指令嵌入转换为最终张量
        # 每个元素是 [指令总数, hidden_size]，我们需要将它们合并为 [batch_size, 指令总数, hidden_size]
        all_inst_embedding_outputs = torch.nn.utils.rnn.pad_sequence(batch_inst_embeddings, batch_first=True, padding_value=0.0)  # [batch_size, max_指令总数, hidden_size]
        
        # logger.info(f"Finished processing all instructions, final all_inst_embedding_outputs shape: {all_inst_embedding_outputs.shape}")


        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        for index, inst_embedding_outputs in enumerate(all_inst_embedding_outputs):
            embedding_output[index, 1: 1 + inst_embedding_outputs.size(0), :] = inst_embedding_outputs

        # logger.info(f"embedding_output shape :{embedding_output.shape}")
        # logger.info(f"start self.encoder, self.encoder = WelkirEncoder(config) ")
        # logger.info(f"WelkirModel cfg_matrix shape: {cfg_matrix.shape}")
        # logger.info(f"WelkirModel dfg_matrix shape: {dfg_matrix.shape}")
        # logger.info(f"WelkirModel rdg_matrix shape: {rdg_matrix.shape}")      
        # logger.info(f"WelkirModel WelkirEncoder, return_dict: {return_dict}, use_cache:{use_cache}")
        # logger.info(f"WelkirModel WelkirEncoder, output_attentions: {output_attentions}")
        #return_dict: True
        # use_cache:False
        # output_attentions: True


        # encoder_outputs : BaseModelOutputWithPastAndCrossAttentions(
        #     last_hidden_state=hidden_states,           # 编码器最后一层的输出
        #     past_key_values=next_decoder_cache,        # 缓存的键和值
        #     hidden_states=all_hidden_states,           # 所有层的隐藏状态
        #     attentions=all_self_attentions,            # 所有层的自注意力
        #     cross_attentions=all_cross_attentions      # 所有层的跨注意力（如果有）
        # )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cfg_matrix=cfg_matrix,
            dfg_matrix=dfg_matrix,
            rdg_matrix=rdg_matrix,
        )


        # logger.info(f"WelkirModel finish encoder, WelkirEncoder(config), return_dict:{return_dict}")

        if return_dict:
            # 打印 encoder_outputs 的属性及其值
            # for attr, value in vars(encoder_outputs).items():
            #     logger.info(f"Attribute: {attr}, Value type: {type(value)}")
            sequence_output = encoder_outputs.last_hidden_state
            hidden_states = encoder_outputs.hidden_states
        else:
            sequence_output = encoder_outputs[0]
            hidden_states = encoder_outputs[2]

        # sequence_output = encoder_outputs[0]
        # hidden_states = encoder_outputs[1]
        first_hidden = hidden_states[1]
        last_hidden = hidden_states[-1]

        pooled_output = (((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) /
                         attention_mask.sum(-1).unsqueeze(-1))
        
        # logger.info(f"WelkirModel,attention_mask shape:{attention_mask.shape} ")
        # logger.info(f"WelkirModel,sequence_output shape:{sequence_output.shape}")
        # logger.info(f"WelkirModel pooled_output shape:{pooled_output.shape}")

        if not return_dict:
            if return_inst_encoder_pooled_outputs:
                # for pre-training
                return sequence_output, pooled_output, encoder_outputs[1:], all_inst_hidden_states
            else:
                return (sequence_output, pooled_output) + encoder_outputs[1:], all_inst_hidden_states

        return CustomOutputWithAllInst(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
            all_inst_hidden_states=all_inst_hidden_states
        )


class WelkirClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class WelkirRelationHead(nn.Module):
    """Welkir Head for relation prediction (CFG/DFG/RDG)"""

    def __init__(self, config, relation_vocab_size):
        super().__init__()
        self.reduce_dim = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=float(config.layer_norm_eps))
        self.decoder = nn.Linear(config.hidden_size, relation_vocab_size)
        self.bias = nn.Parameter(torch.zeros(relation_vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        
        # 输入形状 [batch_size, seq_length, seq_length, 2 * hidden_size]，输出 [batch_size, seq_length, seq_length, hidden_size]
        x = self.reduce_dim(features)  
        x = gelu(x)
        x = self.dense(x)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias (e.g., relation types)
        x = self.decoder(x)
        return x




class WelkirPreTrainedLearningModel(WelkirPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.welkir_model = WelkirModel(config, add_pooling_layer=False)
        # MLM概率计算
        self.instLMHead = WelkirinstLMHead(config)

        # 关系概率计算
        self.cfg_relation_head = WelkirRelationHead(config, config.cfg_vocab_size)
        self.dfg_relation_head = WelkirRelationHead(config, config.dfg_vocab_size)
        self.rdg_relation_head = WelkirRelationHead(config, config.rdg_vocab_size)

        # 损失函数使用交叉熵损失（适用于多分类任务）
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.post_init()
        
    
    # 计算 CFG、DFG 和 RDG 的关系矩阵预测任务的损失函数
    def compute_relation_loss(self, matrix, matrix_mask, relation_head, sequence_output):
        assert matrix.shape == matrix_mask.shape, "Matrix and mask must have the same shape"
        # 使用掩码提取有效的 (i, j) 对，获取需要进行预测的有效位置的索引
        valid_indices = torch.nonzero(matrix_mask, as_tuple=False)  # [batch_idx, i, j]
        if valid_indices.size(0) == 0:
            return torch.tensor(0.0, dtype=torch.float32)
        # logger.info(f"compute_relation_loss valid_indices.shape:{valid_indices.shape} ")
        # 特征的拼接：每对 (i, j) 对应的特征为 [h_i; h_j]，即合并节点 i 和节点 j 的特征
        # 生成 [num_valid_positions, 2 * hidden_size]，将 valid_indices 的每对 (i, j) 特征进行拼接
        # 使用关系预测头对拼接后的特征进行预测
        features = torch.cat((
            sequence_output[valid_indices[:, 0], valid_indices[:, 1], :],
            sequence_output[valid_indices[:, 0], valid_indices[:, 2], :]
        ), dim=-1)  # [num_valid_positions, 2 * hidden_size]

        # logger.info(f"compute_relation_loss features.shape:{features.shape} ")
        pred_logits = relation_head(features)  
        pred_logits = pred_logits.float()
        # 输出为 [num_valid_positions, vocab_size]
        # 提取相应的有效标签,将无效的位置设置为 -100
        valid_labels = matrix[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]  # [num_valid_positions]
        valid_labels[valid_labels == 0] = -100
        valid_labels = valid_labels.to(dtype=torch.long)
        # 计算损失，使用 CrossEntropyLoss 忽略 -100 的位置
        loss = self.loss_fct(pred_logits, valid_labels)
        return loss

      
    # 计算MLM预测任务的损失函数           
    def compute_mlm_loss(self, all_inst_hidden_states, labels):
        # 将 all_inst_hidden_states 从列表拼接为一个 Tensor，形状为 [batch_size * num_groups, seq_length, hidden_size]  
        all_inst_hidden_states = torch.cat(all_inst_hidden_states, dim=0)
        logger.info(f"compute_mlm_loss, all_inst_hidden_states.shape:{all_inst_hidden_states.shape} ")

        # [batch_size * num_groups, seq_length, vocab_size]
        logits = self.instLMHead(all_inst_hidden_states)  
        # logger.info(f"compute_mlm_loss, logits.shape:{logits.shape} ")
        logits = logits.float()
        logits = logits.view(-1, logits.size(-1))
        labels = labels.long()
        # 将 labels 重塑为 [batch_size * num_groups * seq_length]
        labels = labels.view(-1)
        mlm_loss = self.loss_fct(logits, labels)

        # logger.info(f"compute_mlm_loss, self.loss_fct, logits.shape:{logits.shape}")
        # logger.info(f"compute_mlm_loss, self.loss_fct, labels.shape:{labels.shape}")
        return mlm_loss
    
    def forward(
        self,
        input_ids: torch.Tensor,
        all_inst_encoder_input_ids: torch.Tensor,
        total_group_count: torch.Tensor,
        input_inst_tokencount: torch.Tensor,
        input_instcount_list: torch.Tensor,
        inst_mlm_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        cfg_matrix: Optional[torch.Tensor] = None,
        dfg_matrix: Optional[torch.Tensor] = None,
        rdg_matrix: Optional[torch.Tensor] = None,
        cfg_matrix_mask: Optional[torch.Tensor] = None,
        dfg_matrix_mask: Optional[torch.Tensor] = None,
        rdg_matrix_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        

        # logger.info(f"start WelkirPreTrainedLearningModel ")
        # logger.info(f"total_group_count:{total_group_count} ")
        # logger.info(f"input_inst_tokencount:{input_inst_tokencount} ")
        # logger.info(f"input_instcount_list:{input_instcount_list} ")


        # 获取模型输出
        welkir_model_outputs = self.welkir_model(
            input_ids=input_ids,
            all_inst_encoder_input_ids=all_inst_encoder_input_ids,
            total_group_count=total_group_count,
            input_inst_tokencount=input_inst_tokencount,
            input_instcount_list=input_instcount_list,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_inst_encoder_pooled_outputs=True,
            return_dict=return_dict,
            cfg_matrix=cfg_matrix,
            dfg_matrix=dfg_matrix,
            rdg_matrix=rdg_matrix,
            **kwargs
        )

        # pooler_output：整个指令序列的嵌入，用于下游任务。
        # last_hidden_state：指令序列中每个指令的最终输出，用于控制依赖和数据依赖的预测。
        # all_inst_hidden_states：每个指令中各个 token 的嵌入，用于 token 的预测。

        # CustomOutputWithAllInst(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,   
        #     past_key_values=encoder_outputs.past_key_values,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        #     cross_attentions=encoder_outputs.cross_attentions,
        #     all_inst_hidden_states=all_inst_hidden_states
        # )


        # logger.info(f"finish welkir_model,  welkir_model_outputs ")

        if return_dict:
            # 打印 encoder_outputs 的属性及其值
            # for attr, value in vars(welkir_model_outputs).items():
            #     logger.info(f"Attribute: {attr}, Value type: {type(value)}")
            sequence_output = welkir_model_outputs.last_hidden_state
            all_inst_token_hidden_states = welkir_model_outputs.all_inst_hidden_states
        else:
            sequence_output = welkir_model_outputs[0]
            all_inst_token_hidden_states = welkir_model_outputs[2]

        # ==================== 掩码语言模型（MLM）任务 ====================
        # logger.info(f"loss all_inst_hidden_states len: {len(all_inst_token_hidden_states)}")
        
        # logger.info(f"loss total_group_count shape: {total_group_count}")
        logger.info(f"sequence_output shape: {sequence_output.shape}")
        logger.info(f"loss inst_mlm_labels shape: {inst_mlm_labels.shape}")

        if inst_mlm_labels is not None:
            mlm_loss = self.compute_mlm_loss(all_inst_token_hidden_states, inst_mlm_labels)
        else:
            mlm_loss = 0
        logger.info(f"finish welkir_model  all_inst_token_hidden_states compute_mlm_loss,mlm_loss:{mlm_loss}")

        # logger.info(f"finish welkir_model  cfg_matrix.shape:{cfg_matrix.shape} , cfg_matrix_mask.shape:{cfg_matrix_mask}")
        # # 计算 CFG、DFG、RDG 的损失
        # 还需要考虑掩码
        # cfg_loss = 0
        cfg_loss = self.compute_relation_loss(cfg_matrix, cfg_matrix_mask, self.cfg_relation_head, sequence_output)
        logger.info(f"finish welkir_model matrix  cfg_loss compute_relation_loss, cfg_loss:{cfg_loss} ")

        # dfg_loss = 0
        dfg_loss = self.compute_relation_loss(dfg_matrix, dfg_matrix_mask, self.dfg_relation_head, sequence_output)
        logger.info(f"finish welkir_model matrix  dfg_loss compute_relation_loss, dfg_loss:{dfg_loss}")

        # rdg_loss = 0
        rdg_loss = self.compute_relation_loss(rdg_matrix, rdg_matrix_mask, self.rdg_relation_head, sequence_output)
        logger.info(f"finish welkir_model rdg_loss  compute_relation_loss, rdg_loss:{rdg_loss}")
        # 总损失
        # # 01 all loss
        # total_loss = mlm_loss + cfg_loss + dfg_loss + rdg_loss
        
        # # 02 remove rdg_loss 
        # total_loss = mlm_loss + cfg_loss + dfg_loss
        
        # # 03 remove dfg_loss
        # total_loss = mlm_loss + cfg_loss + rdg_loss
        
        # # 04 remove cfg_loss
        # total_loss = mlm_loss + cfg_loss + dfg_loss 
        
        # 04 remove cfg_loss
        total_loss = cfg_loss + dfg_loss + rdg_loss
        
        logger.info(f"finish welkir_model  total_loss:{total_loss}")
        
        
        return total_loss 



class WelkirForVuldetectModel(WelkirPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        # logger.error(f"config.problem_type:{self.config.problem_type}, self.num_labels:{self.num_labels }")

        self.welkir_model = WelkirModel(config, add_pooling_layer=False)
        self.classifier = WelkirClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: torch.Tensor,
        all_inst_encoder_input_ids: List[torch.Tensor],
        total_group_count: torch.Tensor,
        input_inst_tokencount: torch.Tensor,
        input_instcount_list: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cfg_matrix: Optional[torch.Tensor] = None,
        dfg_matrix: Optional[torch.Tensor] = None,
        rdg_matrix: Optional[torch.Tensor] = None,

    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # logger.error(f"config.problem_type:{self.config.problem_type}, self.num_labels:{self.num_labels }")

        
        # 获取模型输出
        outputs = self.welkir_model(
            input_ids=input_ids,
            all_inst_encoder_input_ids=all_inst_encoder_input_ids,
            total_group_count=total_group_count,
            input_inst_tokencount=input_inst_tokencount,
            input_instcount_list=input_instcount_list,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_inst_encoder_pooled_outputs=True,
            return_dict=return_dict,
            cfg_matrix=cfg_matrix,
            dfg_matrix=dfg_matrix,
            rdg_matrix=rdg_matrix,
        )


        # sequence_output = outputs[0]
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        #logger.info(f"WelkirForVuldetectModel logits :{logits}")
        # if labels.shape != logits.shape:
        #     # 调整 labels 的形状
        #     labels = labels.view(logits.shape)


        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # logger.info(f"WelkirForVuldetectModel labels :{labels}, labels.shape:{labels.shape}")

            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return (
                SequenceClassifierOutput( loss=loss, logits=logits, 
                                         hidden_states=outputs.hidden_states,attentions=outputs.attentions,)
                , pooled_output)#  add  t-SNE Plots


class WelkirForVulclassifyModel(WelkirPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.num_labels = 5
        config.num_labels = 5
        self.config.problem_type = "single_label_classification"
        # logger.error(f"config.problem_type:{self.config.problem_type}, self.num_labels:{self.num_labels }")

        self.welkir_model = WelkirModel(config, add_pooling_layer=False)
        self.classifier = WelkirClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: torch.Tensor,
        all_inst_encoder_input_ids: List[torch.Tensor],
        total_group_count: torch.Tensor,
        input_inst_tokencount: torch.Tensor,
        input_instcount_list: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cfg_matrix: Optional[torch.Tensor] = None,
        dfg_matrix: Optional[torch.Tensor] = None,
        rdg_matrix: Optional[torch.Tensor] = None,

    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # logger.error(f"config.problem_type:{self.config.problem_type}, self.num_labels:{self.num_labels }")

        
        # 获取模型输出
        outputs = self.welkir_model(
            input_ids=input_ids,
            all_inst_encoder_input_ids=all_inst_encoder_input_ids,
            total_group_count=total_group_count,
            input_inst_tokencount=input_inst_tokencount,
            input_instcount_list=input_instcount_list,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_inst_encoder_pooled_outputs=True,
            return_dict=return_dict,
            cfg_matrix=cfg_matrix,
            dfg_matrix=dfg_matrix,
            rdg_matrix=rdg_matrix,
        )


        # sequence_output = outputs[0]
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        # logger.info(f" logits :{logits}, logits.shape:{logits.shape}, labels.shape:{labels.shape}")

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # logger.info(f"WelkirForVuldetectModel labels :{labels}, labels.shape:{labels.shape}")

            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))             

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # return SequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
        
        #  add  t-SNE Plots
        return (
                SequenceClassifierOutput( loss=loss, logits=logits, 
                                         hidden_states=outputs.hidden_states,attentions=outputs.attentions,)
                , pooled_output)
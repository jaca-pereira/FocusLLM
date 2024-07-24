import torch
from einops import einops
from transformers import MistralModel, MistralConfig, DynamicCache, Cache
from typing import Optional, Union, Tuple, List

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa, \
    _prepare_4d_causal_attention_mask
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast

from videollama2.model.language_model.ToMe import bipartite_soft_matching, merge_wavg

logger = logging.get_logger(__name__)

class FocusLLMModel(MistralModel):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.last_attention = None

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            if self.config.focus_llm and seq_length == 1:
                batch_size, seq_length = 1, 1
                input_ids = input_ids[0].unsqueeze(0)
                attention_mask = attention_mask[0].unsqueeze(0)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                if self.config.token_merging and decoder_layer.self_attn.layer_idx == self.config.merge_layer and seq_length > 1:
                    hidden_states_image_or_video = hidden_states[:, self.modal_token_index[0]:(
                                self.modal_token_index[0] + self.image_video_tokens)]
                    original_size = len(hidden_states_image_or_video[0])
                    ratio = len(hidden_states_image_or_video[0]) // self.config.ratio
                    merge, _ = bipartite_soft_matching(hidden_states_image_or_video, r=ratio)
                    hidden_states_image_or_video, after_size = merge_wavg(merge, hidden_states_image_or_video,
                                                                          self.config.pad_token)
                    ratio = len(hidden_states_image_or_video[0]) // (self.config.ratio * 2)
                    merge, _ = bipartite_soft_matching(hidden_states_image_or_video, r=ratio)
                    hidden_states_image_or_video, after_size = merge_wavg(merge, hidden_states_image_or_video,
                                                                          self.config.pad_token,
                                                                          original_size=original_size)
                    hidden_states = torch.cat((hidden_states[:, :self.modal_token_index[0].item()],
                                               hidden_states_image_or_video, hidden_states[:, (self.modal_token_index[
                                                                                                   0] + self.image_video_tokens):]),
                                              dim=1)
                    if attention_mask is not None:
                        attention_mask[..., (self.modal_token_index[0].item() + after_size):(
                                    self.modal_token_index[0] + self.image_video_tokens), :] = attention_mask[
                            0, 0, 0, -1].item()
                        attention_mask[..., (self.modal_token_index[0].item() + after_size):(
                                    self.modal_token_index[0] + self.image_video_tokens)] = attention_mask[
                            0, 0, 0, -1].item()
                elif not self.config.token_merging and decoder_layer.self_attn.layer_idx == self.config.merge_layer and seq_length > 1:
                    image_attention_score = self.last_attention.mean(dim=1)[:, -1, self.modal_token_index:(self.modal_token_index + self.image_video_tokens)]
                    image_attention_score = image_attention_score.flatten()
                    image_attention_score = image_attention_score / image_attention_score.norm(dim=0, keepdim=True)
                    top_attention_rank_index = torch.argsort(image_attention_score, dim=0, descending=True)
                    top_attention_rank_index = top_attention_rank_index[:self.image_video_tokens] # get top k tokens that make the same number of tokens as the original length
                    hidden_states_image_or_video = hidden_states[:, self.modal_token_index:(self.modal_token_index + self.image_video_tokens)]
                    hidden_states_image_or_video = einops.rearrange(hidden_states_image_or_video, '(b s) l d -> b (s l) d', b=1)
                    hidden_states_image_or_video = hidden_states_image_or_video[:, top_attention_rank_index, :]
                    hidden_states = torch.cat((hidden_states[..., :self.modal_token_index, :].mean(0).unsqueeze(0), hidden_states_image_or_video, hidden_states[..., (self.modal_token_index + self.image_video_tokens):, :].mean(0).unsqueeze(0)), dim=1)
                    position_ids = position_ids[0].unsqueeze(0)
                    if attention_mask is not None:
                        attention_mask = attention_mask[0].unsqueeze(0)

                    for i, (key_cache, value_cache) in enumerate(zip(past_key_values.key_cache, past_key_values.value_cache)): #TODO can this be turned into matrix operations?
                        key_cache_image_or_video = key_cache[..., self.modal_token_index:(self.modal_token_index + self.image_video_tokens), :]
                        value_cache_image_or_video = value_cache[..., self.modal_token_index:(self.modal_token_index + self.image_video_tokens), :]
                        key_cache_image_or_video = einops.rearrange(key_cache_image_or_video, '(b s) k l d -> b k (s l) d', b=1)
                        value_cache_image_or_video = einops.rearrange(value_cache_image_or_video, '(b s) v l d -> b v (s l) d', b=1)
                        key_cache_image_or_video = key_cache_image_or_video[..., top_attention_rank_index, :]
                        value_cache_image_or_video = value_cache_image_or_video[..., top_attention_rank_index, :]
                        key_cache = torch.cat((key_cache[..., :self.modal_token_index, :].mean(0).unsqueeze(0), key_cache_image_or_video, key_cache[..., (self.modal_token_index + self.image_video_tokens):, :].mean(0).unsqueeze(0)), dim=2)
                        value_cache = torch.cat((value_cache[..., :self.modal_token_index, :].mean(0).unsqueeze(0), value_cache_image_or_video, value_cache[..., (self.modal_token_index + self.image_video_tokens):, :].mean(0).unsqueeze(0)), dim=2)
                        past_key_values.key_cache[i] = key_cache
                        past_key_values.value_cache[i] = value_cache

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

                if not self.config.token_merging  and decoder_layer.self_attn.layer_idx == self.config.merge_layer - 1:
                        self.last_attention = layer_outputs[1]

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


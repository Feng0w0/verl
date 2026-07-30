# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from typing import Optional

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from verl.models.transformers.monkey_patch import is_transformers_version_in_range

# Import compatibility wrapper for flash_attn_supports_top_left_mask
from verl.utils.transformers_compat import flash_attn_supports_top_left_mask
from verl.utils.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
    validate_ulysses_config,
)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _ulysses_flash_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
    kv = (
        self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        .transpose(1, 2)
    )

    k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    # patch
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()
    if ulysses_sp_size > 1:
        validate_ulysses_config(self.num_heads, ulysses_sp_size)

        num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        k_pe = repeat_kv(k_pe, ulysses_sp_size)  # to keep heads=1 after a2a
        k_nope = repeat_kv(k_nope, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)
        q = gather_seq_scatter_heads(q, seq_dim=2, head_dim=1)
        k_pe = gather_seq_scatter_heads(k_pe, seq_dim=2, head_dim=1)
        k_nope = gather_seq_scatter_heads(k_nope, seq_dim=2, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)
        # (batch_size, num_head / sp_size, seq_length, head_size)
        full_q_len = q.size(2)  # full_q_len = seq_length

    else:
        full_q_len = q_len

    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    cos, sin = self.rotary_emb(value_states, seq_len=full_q_len)
    q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

    query_states = k_pe.new_empty(bsz, self.num_heads // ulysses_sp_size, full_q_len, self.q_head_dim)
    query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
    query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

    key_states = k_pe.new_empty(bsz, self.num_heads // ulysses_sp_size, full_q_len, self.q_head_dim)
    key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
    key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

    if self.q_head_dim != self.v_head_dim:
        value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout
    # [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        full_q_len,
        dropout=dropout_rate,
        sliding_window=None,
        is_causal=self.is_causal,
        use_top_left_mask=flash_attn_supports_top_left_mask(),
        position_ids=position_ids,  # important: pass position ids
        softmax_scale=self.softmax_scale,
    )

    if ulysses_sp_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)

    if self.q_head_dim != self.v_head_dim:
        attn_output = attn_output[:, :, :, : self.v_head_dim]

    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim).contiguous()
    attn_output = self.o_proj(attn_output)

    if is_transformers_version_in_range(min_version="4.53.0"):
        return attn_output, None
    else:
        return attn_output, None, None


def patch_kimi_k25_vision_flash_attn():
    """Patch flash_attn_varlen_func in the dynamically loaded KimiK25 modeling module
    with an NPU-compatible implementation using MindSpeed's npu_fusion_attention.
    Also patch KimiK25VLModel.forward to pad inputs_embeds for TP-aligned scatter."""
    try:
        from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention
    except ImportError:
        try:
            import torch_npu
            npu_fusion_attention = torch_npu.npu_fusion_attention
        except ImportError:
            npu_fusion_attention = None

    if npu_fusion_attention is not None:
        def _npu_flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            causal=False,
            softmax_scale=None,
            deterministic=False,
        ):
            head_num = q.shape[1]
            scale = softmax_scale if softmax_scale is not None else (1.0 / (q.shape[-1] ** 0.5))
            actual_seq_qlen = cu_seqlens_q.tolist() if isinstance(cu_seqlens_q, torch.Tensor) else list(cu_seqlens_q)
            actual_seq_kvlen = cu_seqlens_k.tolist() if isinstance(cu_seqlens_k, torch.Tensor) else list(cu_seqlens_k)
            output = npu_fusion_attention(
                q, k, v, head_num, "TND",
                scale=scale,
                keep_prob=1.0,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
            )
            return output[0]

        import sys
        for mod_name, mod in list(sys.modules.items()):
            if "modeling_kimi_k25" in mod_name and hasattr(mod, "flash_attn_varlen_func"):
                if mod.flash_attn_varlen_func is None:
                    mod.flash_attn_varlen_func = _npu_flash_attn_varlen_func
                    print(f"Patched flash_attn_varlen_func in {mod_name} with NPU implementation")

    _patch_kimi_k25_vl_forward_scatter_padding()


def _collapse_media_placeholders(
    inputs_embeds, attention_mask, input_ids, labels, position_ids,
    image_token_index, feature_lengths,
):
    """Collapse consecutive <|media_pad|> tokens into a single placeholder
    per image, keeping only the first token of each run and removing the rest.

    This is needed when vLLM rollout produces input_ids with expanded
    <|media_pad|> tokens (e.g., 240 consecutive placeholders for one image)
    but HF _merge_input_ids_with_image_features expects exactly 1 placeholder
    per image.

    Args:
        inputs_embeds: (seq_len, batch_size, hidden_dim) in SBD format
        attention_mask: (batch_size, seq_len)
        input_ids: (batch_size, seq_len)
        labels: (batch_size, seq_len) or None
        position_ids: (batch_size, seq_len) or None
        image_token_index: token ID for <|media_pad|>
        feature_lengths: list of feature lengths per image

    Returns:
        Collapsed versions of inputs_embeds, attention_mask, input_ids,
        labels, position_ids
    """
    batch_size, seq_len = input_ids.shape
    is_media = (input_ids == image_token_index)

    # A position is the start of a run if it is media AND the previous
    # position is NOT media (or it is position 0).
    run_start = is_media & ~torch.cat(
        [torch.zeros(batch_size, 1, dtype=torch.bool, device=input_ids.device),
         is_media[:, :-1]], dim=1)
    # A position is inside a run (not the start) if it is media AND the
    # previous position is also media.
    run_inner = is_media & torch.cat(
        [torch.zeros(batch_size, 1, dtype=torch.bool, device=input_ids.device),
         is_media[:, :-1]], dim=1)

    # Keep: all non-media positions + start of each media run
    keep = ~is_media | run_start  # (batch_size, seq_len)

    # Build index arrays for gathering kept positions per sample
    kept_counts = keep.sum(dim=1)  # (batch_size,)
    new_seq_len = kept_counts.max().item()

    new_input_ids = torch.zeros(batch_size, new_seq_len, dtype=input_ids.dtype, device=input_ids.device)
    new_attention_mask = torch.zeros(batch_size, new_seq_len, dtype=attention_mask.dtype, device=attention_mask.device)
    hidden_dim = inputs_embeds.shape[2]
    new_inputs_embeds = torch.zeros(new_seq_len, batch_size, hidden_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

    if labels is not None:
        new_labels = torch.full((batch_size, new_seq_len), -100, dtype=labels.dtype, device=labels.device)
    else:
        new_labels = None

    for b in range(batch_size):
        idxs = keep[b].nonzero(as_tuple=True)[0]
        n = idxs.shape[0]
        new_input_ids[b, :n] = input_ids[b, idxs]
        new_attention_mask[b, :n] = attention_mask[b, idxs]
        new_inputs_embeds[:n, b, :] = inputs_embeds[idxs, b, :]
        if labels is not None:
            new_labels[b, :n] = labels[b, idxs]

    # position_ids must be recomputed after collapse since token positions
    # have changed.  Set to None and let downstream code recompute.
    new_position_ids = None

    return new_inputs_embeds, new_attention_mask, new_input_ids, new_labels, new_position_ids


def _patch_kimi_k25_vl_forward_scatter_padding():
    """Patch KimiK25VLModel.forward to pad inputs_embeds so that the first dim
    is divisible by tensor_model_parallel_size before scatter_to_sequence_parallel_region.
    After merging image features, the sequence length may not be TP-aligned."""
    try:
        from megatron.bridge.models.kimi_vl.modeling_kimi_k25_vl import KimiK25VLModel
    except ImportError:
        return

    if hasattr(KimiK25VLModel, "_verl_patched_scatter_padding"):
        return

    _orig_forward = KimiK25VLModel.forward

    def _patched_forward(self, *args, **kwargs):
        result = _orig_forward(self, *args, **kwargs)
        return result

    import torch
    from megatron.core.tensor_parallel.mappings import scatter_to_sequence_parallel_region

    _orig_forward = KimiK25VLModel.forward

    def _patched_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                         inputs_embeds=None, pixel_values=None, image_grid_thw=None,
                         labels=None, runtime_gather_output=None, *, loss_mask=None):
        if self.pre_process:
            if inputs_embeds is None:
                inputs_embeds = self.language_model.embedding(input_ids=input_ids, position_ids=None)

            if pixel_values is not None:
                image_features = self._extract_image_features(pixel_values, image_grid_thw)
                image_features = self.mm_projector(image_features)
                inputs_embeds = inputs_embeds.to(image_features[0].dtype)

                image_token_index = self.config.media_placeholder_token_id
                if input_ids is not None:
                    media_mask = (input_ids == image_token_index)
                    if attention_mask is not None:
                        media_mask = media_mask & attention_mask.bool()
                    n_placeholders = media_mask.sum().item()
                else:
                    n_placeholders = 0
                total_features = sum(f.shape[0] for f in image_features)
                n_images = len(image_features)
                feature_lengths = [f.shape[0] for f in image_features]

                # Determine which branch to take
                if n_placeholders == n_images:
                    branch = "MERGE (unexpanded)"
                elif n_placeholders == total_features:
                    branch = "INJECT (fully expanded)"
                elif n_placeholders > n_images and n_placeholders % n_images == 0:
                    branch = f"REPEAT+MERGE (multi-turn/expanded: ph={n_placeholders} img={n_images})"
                else:
                    branch = f"COLLAPSE+MERGE (fallback: ph={n_placeholders} img={n_images} feat={total_features})"

                print(f"[KimiK25-VL] branch={branch} | n_placeholders={n_placeholders} "
                      f"n_images={n_images} total_features={total_features} "
                      f"feature_lengths={feature_lengths} "
                      f"input_ids_shape={list(input_ids.shape) if input_ids is not None else None} "
                      f"pixel_values_shape={list(pixel_values.shape) if pixel_values is not None else None} "
                      f"image_grid_thw={image_grid_thw}")

                if n_placeholders == n_images:
                    # Unexpanded: 1 placeholder per image → HF merge
                    inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()
                    inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                        image_features, inputs_embeds, input_ids, attention_mask, labels,
                    )
                    inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()
                elif n_placeholders == total_features:
                    # Fully expanded: 1 placeholder per feature slot → direct inject
                    inputs_embeds_btd = inputs_embeds.transpose(1, 0).contiguous()
                    flat_features = torch.cat(image_features, dim=0)
                    mask = (input_ids == image_token_index)
                    if attention_mask is not None:
                        mask = mask & attention_mask.bool()
                    inputs_embeds_btd[mask] = flat_features.to(inputs_embeds_btd.dtype)
                    inputs_embeds = inputs_embeds_btd.transpose(1, 0).contiguous()
                elif n_placeholders > n_images and n_placeholders % n_images == 0:
                    # More placeholders than images, and evenly divisible.
                    # This happens in multi-turn conversations where the same image
                    # appears in multiple turns (each turn adds a <|media_pad|>),
                    # or when vLLM partially expanded placeholders.
                    # Repeat image_features so that each placeholder has a
                    # corresponding feature entry, then use HF merge.
                    repeat = n_placeholders // n_images
                    expanded_image_features = image_features * repeat
                    inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()
                    inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                        expanded_image_features, inputs_embeds, input_ids, attention_mask, labels,
                    )
                    inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()
                else:
                    # Fallback: collapse consecutive placeholders then merge
                    inputs_embeds, attention_mask, input_ids, labels, position_ids = \
                        _collapse_media_placeholders(
                            inputs_embeds, attention_mask, input_ids, labels,
                            position_ids, image_token_index, feature_lengths,
                        )
                    inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()
                    inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                        image_features, inputs_embeds, input_ids, attention_mask, labels,
                    )
                    inputs_embeds = inputs_embeds.transpose(1, 0).contiguous()

        if self.config.sequence_parallel:
            seq_len = inputs_embeds.shape[0]
            tp_size = self.config.tensor_model_parallel_size
            pad_len = (tp_size - seq_len % tp_size) % tp_size
            if pad_len > 0:
                pad_tensor = torch.zeros(pad_len, inputs_embeds.shape[1], inputs_embeds.shape[2],
                                         dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                inputs_embeds = torch.cat([inputs_embeds, pad_tensor], dim=0)
                if attention_mask is not None:
                    pad_mask = torch.zeros(attention_mask.shape[0], pad_len,
                                           dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat([attention_mask, pad_mask], dim=-1)
            inputs_embeds = scatter_to_sequence_parallel_region(inputs_embeds)

        # NPU flash attention requires specific mask format; let Megatron build
        # causal mask internally instead of passing a bool attention_mask.
        from verl.utils.device import is_npu_available
        if is_npu_available:
            attention_mask = None

        outputs = self.language_model.forward(
            input_ids=None, position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=inputs_embeds,
            labels=labels, loss_mask=loss_mask,
            runtime_gather_output=runtime_gather_output,
        )
        return outputs

    KimiK25VLModel.forward = _patched_forward
    KimiK25VLModel._verl_patched_scatter_padding = True
    print("Patched KimiK25VLModel.forward with TP-aligned scatter padding")
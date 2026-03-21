"""Converts Moonshine Streaming models from HuggingFace to CTranslate2 format."""

import gc

import numpy as np
import torch

from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import common_spec, moonshine_spec


class MoonshineConverter(Converter):
    """Converts a Moonshine Streaming model to CTranslate2 format.

    Usage:
        converter = MoonshineConverter("UsefulSensors/moonshine-streaming-medium")
        converter.convert("moonshine-medium-ct2", quantization="float16")
    """

    def __init__(self, model_name_or_path: str):
        self._model_name_or_path = model_name_or_path

    def _load(self):
        import transformers

        with torch.no_grad():
            model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
                self._model_name_or_path,
                dtype=torch.float32,
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self._model_name_or_path,
            )

        spec = self._build_spec(model, tokenizer)
        return spec

    def _build_spec(self, model, tokenizer):
        config = model.config
        enc_config = config.encoder_config

        # Per-layer sliding windows
        sliding_windows = None
        if hasattr(enc_config, "sliding_windows"):
            sliding_windows = [list(sw) for sw in enc_config.sliding_windows]

        # RoPE parameters
        rope_params = getattr(config, "rope_parameters", None)
        if rope_params:
            partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
            rope_theta = rope_params.get("rope_theta", 10000.0)
        else:
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            rope_theta = getattr(config, "rope_theta", 10000.0)

        head_dim = config.head_dim
        rotary_dim = int(head_dim * partial_rotary_factor)

        num_heads = config.num_attention_heads
        num_heads_kv = getattr(config, "num_key_value_heads", num_heads)
        enc_num_heads = enc_config.num_attention_heads
        enc_num_heads_kv = getattr(enc_config, "num_key_value_heads", enc_num_heads)

        # Check if adapter projection is needed
        adapter_project = enc_config.hidden_size != config.hidden_size

        spec = moonshine_spec.MoonshineSpec(
            num_encoder_layers=enc_config.num_hidden_layers,
            num_decoder_layers=config.num_hidden_layers,
            num_encoder_heads=enc_num_heads,
            num_decoder_heads=num_heads,
            num_encoder_heads_kv=enc_num_heads_kv if enc_num_heads_kv != enc_num_heads else None,
            num_decoder_heads_kv=num_heads_kv if num_heads_kv != num_heads else None,
            encoder_head_dim=enc_config.head_dim,
            decoder_head_dim=config.head_dim,
            sliding_windows=sliding_windows,
            rotary_dim=rotary_dim,
            rotary_interleave=False,
            rotary_base=rope_theta,
            adapter_project=adapter_project,
        )

        hf_model = model.model

        self._set_encoder(spec.encoder, hf_model.encoder)
        self._set_adapter(spec.adapter, hf_model.decoder, adapter_project)
        self._set_decoder(spec.decoder, hf_model.decoder)
        spec.decoder.projection.weight = model.proj_out.weight

        # Vocabulary
        vocab = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])
        tokens = [t for t, _ in vocab]
        while len(tokens) < config.vocab_size:
            tokens.append(f"<extra_id_{len(tokens)}>")
        if len(tokens) > config.vocab_size:
            tokens = tokens[:config.vocab_size]
        spec.register_vocabulary(tokens)

        # Config — update the default config (created during spec.__init__)
        bos = tokenizer.bos_token or tokenizer.convert_ids_to_tokens(config.bos_token_id)
        eos = tokenizer.eos_token or tokenizer.convert_ids_to_tokens(config.eos_token_id)
        unk = getattr(tokenizer, "unk_token", None) or "<unk>"
        spec.config.bos_token = bos
        spec.config.eos_token = eos
        spec.config.unk_token = unk

        return spec

    def _set_encoder(self, spec, encoder):
        # Frontend
        fe = spec.frontend
        fe.log_k = encoder.embedder.comp.log_k.detach()
        fe.linear.weight = encoder.embedder.linear.weight
        if hasattr(encoder.embedder.linear, "bias") and encoder.embedder.linear.bias is not None:
            fe.linear.bias = encoder.embedder.linear.bias
        fe.conv1.weight = encoder.embedder.conv1.weight
        fe.conv1.bias = encoder.embedder.conv1.bias
        fe.conv2.weight = encoder.embedder.conv2.weight
        fe.conv2.bias = encoder.embedder.conv2.bias

        # Final layer norm — MoonshineStreamingLayerNorm uses unit_offset:
        # effective_gamma = gamma + 1.0 (unit_offset=True)
        # CT2 LayerNorm applies: LN(x) * gamma + beta
        # So we save gamma+1.0 as gamma, and 0 as beta.
        spec.layer_norm.gamma = encoder.final_norm.gamma + encoder.final_norm.unit_offset
        spec.layer_norm.beta = torch.zeros_like(encoder.final_norm.gamma)

        # Layers
        for layer_spec, layer in zip(spec.layer, encoder.layers):
            # Self-attention norm (apply unit_offset)
            layer_spec.self_attention.layer_norm.gamma = (
                layer.input_layernorm.gamma + layer.input_layernorm.unit_offset)
            layer_spec.self_attention.layer_norm.beta = torch.zeros_like(
                layer.input_layernorm.gamma)

            # Self-attention: fuse Q+K+V → linear[0], O → linear[1]
            wq = layer.self_attn.q_proj.weight
            wk = layer.self_attn.k_proj.weight
            wv = layer.self_attn.v_proj.weight
            layer_spec.self_attention.linear[0].weight = torch.cat([wq, wk, wv])
            layer_spec.self_attention.linear[1].weight = layer.self_attn.o_proj.weight

            # FFN norm (apply unit_offset)
            layer_spec.ffn.layer_norm.gamma = (
                layer.post_attention_layernorm.gamma + layer.post_attention_layernorm.unit_offset)
            layer_spec.ffn.layer_norm.beta = torch.zeros_like(
                layer.post_attention_layernorm.gamma)

            # FFN: standard GELU (NOT SwiGLU for encoder)
            layer_spec.ffn.linear_0.weight = layer.mlp.fc1.weight
            layer_spec.ffn.linear_0.bias = layer.mlp.fc1.bias
            layer_spec.ffn.linear_1.weight = layer.mlp.fc2.weight
            layer_spec.ffn.linear_1.bias = layer.mlp.fc2.bias

            delattr(layer, "self_attn")
            delattr(layer, "mlp")
            gc.collect()

    def _set_adapter(self, spec, decoder_module, has_projection):
        spec.position_embeddings.weight = decoder_module.pos_emb.weight

        if has_projection and hasattr(decoder_module, "proj") and not isinstance(
            decoder_module.proj, torch.nn.Identity
        ):
            spec.projection.weight = decoder_module.proj.weight

    def _set_decoder(self, spec, decoder_module):
        spec.scale_embeddings = False
        spec.embeddings.weight = decoder_module.embed_tokens.weight

        # Final norm (standard LayerNorm, but Moonshine stores weight only → add zero beta)
        spec.layer_norm.gamma = decoder_module.norm.weight
        spec.layer_norm.beta = torch.zeros_like(decoder_module.norm.weight)

        for layer_spec, layer in zip(spec.layer, decoder_module.layers):
            # Self-attention norm
            layer_spec.self_attention.layer_norm.gamma = layer.input_layernorm.weight
            layer_spec.self_attention.layer_norm.beta = torch.zeros_like(
                layer.input_layernorm.weight
            )

            # Self-attention: fuse Q+K+V → linear[0], O → linear[1]
            wq = layer.self_attn.q_proj.weight
            wk = layer.self_attn.k_proj.weight
            wv = layer.self_attn.v_proj.weight
            layer_spec.self_attention.linear[0].weight = torch.cat([wq, wk, wv])
            layer_spec.self_attention.linear[1].weight = layer.self_attn.o_proj.weight

            # Cross-attention norm
            layer_spec.attention.layer_norm.gamma = layer.post_attention_layernorm.weight
            layer_spec.attention.layer_norm.beta = torch.zeros_like(
                layer.post_attention_layernorm.weight
            )

            # Cross-attention: Q → linear[0], K+V → linear[1], O → linear[2]
            layer_spec.attention.linear[0].weight = layer.encoder_attn.q_proj.weight
            layer_spec.attention.linear[1].weight = torch.cat([
                layer.encoder_attn.k_proj.weight,
                layer.encoder_attn.v_proj.weight,
            ])
            layer_spec.attention.linear[2].weight = layer.encoder_attn.o_proj.weight

            # FFN norm
            layer_spec.ffn.layer_norm.gamma = layer.final_layernorm.weight
            layer_spec.ffn.layer_norm.beta = torch.zeros_like(
                layer.final_layernorm.weight
            )

            # FFN: fused SwiGLU split
            # HF: fc1(x).chunk(2, dim=-1) → (value, gate)
            # CT2: linear_0 = value path (with activation), linear_0_noact = gate path
            fc1_w = layer.mlp.fc1.weight
            fc1_b = layer.mlp.fc1.bias
            mid = fc1_w.shape[0] // 2
            layer_spec.ffn.linear_0.weight = fc1_w[:mid]
            layer_spec.ffn.linear_0.bias = fc1_b[:mid]
            layer_spec.ffn.linear_0_noact.weight = fc1_w[mid:]
            layer_spec.ffn.linear_0_noact.bias = fc1_b[mid:]

            layer_spec.ffn.linear_1.weight = layer.mlp.fc2.weight
            layer_spec.ffn.linear_1.bias = layer.mlp.fc2.bias

            delattr(layer, "self_attn")
            delattr(layer, "encoder_attn")
            delattr(layer, "mlp")
            gc.collect()

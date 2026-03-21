"""Declares specification of the Moonshine Streaming ASR model."""

from typing import List, Optional, Tuple

import numpy as np

from ctranslate2.specs import attention_spec, common_spec, model_spec, transformer_spec


class MoonshineConfig(model_spec.LanguageModelConfig):
    """Configuration for the Moonshine model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MoonshineAudioFrontendSpec(model_spec.LayerSpec):
    """Audio preprocessing frontend: CMVN → asinh → Linear → SiLU → CausalConv1d × 2."""

    def __init__(self):
        self.log_k = None                       # learned asinh compression parameter (scalar)
        self.linear = common_spec.LinearSpec()   # Linear(frame_size → hidden_size)
        self.conv1 = common_spec.Conv1DSpec()    # CausalConv1d(hidden → hidden*2, k=5, s=2)
        self.conv2 = common_spec.Conv1DSpec()    # CausalConv1d(hidden*2 → hidden, k=5, s=2)


class MoonshineAdapterSpec(model_spec.LayerSpec):
    """Encoder→decoder adapter: position embeddings + optional linear projection."""

    def __init__(self, project: bool = True):
        self.position_embeddings = common_spec.EmbeddingsSpec()  # [max_pos, dec_hidden]
        if project:
            self.projection = common_spec.LinearSpec()           # Linear(enc_hidden → dec_hidden, no bias)


class MoonshineEncoderSpec(model_spec.LayerSpec):
    """Moonshine streaming encoder.

    Architecture:
      - Audio frontend (CausalConv1d-based, raw waveform input)
      - N transformer encoder layers with per-layer sliding window attention
      - Final layer norm
      - No position embeddings (positions captured by sliding window pattern)
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_heads_kv: Optional[int] = None,
        head_dim: Optional[int] = None,
        sliding_windows: Optional[List[Tuple[int, int]]] = None,
        ffn_glu: bool = False,
        rms_norm: bool = False,
    ):
        """Initializes the Moonshine encoder specification.

        Args:
          num_layers: Number of transformer layers.
          num_heads: Number of attention heads.
          num_heads_kv: Number of KV heads (for GQA). None = same as num_heads.
          head_dim: Dimension per attention head. None = inferred from model dim.
          sliding_windows: Per-layer sliding window config as list of [left, right] pairs.
            If None, no sliding window. Length must equal num_layers.
          ffn_glu: Use gated linear units (SwiGLU) in FFN layers.
          rms_norm: Use RMSNorm instead of LayerNorm.
        """
        self.num_heads = np.dtype("int16").type(num_heads)
        self.pre_norm = True
        self.activation = np.dtype("int8").type(common_spec.Activation.GELU)

        # Audio frontend (replaces Whisper's mel + Conv1d + position embeddings)
        self.frontend = MoonshineAudioFrontendSpec()

        # Final layer norm
        self.layer_norm = common_spec.LayerNormSpec(rms_norm=rms_norm)

        # Validate sliding windows
        if sliding_windows is not None and len(sliding_windows) != num_layers:
            raise ValueError(
                f"sliding_windows length ({len(sliding_windows)}) must equal "
                f"num_layers ({num_layers})"
            )

        # Transformer encoder layers with per-layer sliding window
        self.layer = []
        for i in range(num_layers):
            sw = sliding_windows[i] if sliding_windows is not None else None
            self.layer.append(
                transformer_spec.TransformerEncoderLayerSpec(
                    ffn_glu=ffn_glu,
                    rms_norm=rms_norm,
                    num_heads_kv=num_heads_kv,
                    head_dim=head_dim,
                    sliding_window=sw,
                )
            )


class MoonshineSpec(model_spec.LanguageModelSpec):
    """Describes a Moonshine Streaming ASR model.

    Architecture:
      - Encoder: audio frontend + transformer encoder with sliding window attention
      - Adapter: position embeddings + linear projection (bridges encoder→decoder dims)
      - Decoder: transformer decoder with RoPE + SwiGLU + cross-attention
    """

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_encoder_heads: int,
        num_decoder_heads: int,
        num_encoder_heads_kv: Optional[int] = None,
        num_decoder_heads_kv: Optional[int] = None,
        encoder_head_dim: Optional[int] = None,
        decoder_head_dim: Optional[int] = None,
        sliding_windows: Optional[List[Tuple[int, int]]] = None,
        rotary_dim: Optional[int] = None,
        rotary_interleave: bool = False,
        rotary_base: float = 10000,
        adapter_project: bool = True,
    ):
        """Initializes the Moonshine model specification.

        Args:
          num_encoder_layers: Number of encoder layers.
          num_decoder_layers: Number of decoder layers.
          num_encoder_heads: Number of encoder attention heads.
          num_decoder_heads: Number of decoder attention heads.
          num_encoder_heads_kv: Number of encoder KV heads (GQA). None = MHA.
          num_decoder_heads_kv: Number of decoder KV heads (GQA). None = MHA.
          encoder_head_dim: Encoder head dimension. None = inferred.
          decoder_head_dim: Decoder head dimension. None = inferred.
          sliding_windows: Per-layer encoder sliding window [[left, right], ...].
          rotary_dim: RoPE dimension for decoder (partial rotary). 0 = all dims.
          rotary_interleave: RoPE interleave mode. False for Moonshine.
          rotary_base: RoPE theta base. 10000 for Moonshine.
        """
        super().__init__()

        self.encoder = MoonshineEncoderSpec(
            num_layers=num_encoder_layers,
            num_heads=num_encoder_heads,
            num_heads_kv=num_encoder_heads_kv,
            head_dim=encoder_head_dim,
            sliding_windows=sliding_windows,
            rms_norm=False,  # encoder uses standard LayerNorm (not RMSNorm)
        )

        self.adapter = MoonshineAdapterSpec(project=adapter_project)

        self.decoder = transformer_spec.TransformerDecoderSpec(
            num_layers=num_decoder_layers,
            num_heads=num_decoder_heads,
            pre_norm=True,
            activation=common_spec.Activation.SWISH,
            with_encoder_attention=True,
            ffn_glu=True,
            rms_norm=False,
            num_heads_kv=num_decoder_heads_kv,
            head_dim=decoder_head_dim,
            rotary_dim=rotary_dim,
            rotary_interleave=rotary_interleave,
            rotary_base=rotary_base,
        )
        self.decoder.scale_embeddings = False

    @property
    def name(self):
        return "MoonshineSpec"

    @property
    def revision(self):
        return 1

    def get_default_config(self):
        return MoonshineConfig()

    def get_vocabulary_size(self):
        return self.decoder.embeddings.weight.shape[0]

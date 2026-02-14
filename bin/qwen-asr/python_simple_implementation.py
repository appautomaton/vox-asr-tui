#!/usr/bin/env python3
"""
Standalone inference for Qwen3-ASR (0.6B and 1.7B).
No transformers dependency - just PyTorch + safetensors + soundfile.

Usage:
    pip install torch safetensors soundfile
    python python_simple_implementation.py qwen3-asr-1.7b samples/test_speech.wav
    python python_simple_implementation.py qwen3-asr-0.6b samples/test_speech.wav

Reconstructed from HuggingFace transformers modeling code:
  - modeling_qwen3_asr.py (Qwen3ASRAudioEncoder, Qwen3ASRThinkerForConditionalGeneration)
  - Qwen3 text model (GQA, Q/K norms, MRoPE)
"""

import sys, os, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
import soundfile as sf

# ============================================================================
# Config detection from model directory
# ============================================================================

def load_config(model_dir):
    """Load model config and return parameters dict."""
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)

    tc = cfg["thinker_config"]
    ac = tc["audio_config"]
    txc = tc["text_config"]

    return {
        # Audio encoder
        "enc_d_model": ac["d_model"],
        "enc_layers": ac["encoder_layers"],
        "enc_heads": ac["encoder_attention_heads"],
        "enc_ffn_dim": ac["encoder_ffn_dim"],
        "enc_output_dim": ac["output_dim"],
        "enc_downsample_hidden": ac["downsample_hidden_size"],  # 480
        "enc_num_mel_bins": ac["num_mel_bins"],  # 128
        "enc_max_source_pos": ac["max_source_positions"],  # 1500
        "enc_n_window": ac["n_window"],  # 50
        "enc_n_window_infer": ac["n_window_infer"],  # 800
        "enc_conv_chunksize": ac.get("conv_chunksize", 500),
        # Text decoder
        "dec_hidden_size": txc["hidden_size"],
        "dec_layers": txc["num_hidden_layers"],
        "dec_heads": txc["num_attention_heads"],
        "dec_kv_heads": txc["num_key_value_heads"],
        "dec_head_dim": txc["head_dim"],
        "dec_intermediate": txc["intermediate_size"],
        "dec_rms_norm_eps": txc["rms_norm_eps"],
        "dec_rope_theta": txc["rope_theta"],
        "dec_mrope_section": txc["rope_scaling"]["mrope_section"],
        "dec_vocab_size": txc["vocab_size"],
        # Special tokens
        "audio_start_token_id": tc["audio_start_token_id"],
        "audio_end_token_id": tc["audio_end_token_id"],
        "audio_token_id": tc["audio_token_id"],
    }

# ============================================================================
# Audio preprocessing constants
# ============================================================================

SAMPLE_RATE = 16000
NUM_MEL_BINS = 128
HOP_LENGTH = 160
WINDOW_SIZE = 400  # n_fft

# Special token IDs (from tokenizer_config.json)
TOKEN_IM_START = 151644
TOKEN_IM_END = 151645
TOKEN_AUDIO_START = 151669
TOKEN_AUDIO_END = 151670
TOKEN_AUDIO_PAD = 151676
TOKEN_ENDOFTEXT = 151643
TOKEN_ASR_TEXT = 151704

# EOS token IDs (from generation_config.json)
EOS_TOKEN_IDS = {TOKEN_ENDOFTEXT, TOKEN_IM_END}

# Prompt token IDs (hardcoded from tokenizer)
# <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
PROMPT_PREFIX = [TOKEN_IM_START, 8948, 198, TOKEN_IM_END, 198,
                 TOKEN_IM_START, 872, 198, TOKEN_AUDIO_START]
# <|audio_end|><|im_end|>\n<|im_start|>assistant\n
PROMPT_SUFFIX = [TOKEN_AUDIO_END, TOKEN_IM_END, 198,
                 TOKEN_IM_START, 77091, 198]

# ============================================================================
# Mel filter bank (Slaney-style, matching WhisperFeatureExtractor)
# ============================================================================

def hertz_to_mel(freq):
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0
    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep
    return mels

def mel_to_hertz(mels):
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    log_region = mels >= min_log_mel
    freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return freq

def compute_mel_filters():
    num_frequency_bins = 1 + WINDOW_SIZE // 2  # 201
    fft_freqs = np.linspace(0, SAMPLE_RATE // 2, num_frequency_bins)
    mel_min = hertz_to_mel(0.0)
    mel_max = hertz_to_mel(8000.0)
    mel_freqs = np.linspace(mel_min, mel_max, NUM_MEL_BINS + 2)
    filter_freqs = mel_to_hertz(mel_freqs)
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    fb = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))
    enorm = 2.0 / (filter_freqs[2:NUM_MEL_BINS+2] - filter_freqs[:NUM_MEL_BINS])
    fb *= np.expand_dims(enorm, 0)
    return fb  # [201, 128]

def compute_mel_spectrogram(audio, mel_filters):
    """audio: 1D tensor, mel_filters: [freq_bins, mel_bins] tensor.
    Returns [mel_bins, frames] matching WhisperFeatureExtractor output.
    No padding - the encoder handles chunking internally.
    """
    window = torch.hann_window(WINDOW_SIZE)
    stft = torch.stft(audio, WINDOW_SIZE, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel_spec = mel_filters.T @ magnitudes  # [mel_bins, frames]
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec  # [128, frames]

# ============================================================================
# Weight loading helpers
# ============================================================================

class MultiSafetensors:
    """Load weights from one or more safetensors files."""
    def __init__(self, model_dir):
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        single_path = os.path.join(model_dir, "model.safetensors")

        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            shard_files = set(index["weight_map"].values())
            self.files = {}
            for shard in shard_files:
                path = os.path.join(model_dir, shard)
                self.files[shard] = safe_open(path, framework="pt")
            self.weight_map = index["weight_map"]
        else:
            self.files = {"model.safetensors": safe_open(single_path, framework="pt")}
            self.weight_map = None

    def get_tensor(self, name):
        if self.weight_map:
            shard = self.weight_map[name]
            return self.files[shard].get_tensor(name)
        else:
            for sf in self.files.values():
                try:
                    return sf.get_tensor(name)
                except:
                    continue
            raise KeyError(f"Weight not found: {name}")

def get_weight(sf, name):
    t = sf.get_tensor(name)
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t

# ============================================================================
# LayerNorm (used by encoder - standard LayerNorm with bias)
# ============================================================================

def layer_norm(x, weight, bias, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)

# ============================================================================
# RMSNorm (used by decoder)
# ============================================================================

def rms_norm(x, weight, eps=1e-6):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x.float() * torch.rsqrt(variance + eps)
    return (weight * x).to(x.dtype)

# ============================================================================
# Sinusoidal position embeddings (for encoder)
# ============================================================================

def sinusoidal_position_embedding(length, channels, max_timescale=10000):
    """Returns [length, channels] sinusoidal embeddings."""
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
    scaled_time = torch.arange(length).float().unsqueeze(1) * inv_timescales.unsqueeze(0)
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

# ============================================================================
# RoPE for decoder (interleaved MRoPE)
# ============================================================================

def compute_rope_freqs(positions, head_dim, theta):
    """positions: [seq_len] int tensor.
    Returns cos, sin each [seq_len, head_dim] (full head_dim).

    Qwen3 uses: freqs = inv_freq @ positions -> [seq, hd/2]
    then emb = cat(freqs, freqs) -> [seq, hd]
    cos, sin = emb.cos(), emb.sin()
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)  # [seq, hd/2]
    emb = torch.cat([angles, angles], dim=-1)  # [seq, hd]
    return torch.cos(emb), torch.sin(emb)

def apply_rope_neox(x, cos_f, sin_f, n_heads, head_dim):
    """Apply RoPE with NeoX/split-half style (Qwen3 style).
    x: [seq, n_heads, head_dim]
    cos_f, sin_f: [seq, head_dim]

    rotate_half: x1 = x[..., :hd/2], x2 = x[..., hd/2:]
    result = x * cos + rotate_half(x) * sin
    where rotate_half(x) = cat(-x2, x1)
    """
    cos_f = cos_f.unsqueeze(1)  # [seq, 1, hd]
    sin_f = sin_f.unsqueeze(1)
    # Split-half rotation
    half = head_dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos_f + rotated * sin_f

# ============================================================================
# Attention helpers
# ============================================================================

def full_attention(q, k, v, n_heads, n_kv_heads, head_dim, cu_seqlens=None):
    """Full (non-causal) attention for encoder, with optional windowed attention.
    q: [seq, n_heads*head_dim], k,v: [seq, n_kv_heads*head_dim]
    cu_seqlens: list of cumulative sequence lengths defining attention windows.
                e.g. [0, 104, 143] means tokens 0-103 attend together, 104-142 attend together.
    """
    seq_len = q.shape[0]
    gqa_ratio = n_heads // n_kv_heads

    if cu_seqlens is not None and len(cu_seqlens) > 2:
        # Process each window separately
        outputs = torch.zeros_like(q)
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            window_out = full_attention(
                q[start:end], k[start:end], v[start:end],
                n_heads, n_kv_heads, head_dim, cu_seqlens=None,
            )
            outputs[start:end] = window_out
        return outputs

    q = q.view(seq_len, n_heads, head_dim).transpose(0, 1).unsqueeze(0)
    k = k.view(seq_len, n_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
    v = v.view(seq_len, n_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)

    if gqa_ratio > 1:
        k = k.repeat_interleave(gqa_ratio, dim=1)
        v = v.repeat_interleave(gqa_ratio, dim=1)

    out = F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(),
        scale=1.0 / math.sqrt(head_dim),
        dropout_p=0.0,
    )
    return out.squeeze(0).transpose(0, 1).contiguous().view(seq_len, n_heads * head_dim)

def causal_attention(q, k, v, n_heads, n_kv_heads, head_dim, q_start_pos=0, kv_start_pos=0):
    """Causal attention for decoder with GQA.
    q: [seq_q, n_heads*head_dim], k,v: [seq_kv, n_kv_heads*head_dim]
    """
    seq_q = q.shape[0]
    seq_kv = k.shape[0]
    gqa_ratio = n_heads // n_kv_heads

    q = q.view(seq_q, n_heads, head_dim).transpose(0, 1).unsqueeze(0)
    k = k.view(seq_kv, n_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
    v = v.view(seq_kv, n_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)

    if gqa_ratio > 1:
        k = k.repeat_interleave(gqa_ratio, dim=1)
        v = v.repeat_interleave(gqa_ratio, dim=1)

    # Causal mask
    qi_abs = (q_start_pos + torch.arange(seq_q)).unsqueeze(1)
    kv_abs = (kv_start_pos + torch.arange(seq_kv)).unsqueeze(0)
    attn_mask = kv_abs <= qi_abs

    out = F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(),
        attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),
        scale=1.0 / math.sqrt(head_dim),
        dropout_p=0.0,
    )
    return out.squeeze(0).transpose(0, 1).contiguous().view(seq_q, n_heads * head_dim)

# ============================================================================
# Encoder forward
# ============================================================================

def encoder_forward(mel, sf, cfg):
    """mel: [128, frames] -> [n_audio_tokens, output_dim]

    The encoder applies Conv2D per-chunk of n_window*2 (100) mel frames,
    matching the official implementation. Each chunk of 100 frames produces
    13 output tokens after 3x Conv2D(stride=2).
    """
    prefix = "thinker.audio_tower"
    d_model = cfg["enc_d_model"]
    n_layers = cfg["enc_layers"]
    n_heads = cfg["enc_heads"]
    head_dim = d_model // n_heads
    ffn_dim = cfg["enc_ffn_dim"]
    downsample_hidden = cfg["enc_downsample_hidden"]
    n_window = cfg["enc_n_window"]  # 50
    chunk_size = n_window * 2  # 100 mel frames per chunk

    # ---- Conv2D stem (per-chunk) ----
    conv1_w = get_weight(sf, f"{prefix}.conv2d1.weight")
    conv1_b = get_weight(sf, f"{prefix}.conv2d1.bias")
    conv2_w = get_weight(sf, f"{prefix}.conv2d2.weight")
    conv2_b = get_weight(sf, f"{prefix}.conv2d2.bias")
    conv3_w = get_weight(sf, f"{prefix}.conv2d3.weight")
    conv3_b = get_weight(sf, f"{prefix}.conv2d3.bias")
    conv_out_w = get_weight(sf, f"{prefix}.conv_out.weight")

    total_frames = mel.shape[1]
    chunk_outputs = []

    # Process mel in chunks of chunk_size (100) frames
    for start in range(0, total_frames, chunk_size):
        end = min(start + chunk_size, total_frames)
        chunk_mel = mel[:, start:end]  # [128, chunk_len]

        # Input: [1, 1, mel_bins, chunk_len]
        x = chunk_mel.unsqueeze(0).unsqueeze(0)
        x = F.gelu(F.conv2d(x, conv1_w, conv1_b, stride=2, padding=1))
        x = F.gelu(F.conv2d(x, conv2_w, conv2_b, stride=2, padding=1))
        x = F.gelu(F.conv2d(x, conv3_w, conv3_b, stride=2, padding=1))

        # x: [1, 480, freq, time] -> [1, time, 480*freq]
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        chunk_outputs.append(x.squeeze(0))  # [time, 480*freq]

    # Concatenate all chunks
    x = torch.cat(chunk_outputs, dim=0)  # [total_tokens, 480*freq]
    print(f"  Conv output: {total_frames} frames -> {x.shape[0]} tokens (chunks of {chunk_size})", file=sys.stderr)

    # Linear projection to d_model
    x = F.linear(x, conv_out_w)  # [total_tokens, d_model] (no bias)
    seq_len = x.shape[0]
    print(f"  After conv_out projection: [{seq_len}, {d_model}]", file=sys.stderr)

    # ---- Sinusoidal position embeddings ----
    # Position embeddings are per-chunk (each chunk starts from 0)
    # The official model pads all chunks to max_chunk_len and uses the same pos emb
    # Since all our chunks are the same size (100 frames -> 13 tokens),
    # the max chunk token length is the tokens per full chunk
    tokens_per_chunk = chunk_outputs[0].shape[0]  # 13 for 100-frame chunks
    pos_emb = sinusoidal_position_embedding(tokens_per_chunk, d_model)
    # Apply pos emb per chunk (each chunk gets positions 0..chunk_len-1)
    offset = 0
    for chunk_out in chunk_outputs:
        chunk_len = chunk_out.shape[0]
        x[offset:offset + chunk_len] += pos_emb[:chunk_len]
        offset += chunk_len

    # ---- Compute cu_seqlens for windowed attention ----
    n_window_infer = cfg["enc_n_window_infer"]  # 800
    tokens_per_infer_window = tokens_per_chunk * (n_window_infer // chunk_size)  # 13 * 8 = 104

    # Compute aftercnn total length
    total_tokens = seq_len
    cu_seqlens = [0]
    pos = 0
    while pos < total_tokens:
        window_end = min(pos + tokens_per_infer_window, total_tokens)
        cu_seqlens.append(window_end)
        pos = window_end
    print(f"  Attention windows (cu_seqlens): {cu_seqlens}", file=sys.stderr)

    # ---- Transformer layers ----
    for layer in range(n_layers):
        lp = f"{prefix}.layers.{layer}"

        # Pre-attention LayerNorm
        ln_w = get_weight(sf, f"{lp}.self_attn_layer_norm.weight")
        ln_b = get_weight(sf, f"{lp}.self_attn_layer_norm.bias")
        x_norm = layer_norm(x, ln_w, ln_b)

        # Q, K, V projections (all with bias)
        wq = get_weight(sf, f"{lp}.self_attn.q_proj.weight")
        wq_b = get_weight(sf, f"{lp}.self_attn.q_proj.bias")
        wk = get_weight(sf, f"{lp}.self_attn.k_proj.weight")
        wk_b = get_weight(sf, f"{lp}.self_attn.k_proj.bias")
        wv = get_weight(sf, f"{lp}.self_attn.v_proj.weight")
        wv_b = get_weight(sf, f"{lp}.self_attn.v_proj.bias")
        wo = get_weight(sf, f"{lp}.self_attn.out_proj.weight")
        wo_b = get_weight(sf, f"{lp}.self_attn.out_proj.bias")

        q = F.linear(x_norm, wq, wq_b)
        k = F.linear(x_norm, wk, wk_b)
        v = F.linear(x_norm, wv, wv_b)

        # Windowed bidirectional attention (no RoPE in encoder)
        attn_out = full_attention(q, k, v, n_heads, n_heads, head_dim, cu_seqlens=cu_seqlens)

        # Output projection + residual
        x = x + F.linear(attn_out, wo, wo_b)

        # FFN with pre-LayerNorm
        ffn_ln_w = get_weight(sf, f"{lp}.final_layer_norm.weight")
        ffn_ln_b = get_weight(sf, f"{lp}.final_layer_norm.bias")
        x_norm = layer_norm(x, ffn_ln_w, ffn_ln_b)

        fc1_w = get_weight(sf, f"{lp}.fc1.weight")
        fc1_b = get_weight(sf, f"{lp}.fc1.bias")
        fc2_w = get_weight(sf, f"{lp}.fc2.weight")
        fc2_b = get_weight(sf, f"{lp}.fc2.bias")

        ffn_out = F.gelu(F.linear(x_norm, fc1_w, fc1_b))
        ffn_out = F.linear(ffn_out, fc2_w, fc2_b)
        x = x + ffn_out

        if (layer + 1) % 6 == 0 or layer == 0:
            print(f"  Encoder layer {layer+1}/{n_layers}: range [{x.min():.2f}, {x.max():.2f}]", file=sys.stderr)

    # ---- Final LayerNorm ----
    ln_post_w = get_weight(sf, f"{prefix}.ln_post.weight")
    ln_post_b = get_weight(sf, f"{prefix}.ln_post.bias")
    x = layer_norm(x, ln_post_w, ln_post_b)

    # ---- Projection layers ----
    proj1_w = get_weight(sf, f"{prefix}.proj1.weight")
    proj1_b = get_weight(sf, f"{prefix}.proj1.bias")
    proj2_w = get_weight(sf, f"{prefix}.proj2.weight")
    proj2_b = get_weight(sf, f"{prefix}.proj2.bias")

    x = F.gelu(F.linear(x, proj1_w, proj1_b))
    x = F.linear(x, proj2_w, proj2_b)

    print(f"  Encoder final output: [{x.shape[0]}, {x.shape[1]}]", file=sys.stderr)
    return x  # [n_audio_tokens, output_dim]

# ============================================================================
# Decoder
# ============================================================================

class Decoder:
    def __init__(self, sf, cfg):
        self.sf = sf
        self.cfg = cfg
        self.hidden_size = cfg["dec_hidden_size"]
        self.n_layers = cfg["dec_layers"]
        self.n_heads = cfg["dec_heads"]
        self.n_kv_heads = cfg["dec_kv_heads"]
        self.head_dim = cfg["dec_head_dim"]
        self.intermediate = cfg["dec_intermediate"]
        self.eps = cfg["dec_rms_norm_eps"]
        self.rope_theta = cfg["dec_rope_theta"]
        self.vocab_size = cfg["dec_vocab_size"]

        # Load embedding and LM head
        self.tok_embeddings = get_weight(sf, "thinker.model.embed_tokens.weight")
        self.lm_head = get_weight(sf, "thinker.lm_head.weight")
        self.final_norm = get_weight(sf, "thinker.model.norm.weight")

        # Preload all layer weights
        self.layers = []
        for i in range(self.n_layers):
            self.layers.append(self._load_layer(i))
            if (i + 1) % 8 == 0:
                print(f"  Decoder layer {i+1}/{self.n_layers} loaded", file=sys.stderr)

        self.kv_cache = {}

    def _load_layer(self, i):
        sf = self.sf
        lp = f"thinker.model.layers.{i}"
        return {
            "input_layernorm": get_weight(sf, f"{lp}.input_layernorm.weight"),
            "post_attention_layernorm": get_weight(sf, f"{lp}.post_attention_layernorm.weight"),
            "q_proj": get_weight(sf, f"{lp}.self_attn.q_proj.weight"),
            "k_proj": get_weight(sf, f"{lp}.self_attn.k_proj.weight"),
            "v_proj": get_weight(sf, f"{lp}.self_attn.v_proj.weight"),
            "o_proj": get_weight(sf, f"{lp}.self_attn.o_proj.weight"),
            "q_norm": get_weight(sf, f"{lp}.self_attn.q_norm.weight"),
            "k_norm": get_weight(sf, f"{lp}.self_attn.k_norm.weight"),
            "gate_proj": get_weight(sf, f"{lp}.mlp.gate_proj.weight"),
            "up_proj": get_weight(sf, f"{lp}.mlp.up_proj.weight"),
            "down_proj": get_weight(sf, f"{lp}.mlp.down_proj.weight"),
        }

    def embed_token(self, token_id):
        return self.tok_embeddings[token_id]

    def embed_tokens(self, token_ids):
        return self.tok_embeddings[token_ids]

    def _layer_forward(self, h, layer_idx, pos):
        """Single decoder layer forward.
        h: [seq, hidden_size], pos: starting position.
        """
        L = self.layers[layer_idx]
        seq_len = h.shape[0]

        # Pre-attention RMSNorm
        x_norm = rms_norm(h, L["input_layernorm"], self.eps)

        # Q, K, V (no bias in decoder)
        q = F.linear(x_norm, L["q_proj"])  # [seq, n_heads*head_dim]
        k = F.linear(x_norm, L["k_proj"])  # [seq, n_kv_heads*head_dim]
        v = F.linear(x_norm, L["v_proj"])

        # Reshape for per-head Q/K norm
        q = q.view(seq_len, self.n_heads, self.head_dim)
        k = k.view(seq_len, self.n_kv_heads, self.head_dim)

        # Per-head RMSNorm on Q and K (before RoPE)
        q = rms_norm(q, L["q_norm"], self.eps)
        k = rms_norm(k, L["k_norm"], self.eps)

        # RoPE (interleaved)
        positions = torch.arange(pos, pos + seq_len)
        rope_cos, rope_sin = compute_rope_freqs(positions, self.head_dim, self.rope_theta)
        q = apply_rope_neox(q, rope_cos, rope_sin, self.n_heads, self.head_dim)
        k = apply_rope_neox(k, rope_cos, rope_sin, self.n_kv_heads, self.head_dim)

        # Flatten back
        q = q.view(seq_len, self.n_heads * self.head_dim)
        k = k.view(seq_len, self.n_kv_heads * self.head_dim)
        v = v.view(seq_len, self.n_kv_heads * self.head_dim)

        # Update KV cache
        if layer_idx not in self.kv_cache:
            k_cache, v_cache = k, v
        else:
            k_cache, v_cache = self.kv_cache[layer_idx]
            k_cache = torch.cat([k_cache, k], dim=0)
            v_cache = torch.cat([v_cache, v], dim=0)
        self.kv_cache[layer_idx] = (k_cache, v_cache)

        # Causal attention
        kv_start_pos = (pos + seq_len - 1) - (k_cache.shape[0] - 1)
        attn_out = causal_attention(
            q, k_cache, v_cache,
            self.n_heads, self.n_kv_heads, self.head_dim,
            q_start_pos=pos, kv_start_pos=kv_start_pos,
        )

        # Output projection + residual
        h = h + F.linear(attn_out, L["o_proj"])

        # Pre-FFN RMSNorm
        x_norm = rms_norm(h, L["post_attention_layernorm"], self.eps)

        # SwiGLU MLP
        gate = F.silu(F.linear(x_norm, L["gate_proj"]))
        up = F.linear(x_norm, L["up_proj"])
        h = h + F.linear(gate * up, L["down_proj"])

        return h

    def prefill(self, input_embeds):
        """Prefill KV cache. input_embeds: [seq, hidden_size]."""
        self.kv_cache = {}
        h = input_embeds
        for layer in range(self.n_layers):
            h = self._layer_forward(h, layer, 0)
            if layer < 2 or (layer + 1) % 8 == 0:
                print(f"  Decoder prefill layer {layer+1}/{self.n_layers}: "
                      f"[{h.min():.2f}, {h.max():.2f}]", file=sys.stderr)
        return h

    def forward_one(self, embed, pos):
        """Generate one token. embed: [hidden_size], returns logits [vocab]."""
        h = embed.unsqueeze(0) if embed.dim() == 1 else embed

        for layer in range(self.n_layers):
            h = self._layer_forward(h, layer, pos)

        # Final RMSNorm
        h = rms_norm(h, self.final_norm, self.eps)

        # LM head (separate from embeddings)
        logits = F.linear(h.float().squeeze(0), self.lm_head)
        return logits

# ============================================================================
# Tokenizer (minimal BPE decoder from vocab.json)
# ============================================================================

def load_tokenizer(model_dir):
    """Load a minimal BPE decoder from vocab.json.
    We only need to decode token IDs to text (no encoding needed).
    """
    vocab_path = os.path.join(model_dir, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # vocab.json maps token_string -> token_id
    # Invert to get id -> string
    id_to_token = {v: k for k, v in vocab.items()}

    # Special tokens from tokenizer_config.json
    special_tokens = set()
    tc_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(tc_path):
        with open(tc_path) as f:
            tc = json.load(f)
        for tid_str, info in tc.get("added_tokens_decoder", {}).items():
            special_tokens.add(int(tid_str))

    def decode(token_ids):
        """Decode a list of token IDs to text."""
        pieces = []
        for tid in token_ids:
            if tid in special_tokens:
                # Include some special tokens in output for parsing
                if tid == TOKEN_ASR_TEXT:
                    pieces.append("<asr_text>")
                continue
            token_str = id_to_token.get(tid, "")
            if token_str:
                # BPE tokens use byte-level encoding:
                # Characters are mapped to visible Unicode chars
                # We need to decode them back to bytes
                pieces.append(token_str)
        text = "".join(pieces)
        # Decode byte-level BPE
        return bytearray([byte_decoder[c] for c in text if c in byte_decoder]).decode("utf-8", errors="replace")

    # Build byte decoder (reverse of GPT-2 byte encoding)
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}

    return decode

def bytes_to_unicode():
    """GPT-2 style byte-to-unicode mapping used by Qwen2 tokenizer."""
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("\xa1"), ord("\xac") + 1)) + \
         list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

# ============================================================================
# Full pipeline
# ============================================================================

def transcribe(model_dir, wav_path):
    # Load audio
    audio_array, sr = sf.read(wav_path, dtype='float32')
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if sr != SAMPLE_RATE:
        # Simple resampling (for production, use soxr or scipy)
        import warnings
        warnings.warn(f"Audio sample rate is {sr}, expected {SAMPLE_RATE}. Attempting resample.")
        try:
            import soxr
            audio_array = soxr.resample(audio_array, sr, SAMPLE_RATE, quality="HQ")
        except ImportError:
            # Simple linear resampling fallback
            ratio = SAMPLE_RATE / sr
            new_len = int(len(audio_array) * ratio)
            indices = np.linspace(0, len(audio_array) - 1, new_len)
            audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array).astype(np.float32)

    print(f"Audio: {len(audio_array)} samples ({len(audio_array)/SAMPLE_RATE:.1f}s)", file=sys.stderr)

    # Load config
    cfg = load_config(model_dir)
    print(f"Model: enc_d={cfg['enc_d_model']}, enc_layers={cfg['enc_layers']}, "
          f"dec_hidden={cfg['dec_hidden_size']}, dec_layers={cfg['dec_layers']}", file=sys.stderr)

    # Mel spectrogram
    mel_filters = torch.tensor(compute_mel_filters(), dtype=torch.float32)
    audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
    mel = compute_mel_spectrogram(audio_tensor, mel_filters)
    print(f"Mel spectrogram: [{mel.shape[0]}, {mel.shape[1]}]", file=sys.stderr)

    # Load weights
    print(f"Loading model from {model_dir}...", file=sys.stderr)
    sf_file = MultiSafetensors(model_dir)

    # Encoder
    print("Running encoder...", file=sys.stderr)
    with torch.no_grad():
        audio_embeds = encoder_forward(mel, sf_file, cfg)
    n_audio = audio_embeds.shape[0]
    print(f"Audio embeddings: [{n_audio}, {audio_embeds.shape[1]}]", file=sys.stderr)

    # Build prompt: system + user + audio_pad*n_audio + suffix + assistant
    input_ids = PROMPT_PREFIX + [TOKEN_AUDIO_PAD] * n_audio + PROMPT_SUFFIX
    print(f"Prompt length: {len(input_ids)} tokens ({n_audio} audio pads)", file=sys.stderr)

    # Load decoder
    print("Loading decoder...", file=sys.stderr)
    decoder = Decoder(sf_file, cfg)

    # Build input embeddings
    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    input_embeds = decoder.embed_tokens(input_ids_t)  # [prompt_len, hidden_size]

    # Replace audio_pad positions with audio embeddings
    audio_mask = (input_ids_t == TOKEN_AUDIO_PAD)
    audio_positions = audio_mask.nonzero(as_tuple=True)[0]
    assert len(audio_positions) == n_audio, f"Expected {n_audio} audio positions, got {len(audio_positions)}"
    input_embeds[audio_positions] = audio_embeds

    prompt_len = len(input_ids)
    print(f"Running decoder prefill ({prompt_len} tokens)...", file=sys.stderr)

    with torch.no_grad():
        # Prefill all but last position
        if prompt_len > 1:
            _ = decoder.prefill(input_embeds[:-1])
        # Generate first token from last position
        logits = decoder.forward_one(input_embeds[-1], pos=prompt_len - 1)
        token = int(logits.argmax().item())

    generated = [token]
    print(f"  First token: {token}", file=sys.stderr)

    # Autoregressive generation
    print("Running decoder generation...", file=sys.stderr)
    max_new_tokens = 1024
    with torch.no_grad():
        for step in range(max_new_tokens - 1):
            if token in EOS_TOKEN_IDS:
                break
            pos = prompt_len + step
            embed = decoder.embed_token(token)
            logits = decoder.forward_one(embed, pos=pos)
            token = int(logits.argmax().item())
            generated.append(token)

            if len(generated) <= 5:
                topk_vals, topk_idxs = torch.topk(logits, 5)
                print(f"  Token {len(generated)} (pos={pos+1}): {token}, "
                      f"top5: {list(zip(topk_idxs.tolist(), ['%.2f'%v for v in topk_vals.tolist()]))}",
                      file=sys.stderr)

    print(f"Generated {len(generated)} tokens", file=sys.stderr)

    # Remove EOS from output
    while generated and generated[-1] in EOS_TOKEN_IDS:
        generated = generated[:-1]

    # Decode tokens to text
    decode = load_tokenizer(model_dir)
    text = decode(generated).strip()

    # Parse ASR output: "language <lang><asr_text>transcription"
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]

    return text

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <audio.wav>", file=sys.stderr)
        sys.exit(1)

    model_dir = sys.argv[1]
    wav_path = sys.argv[2]

    text = transcribe(model_dir, wav_path)
    print(text)

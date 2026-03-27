"""
TurboQuant quantizers — Algorithm 1 (MSE) and Algorithm 2 (inner product).

These operate on backend arrays of shape (..., d) where d is the embedding
dimension (typically head_dim = 128 for modern LLMs).

Backend-agnostic: uses the backend abstraction layer for all array operations.
"""

import math
import numpy as np
from typing import NamedTuple

from turboquant_mac.codebook import get_codebook_arrays
from turboquant_mac.rotation import generate_rotation_matrix, generate_qjl_matrix
from turboquant_mac.backends import get_backend


class MSEQuantized(NamedTuple):
    """Output of TurboQuant MSE quantization."""
    indices: object    # (..., packed_len) uint8 bit-packed indices
    norms: object      # (...,) original L2 norms
    bits: int          # number of bits per index (for unpacking)


class ProdQuantized(NamedTuple):
    """Output of TurboQuant inner-product quantization."""
    mse_indices: object    # (..., packed_len) uint8 bit-packed MSE indices
    qjl_signs: object      # (..., packed_len) uint8 packed sign bits
    residual_norms: object # (...,) L2 norms of residual vectors
    norms: object          # (...,) original L2 norms
    mse_bits: int          # bits per MSE index (for unpacking)


def _get_packing_params(bits: int) -> tuple[int, int]:
    """Return (effective_bits, vals_per_byte) for packing."""
    if bits == 1:
        return 1, 8
    elif bits == 2:
        return 2, 4
    elif bits <= 4:
        return 4, 2  # round up to 4-bit packing
    else:
        return 8, 1


def _pack_indices(indices, bits: int, B):
    """Bit-pack integer indices into uint8 bytes."""
    eff_bits, vals_per_byte = _get_packing_params(bits)
    if vals_per_byte == 1:
        return B.to_uint8(indices)

    d = indices.shape[-1]
    batch_shape = indices.shape[:-1]

    # Pad to multiple of vals_per_byte
    padded_d = ((d + vals_per_byte - 1) // vals_per_byte) * vals_per_byte
    if padded_d > d:
        indices = B.pad(B.to_uint8(indices), padded_d - d, value=0)

    reshaped = B.reshape(B.to_uint8(indices), (*batch_shape, -1, vals_per_byte))
    shifts = B.to_uint8(B.arange(vals_per_byte)) * eff_bits
    packed = B.sum_(B.left_shift(reshaped, shifts), dim=-1, dtype=None)
    return B.to_uint8(packed)


def _unpack_indices(packed, bits: int, d: int, B):
    """Unpack bit-packed indices back to integer array."""
    eff_bits, vals_per_byte = _get_packing_params(bits)
    if vals_per_byte == 1:
        return B.to_long(packed)

    batch_shape = packed.shape[:-1]
    mask = (1 << eff_bits) - 1
    shifts = B.to_uint8(B.arange(vals_per_byte)) * eff_bits

    unpacked = B.bitwise_and(B.right_shift(B.unsqueeze(packed, -1), shifts), mask)
    unpacked = B.reshape(unpacked, (*batch_shape, -1))
    # Trim to original dimension
    if unpacked.shape[-1] > d:
        unpacked = unpacked[..., :d]
    return B.to_long(unpacked)


def _pack_qjl_signs(projected, dim: int, B):
    """Pack sign bits into uint8 (8 signs per byte)."""
    signs = B.to_uint8(B.greater_than(projected, 0))
    d = signs.shape[-1]
    if d % 8 != 0:
        signs = B.pad(signs, 8 - d % 8, value=0)
    batch_shape = signs.shape[:-1]
    signs_reshaped = B.reshape(signs, (*batch_shape, -1, 8))
    powers = B.to_uint8(B.tensor([1, 2, 4, 8, 16, 32, 64, 128]))
    packed = B.sum_(signs_reshaped * powers, dim=-1, dtype=None)
    return B.to_uint8(packed)


def _unpack_qjl_signs(packed, dim: int, B):
    """Unpack sign bits from uint8 to float {-1, +1}."""
    powers = B.to_uint8(B.tensor([1, 2, 4, 8, 16, 32, 64, 128]))
    unpacked = B.to_float(B.greater_than(B.bitwise_and(B.unsqueeze(packed, -1), powers), 0))
    batch_shape = packed.shape[:-1]
    unpacked = B.reshape(unpacked, (*batch_shape, -1))
    if unpacked.shape[-1] > dim:
        unpacked = unpacked[..., :dim]
    return unpacked * 2.0 - 1.0


class TurboQuantMSE:
    """
    TurboQuant optimized for MSE (Algorithm 1).

    Quantize: y = Pi * (x / ||x||), then find nearest centroid per coordinate.
    Dequantize: look up centroids, rotate back, rescale by ||x||.
    """

    def __init__(self, dim: int, bits: int = 3, seed: int = 42, backend: str = None):
        self.dim = dim
        self.bits = bits
        self.n_clusters = 2**bits

        B = get_backend(backend)
        self._backend_name = backend

        # Precompute rotation matrix (generated in NumPy, converted to backend)
        Pi_np = generate_rotation_matrix(dim, seed=seed)
        self.Pi = B.from_numpy(Pi_np)

        # Precompute codebook
        centroids_np, boundaries_np = get_codebook_arrays(dim, bits)
        self.centroids = B.from_numpy(centroids_np)
        self.boundaries = B.from_numpy(boundaries_np)
        # Interior boundaries for searchsorted
        self.decision_boundaries = B.from_numpy(boundaries_np[1:-1].copy())

    @property
    def B(self):
        return get_backend(self._backend_name)

    def quantize(self, x) -> MSEQuantized:
        """Quantize vectors x of shape (..., d)."""
        B = self.B
        norms = B.norm(x, dim=-1, keepdim=False)
        x_unit = x / (B.unsqueeze(norms, -1) + 1e-10)

        # Apply random rotation
        y = B.matmul(B.to_float(x_unit), B.transpose(self.Pi, 0, 1))

        # Fused Metal encode (searchsorted + bit-pack) when available
        if self._backend_name != "pytorch":
            try:
                from turboquant_mac.backends.metal_kernels import turboquant_mse_encode_metal
                packed = turboquant_mse_encode_metal(
                    y, self.decision_boundaries, self.bits, self.dim,
                )
                return MSEQuantized(indices=packed, norms=norms, bits=self.bits)
            except Exception:
                pass

        # Fallback: searchsorted + separate bit-pack
        indices = B.searchsorted(self.decision_boundaries, y)
        packed = _pack_indices(indices, self.bits, B)
        return MSEQuantized(indices=packed, norms=norms, bits=self.bits)

    def dequantize(self, q: MSEQuantized):
        """Reconstruct vectors from quantized representation."""
        B = self.B
        indices = _unpack_indices(q.indices, q.bits, self.dim, B)

        # Look up centroids
        y_hat = B.index_select(self.centroids, indices)

        # Rotate back
        x_hat = B.matmul(y_hat, self.Pi)

        # Rescale
        x_hat = x_hat * B.unsqueeze(q.norms, -1)
        return x_hat

    def forward(self, x):
        """Quantize and immediately dequantize (for testing)."""
        return self.dequantize(self.quantize(x))


class TurboQuantProd:
    """
    TurboQuant optimized for inner products (Algorithm 2).

    Two-stage:
      1. Apply TurboQuant_MSE at (b-1) bits -> get residual r = x - x_hat
      2. Apply QJL to residual: sign(S * r) -> 1 bit per coordinate
      3. Store ||r||_2 for rescaling

    The dequantized inner product estimate is unbiased: E[estimate] = <y, x>.
    """

    def __init__(self, dim: int, bits: int = 3, seed: int = 42, backend: str = None):
        self.dim = dim
        self.bits = bits
        self._backend_name = backend
        assert bits >= 2, "Inner product TurboQuant requires at least 2 bits"

        B = get_backend(backend)

        # Stage 1: MSE quantizer at (b-1) bits
        self.mse_quantizer = TurboQuantMSE(
            dim=dim, bits=bits - 1, seed=seed, backend=backend
        )

        # Stage 2: QJL projection matrix
        S_np = generate_qjl_matrix(dim, seed=seed + 1000)
        self.S = B.from_numpy(S_np)

        # QJL dequantization constant
        self.qjl_scale = math.sqrt(math.pi / 2.0) / dim

    @property
    def B(self):
        return get_backend(self._backend_name)

    def quantize(self, x) -> ProdQuantized:
        """Quantize vectors x of shape (..., d)."""
        B = self.B

        # Stage 1: MSE quantize at (b-1) bits
        mse_q = self.mse_quantizer.quantize(x)
        x_hat = self.mse_quantizer.dequantize(mse_q)

        # Compute residual
        residual = x - x_hat
        residual_norms = B.norm(residual, dim=-1)

        # Stage 2: QJL on residual
        projected = B.matmul(B.to_float(residual), B.transpose(self.S, 0, 1))
        packed_signs = _pack_qjl_signs(projected, self.dim, B)

        return ProdQuantized(
            mse_indices=mse_q.indices,
            qjl_signs=packed_signs,
            residual_norms=residual_norms,
            norms=mse_q.norms,
            mse_bits=mse_q.bits,
        )

    def dequantize(self, q: ProdQuantized):
        """Reconstruct vectors from quantized representation."""
        B = self.B
        mse_q = MSEQuantized(indices=q.mse_indices, norms=q.norms, bits=q.mse_bits)
        x_mse = self.mse_quantizer.dequantize(mse_q)

        signs = _unpack_qjl_signs(q.qjl_signs, self.dim, B)
        x_qjl = B.matmul(signs, self.S)
        x_qjl = x_qjl * (self.qjl_scale * B.unsqueeze(q.residual_norms, -1))

        return x_mse + x_qjl

    def _attention_score_metal(self, query, quantized_key: ProdQuantized):
        """Compute attention scores using fused Metal kernels (MLX only)."""
        from turboquant_mac.backends.metal_kernels import turboquant_attention_score_metal

        # Metal path expects (BH, D) for query — flatten batch dims
        orig_shape = query.shape  # (..., n_q, d)
        batch_dims = orig_shape[:-2]
        n_q = orig_shape[-2]

        # Flatten batch dims + n_q into single BH dimension
        flat_query = query.reshape(-1, self.dim)  # (BH*n_q, D)

        # For quantized data, flatten batch dims: (..., N, packed_d) -> (BH, N, packed_d)
        n_k = quantized_key.mse_indices.shape[-2]
        packed_d = quantized_key.mse_indices.shape[-1]
        packed_d_signs = quantized_key.qjl_signs.shape[-1]

        flat_mse = quantized_key.mse_indices.reshape(-1, n_k, packed_d)
        flat_signs = quantized_key.qjl_signs.reshape(-1, n_k, packed_d_signs)
        flat_norms = quantized_key.norms.reshape(-1, n_k)
        flat_res_norms = quantized_key.residual_norms.reshape(-1, n_k)

        n_batch_heads = flat_mse.shape[0]

        # Metal kernel expects (BH, D) query — run per-query if n_q > 1
        import mlx.core as mx
        if n_q == 1:
            scores_flat = turboquant_attention_score_metal(
                query=flat_query.reshape(n_batch_heads, self.dim),
                mse_packed=flat_mse,
                qjl_signs=flat_signs,
                norms=flat_norms,
                residual_norms=flat_res_norms,
                Pi=self.mse_quantizer.Pi,
                S=self.S,
                centroids=self.mse_quantizer.centroids,
                mse_bits=quantized_key.mse_bits,
                qjl_scale=self.qjl_scale,
            )
            return scores_flat.reshape(*batch_dims, 1, n_k)
        else:
            # Multi-query: tile quantized data and batch all queries together
            # Repeat quantized data n_q times along batch dim
            tiled_mse = mx.repeat(flat_mse, n_q, axis=0)
            tiled_signs = mx.repeat(flat_signs, n_q, axis=0)
            tiled_norms = mx.repeat(flat_norms, n_q, axis=0)
            tiled_res_norms = mx.repeat(flat_res_norms, n_q, axis=0)

            scores_flat = turboquant_attention_score_metal(
                query=flat_query,
                mse_packed=tiled_mse,
                qjl_signs=tiled_signs,
                norms=tiled_norms,
                residual_norms=tiled_res_norms,
                Pi=self.mse_quantizer.Pi,
                S=self.S,
                centroids=self.mse_quantizer.centroids,
                mse_bits=quantized_key.mse_bits,
                qjl_scale=self.qjl_scale,
            )
            return scores_flat.reshape(*batch_dims, n_q, n_k)

    def _attention_score_python(self, query, quantized_key: ProdQuantized):
        """Compute attention scores using Python path (any backend)."""
        B = self.B

        # Stage 1: MSE contribution
        mse_q = MSEQuantized(
            indices=quantized_key.mse_indices,
            norms=quantized_key.norms,
            bits=quantized_key.mse_bits,
        )
        k_mse = self.mse_quantizer.dequantize(mse_q)
        scores_mse = B.matmul(B.to_float(query), B.transpose(B.to_float(k_mse), -2, -1))

        # Stage 2: QJL contribution
        q_sketched = B.matmul(B.to_float(query), B.transpose(self.S, 0, 1))
        signs = _unpack_qjl_signs(quantized_key.qjl_signs, self.dim, B)

        scores_qjl = B.matmul(q_sketched, B.transpose(signs, -2, -1))
        scores_qjl = scores_qjl * (self.qjl_scale * B.unsqueeze(quantized_key.residual_norms, -2))

        return scores_mse + scores_qjl

    def attention_score(self, query, quantized_key: ProdQuantized):
        """
        Compute attention scores <query, key> using quantized keys.

        Auto-selects Metal kernels on MLX backend for fused GPU computation,
        falling back to Python path on PyTorch or when Metal is unavailable.

        Args:
            query: (..., n_q, d) — the query vectors
            quantized_key: ProdQuantized with shapes (..., n_k, ...)

        Returns:
            scores: (..., n_q, n_k) — the attention logits
        """
        if self._backend_name != "pytorch":
            try:
                return self._attention_score_metal(query, quantized_key)
            except Exception:
                pass
        return self._attention_score_python(query, quantized_key)

    def forward(self, x):
        """Quantize and immediately dequantize (for testing)."""
        return self.dequantize(self.quantize(x))

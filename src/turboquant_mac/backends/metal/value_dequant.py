"""
Metal shader source for fused value dequantization.

Takes packed uint8 value bytes, per-group scales and zeros, and outputs
float32 dequantized values in one pass. Combines bit-unpacking, int-to-float
conversion, and affine transform (val * scale + zero) into a single kernel.
"""

# Template parameters: {BITS}, {VALS_PER_BYTE}, {BIT_MASK}, {D}, {GROUP_SIZE}, {N_GROUPS}
VALUE_DEQUANT_SOURCE = """
    // Thread handles one (batch, coord) pair
    uint batch_idx = thread_position_in_grid.y;
    uint coord = thread_position_in_grid.x;

    uint N_BATCH = packed_shape[0];
    uint D = {D};
    uint BITS = {BITS};
    uint VALS_PER_BYTE = {VALS_PER_BYTE};
    uint BIT_MASK = {BIT_MASK};
    uint GROUP_SIZE = {GROUP_SIZE};
    uint N_GROUPS = {N_GROUPS};
    uint PACKED_D = {PACKED_D};

    if (batch_idx >= N_BATCH || coord >= D) return;

    // Find which packed byte and sub-element this coordinate maps to
    uint byte_idx = coord / VALS_PER_BYTE;
    uint sub = coord % VALS_PER_BYTE;

    // Extract quantized integer
    uint8_t packed_byte = packed[batch_idx * PACKED_D + byte_idx];
    uint qval = ((uint)packed_byte >> (sub * BITS)) & BIT_MASK;

    // Find which group this coordinate belongs to
    uint group_idx = coord / GROUP_SIZE;

    // Dequantize: val * scale + zero
    float scale_val = scales[batch_idx * N_GROUPS + group_idx];
    float zero_val = zeros[batch_idx * N_GROUPS + group_idx];
    float result = (float)qval * scale_val + zero_val;

    out[batch_idx * D + coord] = result;
"""


def get_value_dequant_source(bits: int, d: int, group_size: int) -> str:
    """Return Metal shader source with template parameters filled in."""
    if bits == 2:
        eff_bits, vals_per_byte = 2, 4
    elif bits == 4:
        eff_bits, vals_per_byte = 4, 2
    else:
        eff_bits, vals_per_byte = 8, 1

    bit_mask = (1 << eff_bits) - 1
    packed_d = d // vals_per_byte
    n_groups = d // group_size

    return VALUE_DEQUANT_SOURCE.format(
        BITS=eff_bits,
        VALS_PER_BYTE=vals_per_byte,
        BIT_MASK=bit_mask,
        D=d,
        PACKED_D=packed_d,
        GROUP_SIZE=group_size,
        N_GROUPS=n_groups,
    )

import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, p3, p4, p5: tl.constexpr, c0: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, c0)
    mask = lane < p5

    a0_row_ptr = p0 + row * p3
    a0_value = tl.load(a0_row_ptr + lane, mask=mask, other=0.0)

    a1_row_ptr = p1 + row * p3
    v7 = tl.load(a1_row_ptr + lane, mask=mask, other=0.0)

    a2_row_ptr = p2 + row * p3
    tl.store(a2_row_ptr + lane, v7, mask=mask)

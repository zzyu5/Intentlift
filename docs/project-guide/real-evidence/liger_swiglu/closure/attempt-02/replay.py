import triton
import triton.language as tl


@triton.jit
def scrc_k0(p0, p1, p2, p3, p4, p5: tl.constexpr, c0: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, triton.next_power_of_2(p5))
    v0 = tl.full((triton.next_power_of_2(p5),), True, tl.int1)
    v9 = v0 & (lane < p5)

    row_ptr_a0 = p0 + row * p3
    v1 = tl.load(row_ptr_a0 + lane, mask=v9, other=0).to(tl.float32)

    v2 = p4.to(tl.float32)
    v3 = v1 * v2
    v4 = tl.sigmoid(v3)
    v5 = v3 * v4
    v6 = v5.to(tl.float32)

    row_ptr_a1 = p1 + row * p3
    t1_value = tl.load(row_ptr_a1 + lane, mask=v9, other=0)
    v7 = v6 * t1_value

    row_ptr_a2 = p2 + row * p3
    tl.store(row_ptr_a2 + lane, v7, mask=v9)
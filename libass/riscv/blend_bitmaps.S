/*
 * Copyright (C) 2024 libass wangyuexiang
 *
 * This file is part of libass.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include "asm.S"

 /*
 * void ass_add_bitmaps(uint8_t *dst, ptrdiff_t dst_stride,
 *                      const uint8_t *src, ptrdiff_t src_stride,
 *                      size_t width, size_t height);
 */

function add_bitmaps_rvv
    mv a6, a0
    mv a7, a2
0:
    vsetvli t0, a4, e64, m2, ta, ma 
    mv a0, a6
    mv a2, a7
1:
    vle64.v v0, (a0)
    sub a4, a4, t0
    vle64.v v2, (a2)
    vminu.vv v0, v0, v2
    vse64.v v0, (a0)
    addi a0, a0, 16
    addi a2, a2, 16 
    bnew a4, 1b
2:
    addi a5, a5, -1
    add a6, a6, a1
    add a7, a7, a3
    bnew a5, 0b 
    ret
      
/*
 * void ass_imul_bitmaps(uint8_t *dst, ptrdiff_t dst_stride,
 *                       const uint8_t *src, ptrdiff_t src_stride,
 *                       size_t width, size_t height);
 */

function mul_bitmaps_rvv
    mv a6, a0
    mv a7, a2
0:
    vsetvli t0, a4, e64, m2, ta, ma  
    mv a0, a6
    mv a2, a7
1:
    vle64.v v0, (a0)                 
    sub a4, a4, t0
    vle64.v v2, (a2)                 
    vrsub.vi v2, v2, 255
    vmulvv v0, v0, v2
    vsrl.vi v0, v0, 8
    vse64.v v0, (a0)
    addi a0, a0, 16                 
    addi a2, a2, 16                 
    bnew a4, 1b
2:
    addi a5, a5, -1
    add a6, a6, a1
    add a7, a7, a3
    bnew a5, 0b 
    ret

 /*
 * void ass_mul_bitmaps(uint8_t *dst, ptrdiff_t dst_stride,
 *                      const uint8_t *src1, ptrdiff_t src1_stride,
 *                      const uint8_t *src2, ptrdiff_t src2_stride,
 *                      size_t width, size_t height);
 */

function mul_bitmaps_rvv
    mv t3, a0
    mv t4, a2
    mv t5, a4
0:
    vsetvli t0, a6, e64, m2, ta, ma  
    mv a0, t3
    mv a2, t4
    mv a4, t5
1:
    vle64.v v0, (a2)                 
    sub a6, a6, t0
    vle64.v v2, (a4)                 
    vmul.vv v4, v0, v2
    vadd.vx v4, v4, 255
    vsrl.vi v4, v4, 8
    vse64.v v4, (a0)
    addi a0, a0, 16                 
    addi a2, a2, 16                  
    addi a4, a4, 16 
    bnew a6, 1b
2:
    addi a7, a7, -1
    add t3, t3, a1
    add t4, t4, a3
    add t5, t5, a5
    bnew a7, 0b 
    ret

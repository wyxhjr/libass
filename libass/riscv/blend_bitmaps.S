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

    .section 
    .data            # 数据段
edge_mask:                    # 定义边缘掩码标签
    .byte 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF  # 16 个 0xFF
    .byte 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00  # 16 个 0x00


function add_bitmaps_rvv

    sub a6, 0, a4 
    andi a6, a6, 15
    ld a7, edge_mask
    add a7, a7, a6
    vmv.s.x v0, a7
    add a6, a6, a4
    sub a6, a6, 16
    sub x1, x1, x6
    sub x3, x3, x6
0:
    addi a6, a4, -16 
    bge x0, a6, 2
1:
    vsetvli t0, a4, e64, m2, ta, ma  
    vle64.v v0, (a0)                  # 从 a0 加载 16 字节到 v1
    sub a4, a4, t0
    addi a5, a5, -1
    vle64.v v2, (a2)                  # 从 a2 加载 16 字节到 v2
    vminu.vv v0, v0, v2              # v1 = min(v1, v2)，无符号加法可以用 vminu 指令代替
    vse64.v v0, (a0)                  # 将 v1 存储回地址 x0
    addi a0, a0, 16                  # 更新 x0
    addi a2, a2, 16                  # 更新 x2
    bnew a1, 1b
2:
    bnew a5, 1b 
    ret


    
       
/*
 * void ass_imul_bitmaps(uint8_t *dst, ptrdiff_t dst_stride,
 *                       const uint8_t *src, ptrdiff_t src_stride,
 *                       size_t width, size_t height);
 */

function mul_bitmaps_rvv
    sub a6, 0, a4 
    andi a6, a6, 15
    ld a7, edge_mask
    add a7, a7, a6
    vmv.s.x v0, a7
    add a6, a6, a4
    sub a6, a6, 16
    sub x1, x1, x6
    sub x3, x3, x6
0:
    addi a6, a4, -16 
    bge a0, a6, 2
1:
    vsetvli t0, a4, e64, m2, ta, ma  
    vle64.v v0, (a0)                  # 从 a0 加载 16 字节到 v1
    sub a4, a4, t0
    addi a5, a5, -1
    vle64.v v2, (a2)                  # 从 a2 加载 16 字节到 v2
    vrsub.vi v2, v2, 255
    vmulvv v0, v0, v2
    vsrl.vi v0, v0, 8
    addi a0, a0, 16                  # 更新 x0
    addi a2, a2, 16                  # 更新 x2
    bnew a1, 1b
2:
    bnew a5, 1b 
    ret





 /*
 * void ass_mul_bitmaps(uint8_t *dst, ptrdiff_t dst_stride,
 *                      const uint8_t *src1, ptrdiff_t src1_stride,
 *                      const uint8_t *src2, ptrdiff_t src2_stride,
 *                      size_t width, size_t height);
 */

function mul_bitmaps_rvv
    sub a6, 0, a4 
    andi a6, a6, 15
    ld a7, edge_mask
    add a7, a7, a6
    vmv.s.x v0, a7
    add a6, a6, a4
    sub a6, a6, 16
    sub x1, x1, x6
    sub x3, x3, x6
0:
    addi a6, a4, -16 
    bge a0, a6, 2
1:
    vsetvli t0, a6, e64, m2, ta, ma  
    vle64.v v0, (a2)                  # 从 a0 加载 16 字节到 v1
    sub a6, a6, t0
    addi a7, a7, -1
    vle64.v v2, (a4)                  # 从 a2 加载 16 字节到 v2
    vmul.vv v4, v0, v2
    vadd.vx v4, v4, 255
    vsrl.vi, v4, v4, 8
    addi a0, a0, 16                  # 更新 x0
    addi a2, a2, 16                  # 更新 x2
    addi a4, a4, 16 
    bnew a1, 1b
2:
    bnew a7, 1b
    ret

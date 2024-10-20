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
 * fill_line
 * Fill size bytes (16 or 32) starting from dst with val
 */

.macro fill_line dst, val, size
li t0, val
.if \size == 16
    sw t0, 0(\dst)
    sw t0, 8(\dst)
.elseif \size == 32
    sw t0, 0(\dst)
    sw t0, 4(\dst)
    sw t0, 8(\dst)
    sw t0, 12(\dst)
.else
.error "invalid line size"
.endif
.endm

/*
 * void fill_solid_tile(uint8_t *buf, ptrdiff_t stride, int set);
 */

.macro fill_solid_tile size
function fill_solid_tile\size\()_rvv
    # a0 = buf, a1 = stride, a2 = set (填充值)
    # 检查填充值是否为 0
    beq a2, zero, set_zero      # 如果 set == 0，跳转到 set_zero
    li t0, 1                     # 否则，设定 t0 = 1
    j fill_tile                  # 跳转到填充逻辑

set_zero:
    li t0, 0                     # 否则，设定 t0 = 0

fill_tile:
    li t1, \tile_size            # t1 = tile_size
    addi t1, t1, -1              # t1 = tile_size - 1

.fill_loop:
    # 使用 fill_line 函数填充一行
    call fill_line               # 调用填充函数
    add a0, a0, a1              # 更新缓冲区指针 (buf + stride)
    bnez t1, fill_loop          # 如果 t1 不为零，继续循环
    call fill_line               # 填充最后一行
    ret                          # 返回

.endm

# 使用宏进行调用
fill_solid_tile 16
fill_solid_tile 32

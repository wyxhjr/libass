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

.irq    size, 16, 32
.macro fill_solid_tile size
function fill_solid_tile\size\()_rvv
    beq a2, zero, set_zero      
    li t0, 1                     
    j fill_tile                  
set_zero:
    li t0, 0                     
fill_tile:
    li t1, \tile_size            
    addi t1, t1, -1              
.fill_loop:   
    call fill_line               
    add a0, a0, a1              
    bnez t1, fill_loop         
    call fill_line             
    ret                         
.endm


/*
 * void merge_tile(uint8_t *buf, ptrdiff_t stride, const uint8_t *tile);
 */

.irq    w, 16, 32
function merge_tile\w\()_rvv
    li      a3, w
    li      a4, w
0:
    addi    a4, a4, -1
1:
    vsetvli t0, a3, e64, m2, ta, ma
    vle64.v v0, (a0)
    sub     a3, a3, t0
    vle64.v v0, (a2)
    vmslt   v0, v2, v0
    add     a0, a0, a1
    addi    a2, a2, \w
    bnew a1, 1b
2:
    bnew a4, 1b
    ret 


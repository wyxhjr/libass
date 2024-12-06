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
 * load_line
 * Load vN register with correct source bitmap data
 */

.macro load_line dst, base, offs, max, zero_offs, tmp
    bge \offs, \max, 1f           
    add \tmp, \base, \offs        
    j 2f                         
1:  
    add \tmp, \base, \zero_offs   
2:  
    vle16.v \dst, (\tmp)         
.endm


    .section .rodata       
    .align 4               
words_zero:
    .word 0, 0, 0, 0       
    .word 0, 0, 0, 0

/*
 * void shrink_horz(int16_t *dst, const int16_t *src,
 *                  size_t src_width, size_t src_height);
 */

function shrink_horz16_rvv
    slli a4, a2, 1
    add a4, a4, 15
    andi a4, a4, -16
    mul a4, a4, a3
    addi a2, a2, 3
    la a5, words_zero
    sub a5, a5, a1
    li a6, 0
0:
    mv a7, a3
    vsetvli t0, a2, e64, m2, ta, ma
1:
    slli t4, a3, 4
    sub t5, a6, t4
    load_line v1, a1, t5, a4, a5, t6
    load_line v2, a1, a6, a4, a5, t6
    add t5, a6, t4
    load_line v3, a1, t5, a4, a5, t6
    vmv.v.v v0, v1              
    vslide1down.vx v0, v1, 0 
    vslide1down.vx v1, v1, 1 
    vmv.v.v v4, v2          
    vslide1down.vv v4, v3, v2
    vslide1down.vv v5, v2, v3
    vslideup.vx v2, v0, 14   
    vslideup.vx v3, v1, 14   
    vslideup.vx v0, v4, 12   
    vslideup.vx v1, v5, 12  
    vadd.vv v0, v0, v5
    vadd.vv v1, v1, v4
    vadd.vv v2, v2, v3
    vadd.vv v0, v0, v1
    vsrl.vi v0, v0, 1
    vadd.vv v0, v0, v2
    vsrl.vi v0, v0, 1
    vadd.vv v0, v0, v1
    vsrl.vi v0, v0, 1
    vadd.vv v0, v0, v2
    vsrl.vi v0, v0, 1
    vse16.v v0, (a0)             
    add a0, a0, 16 
    sub a7, a7, 1
    add t4, t4, 16
    bnez a7, 1b
    sub a2, a2, 16
    add a6, a6, t4
    bnez a2, 0b
    ret        


/*
 * void expand_vert(int16_t *dst, const int16_t *src,
 *                  size_t src_width, size_t src_height);
 */


function expand_vert16_rvv
    slli a3, a3, 4                
    la t4, words_zero             
    sub t4, t4, a1                
1:     
    add t5, a2, 32           
    vsetvli t0, a2, e64, m2, ta, ma      
    vmv.v.i v0, 0                 
    vmv.v.i v1, 0                
    li t6, 0                     
2:
    load_line v2, a1, t6, a3, t4, t7
    addi t6, t6, 16                
    vadd.vv v3, v0, v2                  
    vsrl.vi v3, v3, 1             
    vadd.vv v3, v3, v1            
    vsrl.vi v3, v3, 1             
    vadd.vv v0, v0, v3            
    vsrl.vi v0, v0, 1             
    vadd.vv v3, v2, v3            
    vsrl.vi v3, v3, 1             
    vadd.vv v0, v0, v1
    vaddi.vi v0, v0, 1
    vsrl.vi v0, v0, 1
    vadd.vv v3, v3, v1
    vaddi.vi v3, v3, 1
    vsrl.vi v3, v3, 1
    vse16.v v0, (a0)             
    add a0, a0, 16               
    vse16.v v3, (a0)             
    add a0, a0, 16               
    sub a2, a2, t0               
    vmv.v.v v0, v1               
    vmv.v.v v1, v2             
    bnez a2, 2b          
    sub a2, a2, 2                
    add a1, a1, a3               
    sub t4, t4, a3               
    bnez a3, 1b          
    ret
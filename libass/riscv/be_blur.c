/*
 * Copyright (C) 2009-2022 libass contributors
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

#include <riscv_vector.h>
#include "config.h"
#include "ass_compat.h"
#include "ass_utils.h"

#include <stddef.h>
#include <stdint.h>

#define ALIGNMENT 16

static inline uint16_t sliding_sum(uint16_t *prev, uint16_t next)
{
    uint16_t sum = *prev + next;
    *prev = next;
    return sum;
}

void ass_be_blur_c(uint8_t *restrict buf, ptrdiff_t stride,
                   size_t width, size_t height, uint16_t *restrict tmp)
{
    ASSUME(!((uintptr_t) buf % ALIGNMENT) && !(stride % ALIGNMENT));
    ASSUME(!((uintptr_t) tmp % ALIGNMENT));
    ASSUME(width > 1 && height > 1);

    uint16_t *col_pix_buf = tmp;
    uint16_t *col_sum_buf = tmp + stride;

    size_t vl;
    uint16_t *col_pix_ptr = col_pix_buf;
    uint16_t *col_sum_ptr = col_sum_buf;
    uint8_t *buf_ptr = buf;

   
    for (size_t y = 0; y < height; y++) {
        size_t avl = width;
        while (avl > 0) {
            vl = __riscv_vsetvl_e16m1(avl);
            vuint16m1_t prev_col_pix = __riscv_vle16_v_u16m1(col_pix_ptr, vl);
            vuint16m1_t prev_col_sum = __riscv_vle16_v_u16m1(col_sum_ptr, vl);
            vuint8m1_t current_buf = __riscv_vle8_v_u8m1(buf_ptr, vl);
            vuint16m1_t current_buf_u16 = __riscv_vwcvtu_x_x_v_u16m1(current_buf, vl);

            vuint16m1_t col_pix = __riscv_vadd_vv_u16m1(prev_col_pix, current_buf_u16, vl);
            vuint16m1_t col_sum = __riscv_vadd_vv_u16m1(prev_col_sum, col_pix, vl);

            __riscv_vse16_v_u16m1(col_pix_ptr, col_pix, vl);
            __riscv_vse16_v_u16m1(col_sum_ptr, col_sum, vl);

            col_pix_ptr += vl;
            col_sum_ptr += vl;
            buf_ptr += vl;
            avl -= vl;
        }
    }


    for (size_t y = 1; y < height; y++) {
        uint8_t *dst = buf;
        buf += stride;

        size_t avl = width;
        while (avl > 0) {
            vl = __riscv_vsetvl_e16m1(avl);
            vuint16m1_t prev_col_pix = __riscv_vle16_v_u16m1(col_pix_buf, vl);
            vuint16m1_t prev_col_sum = __riscv_vle16_v_u16m1(col_sum_buf, vl);
            vuint8m1_t current_buf = __riscv_vle8_v_u8m1(buf, vl);
            vuint16m1_t current_buf_u16 = __riscv_vwcvtu_x_x_v_u16m1(current_buf, vl);

            vuint16m1_t col_pix = __riscv_vadd_vv_u16m1(prev_col_pix, current_buf_u16, vl);
            vuint16m1_t col_sum = __riscv_vadd_vv_u16m1(prev_col_sum, col_pix, vl);

            vuint8m1_t result = __riscv_vnsrl_wx_u8m1(col_sum, 4, vl);
            __riscv_vse8_v_u8m1(dst, result, vl);

            col_pix_buf += vl;
            col_sum_buf += vl;
            buf += vl;
            dst += vl;
            avl -= vl;
        }
    }

    
    size_t avl = width;
    while (avl > 0) {
        vl = __riscv_vsetvl_e16m1(avl);
        vuint16m1_t col_pix = __riscv_vle16_v_u16m1(col_pix_buf, vl);
        vuint16m1_t col_sum = __riscv_vle16_v_u16m1(col_sum_buf, vl);
        vuint16m1_t final_sum = __riscv_vadd_vv_u16m1(col_pix, col_sum, vl);
        vuint8m1_t result = __riscv_vnsrl_wx_u8m1(final_sum, 4, vl);
        __riscv_vse8_v_u8m1(buf, result, vl);

        col_pix_buf += vl;
        col_sum_buf += vl;
        buf += vl;
        avl -= vl;
    }
}
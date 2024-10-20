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

#ifndef LIBASS_RISCV_ASM_S
#define LIBASS_RISCV_ASM_S

#include "config.h"

#if !defined(PIC)
#if defined(__PIC__)
#define PIC __PIC__
#elif defined(__pic__)
#define PIC __pic__
#endif
#endif

#ifndef PRIVATE_PREFIX
#define PRIVATE_PREFIX ass_
#endif

#define PASTE(a,b) a ## b
#define CONCAT(a,b) PASTE(a,b)

#ifdef PREFIX
#define EXTERN CONCAT(_,PRIVATE_PREFIX)
#else
#define EXTERN PRIVATE_PREFIX
#endif

.macro function name
    .text
    .global \name
    .type   \name, %function
    .option norvc
    .align  2
    \name:
.endm

        
        .macro const sym, align=3, relocate=0
            .if \relocate
                .pushsection .data.rel.ro
            .else
                .pushsection .rodata
            .endif
            .align \align
            \sym:

            .macro endconst
                .size  \sym, . - \sym
                .popsection
                .purgem endconst
            .endm
        .endm


#endif /* LIBASS_AARCH64_ASM_S */
//! Tests for assembly extraction (line-based, see parser/asm.rs).

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const KERNEL_STYLE: &str = r#"/* SPDX-License-Identifier: GPL-2.0 */
#include <linux/linkage.h>
#include <asm/export.h>

	.text
/*
 * memset - fill memory with a byte
 * rdi destination, rsi value, rdx count
 */
SYM_FUNC_START(__memset)
	movq %rdi, %r9
	movzbl %sil, %eax
.Lloop:
	movb %al, (%rdi)
	incq %rdi
	decq %rdx
	jnz .Lloop
	call memset_tail
	movq %r9, %rax
	RET
SYM_FUNC_END(__memset)
EXPORT_SYMBOL(__memset)

	.globl	clear_page
	.type	clear_page, @function
# Zero a 4K page.
clear_page:
	movl $512, %ecx
	xorl %eax, %eax
	rep stosq
	ret
	.size	clear_page, .-clear_page

	.section .rodata
fill_pattern:	.quad 0x0101010101010101
"#;

#[test]
fn test_kernel_function_embedding_text() {
    let units =
        assert_extractor_invariants(KERNEL_STYLE, Language::Assembly, "arch/x86/lib/memset.S");
    let memset = get_unit_by_name(&units, "__memset").unwrap();
    let expected = r#"Function: __memset
Signature: SYM_FUNC_START(__memset)
Description: memset - fill memory with a byte rdi destination, rsi value, rdx count
Calls: memset_tail
File: arch x86 lib memset memset.S
Code:
/*
 * memset - fill memory with a byte
 * rdi destination, rsi value, rdx count
 */
SYM_FUNC_START(__memset)
	movq %rdi, %r9
	movzbl %sil, %eax
.Lloop:
	movb %al, (%rdi)
	incq %rdi
	decq %rdx
	jnz .Lloop
	call memset_tail
	movq %r9, %rax
	RET
SYM_FUNC_END(__memset)"#;
    assert_eq!(build_embedding_text(memset), expected);
}

/// A `.globl` label is a function from its declarations to `.size`; local
/// `.L` labels and data labels are not functions.
#[test]
fn test_global_label_function() {
    let units = parse(KERNEL_STYLE, Language::Assembly, "memset.S");
    let clear = get_unit_by_name(&units, "clear_page").unwrap();
    assert_eq!(clear.unit_type, UnitType::Function);
    assert_eq!((clear.line, clear.end_line), (24, 32));
    assert_eq!(clear.docstring.as_deref(), Some("Zero a 4K page."));
    assert!(get_unit_by_name(&units, ".Lloop").is_none());
    assert!(get_unit_by_name(&units, "fill_pattern").is_none());
    let functions = units
        .iter()
        .filter(|u| u.unit_type == UnitType::Function)
        .count();
    assert_eq!(functions, 2);
}

/// x86inc.asm (FFmpeg, x264, dav1d): `cglobal` functions, named arguments,
/// `INIT_XMM` attached to the function below it, and macros as containers
/// of the functions they generate.
#[test]
fn test_nasm_x86inc() {
    let source = r#"%include "x86util.asm"

SECTION_RODATA
pw_1: times 8 dw 1

SECTION .text

; Average two blocks of pixels.
INIT_XMM sse2
cglobal pixel_avg_w8, 4,5,2, dst, dst_stride, src, src_stride
    movu    m0, [srcq]
    pavgb   m0, [dstq]
    call    ff_helper
    RET

%macro AVG_FUNC 1
cglobal pixel_avg_w%1, 4,5
    movu    m0, [r0]
    RET
%endmacro

AVG_FUNC 16
"#;
    let units = assert_extractor_invariants(source, Language::Assembly, "common/x86/mc-a.asm");
    let avg = get_unit_by_name(&units, "pixel_avg_w8").unwrap();
    assert_eq!((avg.line, avg.end_line), (8, 14));
    assert_eq!(
        avg.docstring.as_deref(),
        Some("Average two blocks of pixels.")
    );
    assert_eq!(
        avg.parameters,
        vec!["dst", "dst_stride", "src", "src_stride"]
    );
    assert_eq!(avg.calls, vec!["ff_helper"]);

    let mac = get_unit_by_name(&units, "AVG_FUNC").unwrap();
    assert_eq!((mac.line, mac.end_line), (16, 20));
    let generated = get_unit_by_name(&units, "pixel_avg_w%1").unwrap();
    assert_eq!((generated.line, generated.end_line), (17, 19));
}

/// FFmpeg / dav1d ARM: `function name, export=1` ... `endfunc`, `bl` calls,
/// `//` comments, and `.macro` ... `.endm`.
#[test]
fn test_arm_function_macros() {
    let source = r#"#include "libavutil/aarch64/asm.S"

.macro  transpose_4x4 r0, r1
        trn1            \r0\().4h, \r1\().4h, \r1\().4h
.endm

// void ff_add_pixels(int16_t *block, const uint8_t *pixels)
function ff_add_pixels_neon, export=1
        ld1             {v0.8b}, [x1]
        transpose_4x4   v0, v1
        b.eq            1f
        bl              X(ff_clear_block)
1:      ret
endfunc
"#;
    let units = assert_extractor_invariants(source, Language::Assembly, "aarch64/pixels.S");
    let mac = get_unit_by_name(&units, "transpose_4x4").unwrap();
    assert_eq!((mac.line, mac.end_line), (3, 5));
    let add = get_unit_by_name(&units, "ff_add_pixels_neon").unwrap();
    assert_eq!((add.line, add.end_line), (7, 14));
    assert_eq!(
        add.docstring.as_deref(),
        Some("void ff_add_pixels(int16_t *block, const uint8_t *pixels)")
    );
    assert_eq!(add.calls, vec!["ff_clear_block"]);
}

/// A file declaring nothing global (a bare-metal program, an included
/// fragment) treats every non-local label as a function.
#[test]
fn test_plain_labels_without_globals() {
    let source = r#"; boot sector
start:
    mov si, msg
    call print
    hlt

print:
    lodsb
    ret

msg: db "hi", 0
"#;
    let units = assert_extractor_invariants(source, Language::Assembly, "boot.asm");
    let start = get_unit_by_name(&units, "start").unwrap();
    assert_eq!((start.line, start.end_line), (1, 5));
    assert_eq!(start.calls, vec!["print"]);
    let print = get_unit_by_name(&units, "print").unwrap();
    assert_eq!((print.line, print.end_line), (7, 9));
    assert!(get_unit_by_name(&units, "msg").is_none());
}

/// Multi-line `#define` macros (kernel crypto rounds) are units of their own.
#[test]
fn test_cpp_macro_definitions() {
    let source = r#"#include <linux/linkage.h>

/* load 8 blocks */
#define load_8way(src, x0, x1) \
	vmovdqu (0*16)(src), x0; \
	vmovdqu (1*16)(src), x1;

SYM_FUNC_START(encrypt_8way)
	load_8way(%rdx, %xmm0, %xmm1)
	RET
SYM_FUNC_END(encrypt_8way)
"#;
    let units = assert_extractor_invariants(source, Language::Assembly, "glue.S");
    let load = get_unit_by_name(&units, "load_8way").unwrap();
    assert_eq!((load.line, load.end_line), (3, 6));
    assert_eq!(load.docstring.as_deref(), Some("load 8 blocks"));
    assert!(get_unit_by_name(&units, "encrypt_8way").is_some());
}

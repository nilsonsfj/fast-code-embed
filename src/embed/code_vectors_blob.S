/* nomic-embed-code vector blob embedded via assembler.
 * Cross-platform: macOS (Mach-O) vs Linux (ELF) vs Windows (COFF). */

#if defined(__APPLE__)
    .section __DATA,__const
    .globl _FCE_PRETRAINED_VECTOR_BLOB
    .globl _FCE_PRETRAINED_VECTOR_BLOB_LEN
    .p2align 4
_FCE_PRETRAINED_VECTOR_BLOB:
    .incbin "src/embed/code_vectors.bin"
_FCE_PRETRAINED_VECTOR_BLOB_END:

    .section __DATA,__const
    .p2align 2
_FCE_PRETRAINED_VECTOR_BLOB_LEN:
    .long _FCE_PRETRAINED_VECTOR_BLOB_END - _FCE_PRETRAINED_VECTOR_BLOB

#elif defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
    .section .rdata,"dr"
    .globl FCE_PRETRAINED_VECTOR_BLOB
    .globl FCE_PRETRAINED_VECTOR_BLOB_LEN
    .p2align 4
FCE_PRETRAINED_VECTOR_BLOB:
    .incbin "src/embed/code_vectors.bin"
FCE_PRETRAINED_VECTOR_BLOB_END:

    .section .rdata,"dr"
    .p2align 2
FCE_PRETRAINED_VECTOR_BLOB_LEN:
    .long FCE_PRETRAINED_VECTOR_BLOB_END - FCE_PRETRAINED_VECTOR_BLOB

#else
    .section .rodata,"a",@progbits
    .globl FCE_PRETRAINED_VECTOR_BLOB
    .globl FCE_PRETRAINED_VECTOR_BLOB_LEN
    .p2align 4
FCE_PRETRAINED_VECTOR_BLOB:
    .incbin "src/embed/code_vectors.bin"
FCE_PRETRAINED_VECTOR_BLOB_END:

    .section .rodata,"a",@progbits
    .p2align 2
FCE_PRETRAINED_VECTOR_BLOB_LEN:
    .long FCE_PRETRAINED_VECTOR_BLOB_END - FCE_PRETRAINED_VECTOR_BLOB

    /* Mark the stack non-executable. Without this marker the linker stamps
     * PT_GNU_STACK RWE on the shared object, silently disabling NX for the
     * whole JVM process and causing dlopen failures on glibc >= 2.41. */
    .section .note.GNU-stack,"",@progbits
#endif

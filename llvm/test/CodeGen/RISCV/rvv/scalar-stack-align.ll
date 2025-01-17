; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=riscv32 -mattr=+zve64x -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=RV32
; RUN: llc -mtriple=riscv64 -mattr=+zve64x -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=RV64
; RUN: llc -mtriple=riscv32 -mattr=+v -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=RV32
; RUN: llc -mtriple=riscv64 -mattr=+v -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=RV64

; FIXME: The stack is assumed and required to be aligned to 16 bytes, but we
; only ensure an 8-byte alignment for the size of the section containing RVV
; objects. After establishing sp, on zve64x the stack is only 8-byte aligned.
; This is wrong in and of itself, but we can see that this also has the effect
; that the 16-byte-aligned object at the bottom of the stack is misaligned.

define i64* @scalar_stack_align16() nounwind {
; RV32-LABEL: scalar_stack_align16:
; RV32:       # %bb.0:
; RV32-NEXT:    addi sp, sp, -32
; RV32-NEXT:    sw ra, 28(sp) # 4-byte Folded Spill
; RV32-NEXT:    csrr a0, vlenb
; RV32-NEXT:    sub sp, sp, a0
; RV32-NEXT:    addi a0, sp, 16
; RV32-NEXT:    call extern@plt
; RV32-NEXT:    mv a0, sp
; RV32-NEXT:    csrr a1, vlenb
; RV32-NEXT:    add sp, sp, a1
; RV32-NEXT:    lw ra, 28(sp) # 4-byte Folded Reload
; RV32-NEXT:    addi sp, sp, 32
; RV32-NEXT:    ret
;
; RV64-LABEL: scalar_stack_align16:
; RV64:       # %bb.0:
; RV64-NEXT:    addi sp, sp, -16
; RV64-NEXT:    sd ra, 8(sp) # 8-byte Folded Spill
; RV64-NEXT:    csrr a0, vlenb
; RV64-NEXT:    sub sp, sp, a0
; RV64-NEXT:    addi a0, sp, 8
; RV64-NEXT:    call extern@plt
; RV64-NEXT:    mv a0, sp
; RV64-NEXT:    csrr a1, vlenb
; RV64-NEXT:    add sp, sp, a1
; RV64-NEXT:    ld ra, 8(sp) # 8-byte Folded Reload
; RV64-NEXT:    addi sp, sp, 16
; RV64-NEXT:    ret
  %a = alloca <vscale x 2 x i32>
  %c = alloca i64, align 16
  call void @extern(<vscale x 2 x i32>* %a)
  ret i64* %c
}

declare void @extern(<vscale x 2 x i32>*)

from llvmlite import ir
from numba import cuda, types
from numba.core import cgutils
from numba.core.extending import intrinsic, overload
from numba.core.typing import signature


code = """\
; ModuleID = 'LFortran'
source_filename = "LFortran"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define i32 @__module___main___add(i32* %x, i32* %y) {
.entry:
  %_lpython_return_variable = alloca i32, align 4
  %res = alloca i32, align 4
  %0 = load i32, i32* %x, align 4
  %1 = add i32 %0, 3
  %2 = load i32, i32* %y, align 4
  %3 = mul i32 %1, %2
  store i32 %3, i32* %res, align 4
  %4 = load i32, i32* %res, align 4
  store i32 %4, i32* %_lpython_return_variable, align 4
  br label %return

unreachable_after_return:                         ; No predecessors!
  br label %return

return:                                           ; preds = %unreachable_after_return, %.entry
  %5 = load i32, i32* %_lpython_return_variable, align 4
  ret i32 %5
}

!nvvmir.version = !{!0}

!0 = !{i32 2, i32 0, i32 3, i32 1}
"""


def lfortran_add(x, y):
    pass


@intrinsic
def lfortran_add_intrinsic(typingctx, x, y):
    sig = signature(types.int32, types.int32, types.int32)

    def codegen(context, builder, sig, args):
        i32 = context.get_value_type(types.int32)
        stack_x = builder.alloca(i32)
        stack_y = builder.alloca(i32)

        x, y = args
        builder.store(x, stack_x)
        builder.store(y, stack_y)

        i32_ptr = ir.PointerType(i32)
        fnty = ir.FunctionType(i32, (i32_ptr, i32_ptr))
        fname = '__module___main___add'
        mod = builder.module
        fn = cgutils.get_or_insert_function(mod, fnty, fname)

        return builder.call(fn, (stack_x, stack_y))

    return sig, codegen


@overload(lfortran_add, target='cuda')
def ol_lfortran_add(x, y):
    def impl(x, y):
        return lfortran_add_intrinsic(x, y)
    return impl


@cuda.jit(extra_llvm=[code])
def f(x, y):
    res = lfortran_add(x, y)
    print(res)


f[1, 1](3, 3)
cuda.synchronize()

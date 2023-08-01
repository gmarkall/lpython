from llvmlite import ir
from numba import cuda, types
from numba.core import cgutils
from numba.core.extending import intrinsic, overload
from numba.core.typing import signature

import subprocess

cmd = ['./src/bin/lpython', '--show-nvvm', 'examples/device_function.py']
cp = subprocess.run(cmd, capture_output=True)

code = cp.stdout.decode()
print(code)

# Dummy function for overload
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

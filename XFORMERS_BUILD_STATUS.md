# xformers Build Status on M1 Mac

## Summary
❌ **xformers cannot be compiled on M1 Mac due to fundamental architectural issues**
✅ **PyTorch 2.10+ provides native attention optimization - no xformers needed**

## Why xformers Won't Compile

### Issue 1: clang Does Not Support `-fopenmp`
- xformers uses OpenMP for parallelization
- macOS clang doesn't support `-fopenmp` (only libomp library provides it)
- Even with `gcc-15`, PyTorch was compiled with clang, causing ABI mismatch

### Issue 2: CUDA Headers Required Unconditionally
- xformers/csrc/pt_stable_utils.h includes `<cuda.h>` and `<cuda_runtime.h>` without guards
- M1 Mac doesn't have CUDA (Apple Silicon uses Metal)
- Removing CUDA includes breaks other parts of xformers code that depend on `cudaDeviceProp` and CUDA types
- Conditional compilation would require patching 10+ files

### Issue 3: Compiler Mismatch
- PyTorch 2.10 for M1 was compiled with clang
- Mixing gcc-15 with clang-built PyTorch causes ABI incompatibilities
- Cannot force clang because it doesn't support `-fopenmp`

## Solution: Use PyTorch Native Implementation

**PyTorch 2.10.0+ includes `torch.nn.functional.scaled_dot_product_attention()`**

This provides the same functionality as xformers:
- Hardware-optimized attention implementation
- Automatic backend selection (uses native Metal on M1 Mac)
- Zero-copy operations
- Dropout and attention masking support

### Verification
```python
import torch
print(torch.__version__)  # 2.10.0
print(hasattr(torch.nn.functional, 'scaled_dot_product_attention'))  # True
```

## For Your Code

If your code was using:
```python
from xformers.ops import memory_efficient_attention
```

Replace with:
```python
import torch.nn.functional as F
# PyTorch automatically optimizes this on M1 Mac
output = F.scaled_dot_product_attention(query, key, value, ...)
```

## Attempted Solutions (All Failed)

1. ❌ Using libomp - Still fails at CUDA header inclusion
2. ❌ Using gcc-15 - ABI mismatch with clang-built PyTorch
3. ❌ Patching CUDA headers - Creates cascading failures in cuda-dependent code
4. ❌ Using XFORMERS_SKIP_CUDA_BUILD=1 - Environment variable not recognized
5. ❌ Older xformers versions - Same fundamental issues

## Conclusion

**xformers is not needed on M1 Mac with PyTorch 2.10+**

The native `torch.nn.functional.scaled_dot_product_attention()` provides:
- Better M1 optimization via Metal backend
- No compilation hassles
- Drop-in replacement for xformers attention
- Actively maintained by PyTorch team

**Recommendation**: Remove xformers from your dependencies and use native PyTorch attention.

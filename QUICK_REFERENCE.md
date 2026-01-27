# Quick Commands Reference

## Environment Setup

```bash
# Activate venv
source .venv/bin/activate

# Verify environment
python --version              # Should be 3.11.14
python -c "import torch; print(torch.__version__)"  # Should be 2.10.0

# Check M1 optimization
python -c "import torch; print('Metal available:', torch.backends.mps.is_available())"
```

## Testing & Validation

```bash
# Validate Python syntax
python -m py_compile app.py

# Show system info
python app.py info

# Run tests
python app.py test

# Test dependencies are installed
python -c "import torch, transformers, librosa, audiocraft; print('✅ All core deps installed')"
```

## Using Native PyTorch Attention

```python
# Instead of xformers, use PyTorch's native attention:
import torch.nn.functional as F

# Equivalent to xformers.ops.memory_efficient_attention()
output = F.scaled_dot_product_attention(
    query,      # Shape: (batch, seq_len, dim)
    key,        # Shape: (batch, seq_len, dim)
    value,      # Shape: (batch, seq_len, dim)
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False
)
# Automatically optimized for M1 Mac using Metal!
```

## Dependency Management

```bash
# Install all dependencies
pip install -r requirements.txt

# Install specific package
pip install torch==2.1.2

# Check what's installed
pip list | grep torch

# Update requirements
pip freeze > requirements.txt

# Install development tools
pip install -r requirements-dev.txt  # When created
```

## Common Issues & Solutions

### Issue: "xformers not found"
```python
# Solution: Use native PyTorch instead
# import xformers  ❌
import torch.nn.functional as F  # ✅
output = F.scaled_dot_product_attention(q, k, v)
```

### Issue: "CUDA_VISIBLE_DEVICES error"
```bash
# xformers tries to access CUDA - just ignore
# If using transformers, it will automatically use Metal on M1
unset CUDA_VISIBLE_DEVICES
```

### Issue: "Module not found" errors
```bash
# Reinstall from requirements
pip install -r requirements.txt --force-reinstall
```

## Performance Tips

### Enable M1 GPU Acceleration
```python
import torch
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
# Metal Performance Shaders will be used automatically
```

### Memory-Efficient Inference
```python
import torch

# Use bf16 precision (M1 native support)
with torch.autocast(device_type="mps", dtype=torch.bfloat16):
    output = model(input)  # Faster on M1
```

### Parallel Processing
```python
# For CPU-bound tasks
import multiprocessing as mp
workers = mp.cpu_count()
```

## Development Workflow

```bash
# Make changes to code
# ...

# Validate syntax
python -m py_compile src/**/*.py

# Test imports work
python -c "from src.services.multi_stem_generator import MultiStemGenerator"

# Run specific command
python app.py generate --prompt "trap beat" --duration 30

# View logs
cat logs/*.log
```

## File Structure

```
rvc-artist/
├── app.py                 # Main CLI (fixed ✅)
├── requirements.txt       # Dependencies (audited ✅)
├── .python-version        # Python 3.11 (fixed ✅)
├── .venv/                 # Virtual environment (Python 3.11)
│   └── lib/python3.11/    # All 104 packages
├── src/
│   ├── services/
│   │   ├── multi_stem_generator.py
│   │   ├── style_analyzer.py
│   │   └── lyrics_generator.py
│   └── routes/
├── data/
│   ├── audio/
│   ├── features/
│   └── transcripts/
├── output/
│   └── generated/
└── docs/
    ├── AUDIT_COMPLETE.md
    ├── STATUS.md
    └── ...
```

## Useful Environment Variables

```bash
# Reduce verbosity
export TF_CPP_MIN_LOG_LEVEL=3

# M1 optimization (auto-enabled)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Flask (if using web server)
export FLASK_PORT=5030
export FLASK_DEBUG=True

# Job parallelism
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## Getting Help

```bash
# Show all CLI commands
python app.py --help

# Show help for specific command
python app.py generate --help
python app.py analyze --help

# Check documentation
# See: AUDIT_COMPLETE.md
# See: STATUS.md
# See: XFORMERS_BUILD_STATUS.md
```

## Performance Monitoring

```bash
# Monitor M1 GPU usage (in separate terminal)
# macOS Activity Monitor → GPU tab
# or use:
# watch -n 1 "ps aux | grep python"

# Check available memory
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB')"
```

---

**Last Updated**: January 27, 2026
**Tested On**: Apple Silicon M1, Python 3.11.14, PyTorch 2.10.0

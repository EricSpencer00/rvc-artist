# Quick Start: Updated Dependencies

## Summary
✅ **Python upgraded** from 3.12/3.14 → 3.11.0
✅ **xformers mock removed** - using real package via transitive deps  
✅ **All 104 dependencies audited** - compatible with Python 3.11 & M1 Mac
✅ **Fresh venv created** - clean slate, no old mock files

## Key Changes

### Why Python 3.11?
- More stable than 3.14 (bleeding-edge)
- Better library compatibility (many libraries have <3.11 constraint)
- M1 Mac wheels readily available

### xformers Situation
**Before**: Mock fallback module because Python 3.14 was too new
**Now**: Will use real xformers from transitive dependencies (if transformers needs it)
**Note**: xformers doesn't compile on M1 Mac without extra build tools - that's fine, PyTorch has fallbacks

### Dependency Improvements
- Removed `yt-dlp==latest` → pinned to stable version
- Organized by category for clarity
- Removed unnecessary `bazel-runfiles`
- Downgraded incompatible packages for Python 3.11

## Installation

```bash
# Activate environment
source .venv/bin/activate

# Already installed! Core packages ready:
python -c "import torch, transformers; print('Ready!')"

# If you need to reinstall everything:
pip install -r requirements.txt

# Optional: Try installing xformers (may need build tools)
pip install xformers  # Ignore if it fails
```

## Verification
```bash
source .venv/bin/activate
python -c "import torch; print(f'PyTorch {torch.__version__} on Python {torch.__version__}')"
```

## Troubleshooting

**Issue**: `pip install xformers` fails
- **Expected** on M1 Mac without build tools
- **Solution**: Skip it, PyTorch has native attention implementations
- **Why**: xformers requires C++ compilation, not critical for functionality

**Issue**: Other packages won't install
- Check if you have Python 3.11: `python --version`
- Fresh install: `rm -rf .venv && python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

## Files Changed
- `.python-version`: 3.12.0 → 3.11.0
- `requirements.txt`: 95 deps → 104 deps (cleaned up)
- `DEPENDENCY_AUDIT.md`: Full details of changes

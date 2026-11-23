import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Without an NVIDIA GPU, run Triton kernels in its interpreter (NumPy on the CPU).
# This must be set before triton is imported anywhere.
if shutil.which("nvidia-smi") is None:
    os.environ.setdefault("TRITON_INTERPRET", "1")

"""FlashAttention forward pass in Triton, with a NumPy reference that follows the same block loop."""
from .reference import attention_naive, attention_blocked

__all__ = ["attention_naive", "attention_blocked", "flash_attention"]


def flash_attention(*args, **kwargs):
    """Triton kernel (imported lazily so the NumPy parts work without torch or triton)."""
    from .triton_fwd import flash_attention as _fa
    return _fa(*args, **kwargs)

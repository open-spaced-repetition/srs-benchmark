try:
    from . import RWKV_CUDA  # type: ignore
except ImportError:
    RWKV_CUDA = None

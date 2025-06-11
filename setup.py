import os
import torch
import glob
from setuptools import find_packages, setup

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
)


def get_rwkv_extensions():
    use_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.9
        ],
        "nvcc": ["-O3"],
    }

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, "rwkv", "model", "csrc")
    sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))

    extensions_cuda_dir = os.path.join(extensions_dir, "cuda")
    cuda_sources = list(glob.glob(os.path.join(extensions_cuda_dir, "*.cu")))

    if use_cuda:
        sources += cuda_sources

    ext_modules = [
        extension(
            "rwkv.model.RWKV_CUDA",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=False,
        )
    ]

    return ext_modules


setup(
    name="srs-benchmark",
    packages=find_packages(),
    ext_modules=get_rwkv_extensions(),
    install_requires=[
        "torch",
        "tqdm",
        "lmdb",
        "tomli",
        "pandas",
        "pyarrow",
        "fastparquet",
        "wandb",
        "scikit-learn",
    ],
    cmdclass={"build_ext": BuildExtension},
    options={},
)

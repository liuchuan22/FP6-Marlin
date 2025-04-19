from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Compiler flags
CXX_FLAGS = ['-g', 
             '-O3', 
             '-fopenmp', 
             '-lgomp', 
             '-std=c++17', 
             '-DENABLE_BF16', 
             '-D_GLIBCXX_USE_CXX11_ABI=0'
]
NVCC_FLAGS = [  '-O3', 
                '-std=c++17', 
                '-U__CUDA_NO_HALF_OPERATORS__', 
                '-U__CUDA_NO_HALF_CONVERSIONS__', 
                '-lineinfo',
                '--use_fast_math',
                '--threads=8', 
                '-Xptxas=-v', 
                '-D_GLIBCXX_USE_CXX11_ABI=0', 
                '-gencode', 
                # on 4090 gpu
                'arch=compute_89,code=sm_89',
                '-Wno-deprecated-declarations'
]

setup(
    name='fp6_marlin',
    version='0.0.1',
    description='mixed precision matmul',
    install_requires=['numpy', 'torch'],
    packages = find_packages(),
    ext_modules=[
        CUDAExtension(
            name = 'fp6_marlin.cuda',
            sources = [
                'fp6_marlin/fp6_marlin_cuda.cpp', 
                'fp6_marlin/fp6_marlin_cuda_kernel.cu'],
            extra_compile_args = {
                'cxx': CXX_FLAGS, 
                'nvcc': NVCC_FLAGS
            },
        ),
        CUDAExtension(
            name = 'fp6_marlin.cpu',
            sources = [
                'fp6_marlin/pybind.cpp',
                'fp6_marlin/weight_prepacking.cpp', 
                'fp6_marlin/weight_quant.cpp'],
            extra_compile_args = {
                'cxx': CXX_FLAGS, 
                'nvcc': NVCC_FLAGS
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)

from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='aggupscaleop', ext_modules=[cpp_extension.CppExtension('aggupscaleop', ['aggupscaleop.cpp'])], include_dirs = [], cmdclass={'build_ext': cpp_extension.BuildExtension})
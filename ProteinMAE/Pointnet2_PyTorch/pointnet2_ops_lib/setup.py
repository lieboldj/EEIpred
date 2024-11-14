import glob
import os
import os.path as osp

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = osp.join("pointnet2_ops", "_ext-src")
_ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "src", "*.cu")
)
_ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

requirements = ["torch>=1.4"]

exec(open(osp.join("pointnet2_ops", "_version.py")).read())

os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0;6.0;6.1;6.2;7.0;7.5;8.6"
setup(
    name="pointnet2_ops",
    version=__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pointnet2_ops._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3", "-D_GLIBCXX_USE_CXX11_ABI=0"],
                "nvcc": [
                    "-O3",
                    "-Xfatbin", "-compress-all",
                    "-gencode=arch=compute_86,code=sm_86",  # Ensure we target the right architecture
                    "-D_GLIBCXX_USE_CXX11_ABI=0",
                ],
            },
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
    include_package_data=True,
)
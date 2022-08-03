import os
import setuptools
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True
import numpy as np

ext_modules = [
    setuptools.Extension(
        "tt_sketch.drm.fast_lazy_gaussian",
        [os.path.join("tt_sketch", "drm", "fast_lazy_gaussian.pyx")],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
        include_dirs=[np.get_include(),"tt_sketch/drm"],
    )
]

short_description = """
Streaming randomized TT-decompositions for structured tensors.
"""


long_description = short_description

setuptools.setup(
    name="tt_sketch",
    version="1.1",
    author="Rik Voorhaar",
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RikVoorhaar/tt-sketch",
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(ext_modules),
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={"": ["*.pyx"]},
    install_requires=[
        "numpy",
        "scipy",
    ],
    setup_requires=["pytest-runner", "cython","numpy","scipy"],
    tests_require=[
        "mypy",
        "pycodestyle",
        "pytest",
        "pytest-cov",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="mathematics machine-learning tensors linear-algebra",
)

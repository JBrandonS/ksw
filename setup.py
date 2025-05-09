from sys import version
from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_inc
from Cython.Build import cythonize
import numpy as np
import os
from pathlib import Path

opj = os.path.join

path = str(Path(__file__).parent.absolute())

compile_opts = {
    "extra_compile_args": [
        "-shared",
        "-std=c99",
        "-fopenmp",
        "-Wno-strict-aliasing",
    ],
    "extra_link_args": ["-Wl,-rpath," + opj(path, "lib")],
}

compiler_directives = {"language_level": 3}

ext_modules = [
    Extension(
        "ksw.radial_functional",
        [opj(path, "cython", "radial_functional.pyx")],
        libraries=["radial_functional", "gomp"],
        library_dirs=[opj(path, "lib")],
        include_dirs=[opj(path, "include"), np.get_include()],
        **compile_opts,
    ),
    Extension(
        "ksw.estimator_core",
        [opj(path, "cython", "estimator_core.pyx")],
        libraries=["ksw_estimator"],
        library_dirs=[opj(path, "lib")],
        include_dirs=[opj(path, "include"), np.get_include()],
        **compile_opts,
    ),
    Extension(
        "ksw.fisher_core",
        [opj(path, "cython", "fisher_core.pyx")],
        libraries=["ksw_fisher"],
        library_dirs=[opj(path, "lib")],
        include_dirs=[opj(path, "include"), np.get_include()],
        **compile_opts,
    ),
    Extension(
        "ksw.legendre",
        [
            opj(path, "cython", "legendre.pyx"),
            opj(path, "libpshtlight", "ylmgen_c.c"),
            opj(path, "libpshtlight", "c_utils.c"),
            opj(path, "libpshtlight", "walltime_c.c"),
        ],
        libraries=["gomp"],
        include_dirs=[
            opj(path, "include"),
            opj(path, "libpshtlight"),
            get_python_inc(),
            np.get_include(),
        ],
        **compile_opts,
    ),
]

setup(
    name="ksw",
    packages=["ksw"],
    version="0.0.1",
    ext_modules=cythonize(ext_modules, compiler_directives=compiler_directives),
)

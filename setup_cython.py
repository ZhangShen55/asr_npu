# setup_cython.py
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import glob, os

def list_py(patterns, excludes=("__init__.py",)):
    files = []
    for pat in patterns:
        files += glob.glob(pat, recursive=True)
    files = [f for f in files if f.endswith(".py") and os.path.basename(f) not in excludes]
    return files

def to_module_name(path: str) -> str:
    # "core/concurrency.py" -> "core.concurrency"
    return os.path.splitext(path)[0].replace(os.sep, ".")

modules = list_py([
    "core/**/*.py",
    "utils/**/*.py",
    "api/routes/**/*.py",
    "entity/**/*.py",
])

extensions = [
    Extension(
        name=to_module_name(m),
        sources=[m],
        extra_compile_args=["-std=gnu99", "-O3", "-fPIC"],
        extra_link_args=[],
    )
    for m in modules
]

ext_modules = cythonize(
    extensions,
    nthreads=4,
    compiler_directives={
        "language_level": "3",
        "boundscheck": False,
        "wraparound": False,
        "initializedcheck": False,
        "cdivision": True,
    },
)

setup(
    name="seacraftasr_core",
    ext_modules=ext_modules,
)

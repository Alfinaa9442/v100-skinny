"""Path resolution for the benchmark harnesses.

Each harness needs a CUDA source to JIT-build, somewhere to write result
rows, and (for the AIME cells) the fixture directory. Each resolves to the
copy in this repository first, falls back to the deployed copy on the
serving box, and takes an environment override.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_BOX = os.path.expanduser("~/flatness-run")


def kernel_src(name="skinny_kernels.cu"):
    """Absolute path to a CUDA source, production or research.

    SKINNY_KERNEL_DIR overrides the search root.
    """
    override = os.environ.get("SKINNY_KERNEL_DIR")
    roots = ([override] if override else []) + [
        os.path.join(_ROOT, "kernels"),
        os.path.join(_ROOT, "kernels", "research"),
        _BOX,
    ]
    for root in roots:
        cand = os.path.join(root, name)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"CUDA source {name!r} not found in: {', '.join(roots)}")


def out_csv(name):
    """Absolute path for a result CSV. SKINNY_OUT_DIR overrides."""
    out_dir = os.environ.get("SKINNY_OUT_DIR") or os.path.join(_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, name)


def fixtures_dir():
    """Directory holding the AIME fixtures. SKINNY_FIXTURES overrides."""
    override = os.environ.get("SKINNY_FIXTURES")
    if override:
        return override
    cand = os.path.join(_HERE, "ninfer_fixtures")
    return cand if os.path.isdir(cand) else os.path.join(_BOX, "ninfer_fixtures")

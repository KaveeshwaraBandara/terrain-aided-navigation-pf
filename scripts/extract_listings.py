"""Copy the code excerpts shown in the report out of the source, so they never drift.

Writes report/generated/code_<name>.py for each function listed below (docstrings removed
to keep the listings short; the full code is in the repository).
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from tan.filters import resampling
from tan.filters.ekf import ExtendedKalmanFilter
from tan.filters.particle_filter import ParticleFilter
from tan.filters.rbpf import RaoBlackwellizedPF
from tan.pcrb import pcrb
from tan.reporting import GENERATED

EXCERPTS = {
    "pf_predict": ParticleFilter.predict,
    "pf_update": ParticleFilter.update,
    "pf_step": ParticleFilter.step,
    "systematic": resampling.systematic,
    "ekf_update": ExtendedKalmanFilter.update,
    "rbpf_predict": RaoBlackwellizedPF.predict,
    "pcrb": pcrb,
}


def strip_docstring(src: str) -> str:
    tree = ast.parse(src)
    fn = tree.body[0]
    lines = src.splitlines()
    if (
        fn.body
        and isinstance(fn.body[0], ast.Expr)
        and isinstance(getattr(fn.body[0], "value", None), ast.Constant)
        and isinstance(fn.body[0].value.value, str)
    ):
        doc = fn.body[0]
        del lines[doc.lineno - 1 : doc.end_lineno]
    return "\n".join(lines) + "\n"


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    for name, obj in EXCERPTS.items():
        src = textwrap.dedent(inspect.getsource(obj))
        out = GENERATED / f"code_{name}.py"
        out.write_text(strip_docstring(src))
        print(out)


if __name__ == "__main__":
    main()

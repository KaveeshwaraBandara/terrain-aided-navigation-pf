"""Static checks of the LaTeX report (no TeX installation needed).

- every generated-number macro used in the text is defined in report/generated/*.tex
- every \\ref / \\cref / \\Cref target has a \\label
- every \\input file and every \\includegraphics / \\lstinputlisting file exists
- every \\cite key is in references.bib, and every bib entry is cited
- braces balance in every section file
Exit code 1 if anything is wrong.
"""

from __future__ import annotations

import re
import sys

from tan.config import REPO_ROOT

LATEX = REPO_ROOT / "report" / "latex"
GEN = REPO_ROOT / "report" / "generated"
GENERATED_PREFIXES = (
    "EOne",
    "ETwo",
    "EThree",
    "EFour",
    "EFive",
    "ESix",
    "ESeven",
    "Terr",
    "Rt",
    "Val",
    "NumTests",
    "LinesOfCode",
    "Machine",
)


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def main() -> int:
    errors: list[str] = []
    files = [LATEX / "main.tex", *sorted((LATEX / "sections").glob("*.tex"))]
    texts = {f: strip_comments(f.read_text()) for f in files}
    alltext = "\n".join(texts.values())

    defined = set()
    for g in GEN.glob("*.tex"):
        defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", g.read_text()))
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)", alltext))
    for m in sorted(used):
        if m.startswith(GENERATED_PREFIXES) and m not in defined:
            errors.append(f"undefined generated macro \\{m}")

    labels = set(re.findall(r"\\label\{([^}]+)\}", alltext))
    labels |= set(re.findall(r"label=\{([^}]+)\}", alltext))  # \lstinputlisting[label={...}]
    for group in re.findall(r"\\(?:c|C)?ref\{([^}]+)\}", alltext):
        for lab in group.split(","):
            if lab.strip() not in labels:
                errors.append(f"reference to missing label {lab.strip()}")

    for path in re.findall(r"\\input\{([^}]+)\}", alltext):
        p = LATEX / path
        if not (p.exists() or p.with_suffix(".tex").exists()):
            errors.append(f"\\input file missing: {path}")
    for path in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", alltext):
        if not (REPO_ROOT / "report" / "figures" / path).exists():
            errors.append(f"figure missing: {path}")
    for path in re.findall(r"\\lstinputlisting(?:\[[^\]]*\])?\{([^}]+)\}", alltext):
        if not (LATEX / path).exists():
            errors.append(f"listing missing: {path}")

    bib = (LATEX / "references.bib").read_text()
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = {
        k.strip() for g in re.findall(r"\\cite[pt]?\{([^}]+)\}", alltext) for k in g.split(",")
    }
    for k in sorted(cited - keys):
        errors.append(f"citation key not in references.bib: {k}")
    for k in sorted(keys - cited):
        errors.append(f"bib entry never cited (remove or cite): {k}")

    for f, t in texts.items():
        if t.count("{") != t.count("}"):
            errors.append(f"unbalanced braces in {f.name}: {t.count('{')} vs {t.count('}')}")
        for env in set(re.findall(r"\\begin\{(\w+\*?)\}", t)):
            if t.count(f"\\begin{{{env}}}") != t.count(f"\\end{{{env}}}"):
                errors.append(f"unbalanced environment {env} in {f.name}")

    for e in errors:
        print("ERROR:", e)
    print(
        f"{len(errors)} problem(s); {len(used & defined)} generated macros used, "
        f"{len(labels)} labels, {len(cited)} citations"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

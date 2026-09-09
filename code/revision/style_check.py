#!/usr/bin/env python3
"""Check the manuscript against the author style rules.

Rules
  no em dash, no en dash used as punctuation
  no colon and no semicolon in body prose
  no use of the words experiment or experimental
Math, labels, references, citations, URLs and the bibliography are exempt.

usage  python3 revision/style_check.py path/to/file.tex
"""
import re
import sys

EXEMPT_CMDS = r"\\(label|ref|eqref|cite|includegraphics|input|include|url|href|hypersetup|usepackage|documentclass|newtheorem|def|renewcommand|newcommand|bibitem|definecolor|setlength|resizebox|usetikzlibrary|markboth|doiinfo)\b"


def strip_math(line):
    line = re.sub(r"\$\$.*?\$\$", " MATH ", line, flags=re.S)
    line = re.sub(r"\$[^$]*\$", " MATH ", line)
    line = re.sub(r"\\\(.*?\\\)", " MATH ", line, flags=re.S)
    line = re.sub(r"\\\[.*?\\\]", " MATH ", line, flags=re.S)
    return line


def body_lines(path):
    """Yield (lineno, text) for prose lines only."""
    in_bib = False
    in_math_env = False
    math_envs = ("equation", "align", "aligned", "gather", "eqnarray", "array", "cases", "tikzpicture")
    for i, raw in enumerate(open(path, encoding="utf-8"), 1):
        s = raw.rstrip("\n")
        if "\\begin{thebibliography}" in s:
            in_bib = True
        if "\\end{thebibliography}" in s:
            in_bib = False
            continue
        if in_bib:
            continue
        if any(f"\\begin{{{e}" in s for e in math_envs):
            in_math_env = True
        if any(f"\\end{{{e}" in s for e in math_envs):
            in_math_env = False
            continue
        if in_math_env:
            continue
        if s.strip().startswith("%"):
            continue
        s = s.split("  %")[0]
        if re.search(EXEMPT_CMDS, s):
            s = re.sub(EXEMPT_CMDS + r"\s*(\[[^\]]*\])?\s*(\{[^{}]*\})?", " CMD ", s)
        yield i, strip_math(s)


def main(path):
    problems = []
    for i, s in body_lines(path):
        for pat, msg in (
            (r"---", "em dash"),
            (r"—", "em dash"),
            (r"–", "en dash"),
            (r":", "colon"),
            (r";", "semicolon"),
            (r"\bexperiment", "word experiment"),
            (r"\bExperiment", "word experiment"),
        ):
            for m in re.finditer(pat, s):
                ctx = s[max(0, m.start() - 45):m.start() + 45].strip()
                problems.append((i, msg, ctx))
    if not problems:
        print(f"clean  {path}")
        return 0
    counts = {}
    for _, msg, _ in problems:
        counts[msg] = counts.get(msg, 0) + 1
    print(f"{len(problems)} issues in {path}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {v:4d}  {k}")
    print()
    for i, msg, ctx in problems[:60]:
        print(f"  line {i:4d}  {msg:16s}  {ctx}")
    if len(problems) > 60:
        print(f"  ... {len(problems)-60} more")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "submission_ojcs/tj-template-s1.tex"))

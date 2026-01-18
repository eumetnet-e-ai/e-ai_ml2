#!/usr/bin/env python3
"""
Create empty slide files lec10_02.tex to lec10_06.tex
with a correct LaTeX frame skeleton and header.
"""

LECTURE = 10
START = 2
END = 6

TEMPLATE = """%!TEX root = lec10.tex
% ================================================================================
% Lecture {lecture} — Slide {slide:02d}
% ================================================================================
\\begin{{frame}}[t,fragile]
\\begin{{tightmath}}

\\mytitle{{}}

\\begin{{columns}}[T,totalwidth=\\textwidth]

% ------------------------------------------------------------
\\begin{{column}}[T]{{0.48\\textwidth}}

% TODO

\\end{{column}}

% ------------------------------------------------------------
\\begin{{column}}[T]{{0.48\\textwidth}}

% TODO

\\end{{column}}

\\end{{columns}}

\\end{{tightmath}}
\\end{{frame}}
"""

for slide in range(START, END + 1):
    fname = f"lec{LECTURE}_{slide:02d}.tex"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(TEMPLATE.format(lecture=LECTURE, slide=slide))
    print(f"Created {fname}")

print("✅ Empty slides created.")


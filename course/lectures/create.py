#!/usr/bin/env python3
"""
generate_lectures.py

Creates lecture folders lec01 ... lec20 with:
- main file lecXX.tex
- slide files lecXX_01.tex ... lecXX_25.tex

Existing files are NOT overwritten.
"""

from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------
LECTURE_START = 1
LECTURE_END   = 20
SLIDES_PER_LECTURE = 25

BASE_DIR = Path.cwd()   # run from course/lectures/

# --------------------------------------------------
# Templates
# --------------------------------------------------
MAIN_TEX_TEMPLATE = r"""
% ================================================================================
% E-AI Tutorial Slides
% Filename: lec{lec:02d}.tex
%
% Roland Potthast 2025/2026
% Licence: CC-BY4.0
% ================================================================================
\documentclass[aspectratio=169]{{beamer}}

% --- Load lecture macros --------------------------------------------------------
\input{{../lec_macros.tex}}
\newcommand{{\LectureNumber}}{{Lecture {lec}}}

% --- Document -------------------------------------------------------------------
\begin{{document}}

\input{{../lec_agenda.tex}}
{inputs}

% --- End Document ---------------------------------------------------------------
\end{{document}}
""".lstrip()

SLIDE_TEX_TEMPLATE = r"""
%!TEX root = lec{lec:02d}.tex
% ================================================================================
% Lecture {lec}, Slide {slide:02d}
% ================================================================================
\begin{{frame}}
  \frametitle{{Lecture {lec} — Slide {slide:02d}}}

  % Content goes here

\end{{frame}}
""".lstrip()

# --------------------------------------------------
# Generation logic
# --------------------------------------------------
for lec in range(LECTURE_START, LECTURE_END + 1):
    lec_dir = BASE_DIR / f"lec{lec:02d}"
    lec_dir.mkdir(exist_ok=True)

    # ---------- main lecture file ----------
    main_file = lec_dir / f"lec{lec:02d}.tex"

    if not main_file.exists():
        inputs = "\n".join(
            rf"\input{{lec{lec:02d}_{i:02d}.tex}}"
            for i in range(1, SLIDES_PER_LECTURE + 1)
        )

        main_file.write_text(
            MAIN_TEX_TEMPLATE.format(lec=lec, inputs=inputs),
            encoding="utf-8"
        )
        print(f"Created {main_file}")
    else:
        print(f"Exists  {main_file}")

    # ---------- slide files ----------
    for slide in range(1, SLIDES_PER_LECTURE + 1):
        slide_file = lec_dir / f"lec{lec:02d}_{slide:02d}.tex"

        if not slide_file.exists():
            slide_file.write_text(
                SLIDE_TEX_TEMPLATE.format(lec=lec, slide=slide),
                encoding="utf-8"
            )
            print(f"  Created {slide_file.name}")
        else:
            print(f"  Exists  {slide_file.name}")

print("\nDone.")


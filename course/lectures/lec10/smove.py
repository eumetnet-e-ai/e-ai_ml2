#!/usr/bin/env python3
"""
Shift Lecture 10 slide files forward by N positions.

Example:
  shift_by = 5
  lec10_27.tex -> lec10_32.tex
  lec10_02.tex -> lec10_07.tex
"""

import os
import re

LECTURE = 10
START = 2
END = 27          # last REAL slide before inserting new ones
SHIFT_BY = 5      # number of new slides to insert

def fname(n):
    return f"lec{LECTURE}_{n:02d}.tex"

HEADER_RE = re.compile(
    r"%\s*Lecture\s+10\s+—\s+Slide\s+\d+"
)

for old in range(END, START - 1, -1):
    new = old + SHIFT_BY
    old_file = fname(old)
    new_file = fname(new)

    if not os.path.exists(old_file):
        print(f"⚠️  Missing: {old_file}")
        continue

    os.rename(old_file, new_file)
    print(f"Renamed {old_file} -> {new_file}")

    with open(new_file, "r", encoding="utf-8") as f:
        content = f.read()

    new_header = f"% Lecture {LECTURE} — Slide {new:02d}"
    content, count = HEADER_RE.subn(new_header, content, count=1)

    if count == 0:
        print(f"⚠️  Header not updated in {new_file}")

    with open(new_file, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ Shift-by completed.")

# Incident Root-Cause Rating — Instructions

Thank you for helping evaluate automated incident-analysis outputs. This
takes about 45–60 minutes. Your ratings will be used in aggregate in an
academic paper; no personal information beyond the role/experience fields
below will be published.

## What you are rating

Each row of `rating_sheet.csv` is a real production incident (from public
postmortems). You see the documented **ground-truth root cause** and three
anonymous predictions (**A**, **B**, **C**) produced by different automated
methods. The labels are shuffled per row — do not try to guess which
method is which.

For EACH prediction enter one of:

  2 — correct: identifies the actual root cause
  1 — partially correct: right direction or component, but incomplete/vague
  0 — wrong: incorrect or unrelated

## Rules

- Rate each prediction independently. Ties are fine, including all-0 rows.
- Use only the ground-truth text shown — no web searches needed.
- Do not discuss ratings with other raters until everyone has submitted.
- Fill every rating cell (no blanks).

## Before you start, add one row at the top of your file or in your reply:

  Role: (e.g., SRE / backend engineer / SWE)
  Years of professional experience:
  Have you been on-call for production systems? (yes/no)

## Submitting

Save your completed file as `rating_sheet_<yourname>.csv` and send it back.

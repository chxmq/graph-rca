# Human evaluation instructions

You are rating root-cause predictions for real production incidents.

For each row in rating_sheet.csv you see the incident's documented ground
truth and three anonymous predictions (A, B, C). For EACH prediction enter:

  2 — correct: identifies the actual root cause
  1 — partially correct: right direction/component but incomplete or vague
  0 — wrong: incorrect or unrelated to the actual cause

Rules:
- Rate each prediction independently; ties are fine.
- Do not discuss ratings with other raters until everyone has finished.
- Do not try to guess which system produced which prediction.
- Save your filled sheet as rating_sheet_<yourname>.csv in this folder.

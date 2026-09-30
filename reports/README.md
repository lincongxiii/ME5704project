# English colour paper

`ME5704_English_Colour.docx` retains the A4 geometry, font sizes, 1.5-spaced body text, heading hierarchy, native numbered equations, three-line tables and page footer of `template_zh.docx`. All visible text is English and the three comparison figures are regenerated in colour. English reflow changes pagination naturally. The contribution statement starts on its own page.

## Rebuild

With the scientific dependencies in `validation/requirements.txt` installed:

```powershell
python reports/make_figures.py
```

The document builder requires `lxml` and reads the checked-in validation evidence:

```powershell
python reports/build_english.py
```

For document authoring in Codex, use the bundled document Python runtime (which already has lxml); no external translation service is used. The text is in `english_text.json`. `template_fidelity.json` records preserved package parts. Original source equations and styles are preserved; only translated text, selected validation evidence, image bytes, contribution-page pagination and document metadata change.

## Numerical sources

`integrated_data.json` is the comparative analysis snapshot from the preceding Chinese report. Its TPS extrema are numerical multistart results, not globally certified bounds. Colour field plots are computed afresh using actual quadratic and contact functions and the independent TPS implementation. The bar chart uses full-precision snapshot extrema and contact fit CSVs. Ring Monte Carlo values in the paper come from the stored full contact study. The new integrated validation run is recorded separately under `validation/evidence/`; output from future reruns goes to `validation/output/`.

After changing data or methods, update the analysis snapshot and paper text deliberately; regenerating figures alone is not a full scientific reanalysis. Fill in all full names and agreed contributions, then export the completed document to PDF before submission.

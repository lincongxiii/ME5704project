# ME5704project
## Contents

- [`physical_contact_models/`](physical_contact_models/) - Hertz and ring-shaped Gaussian contact models (see its README)
- [`c/`](c/) - quadratic pressure model and corrected exact-hull Monte Carlo analysis
- [`validation/`](validation/README.md) - one-command known-answer validation and local HTML report
- [`reports/ME5704_English_Colour.docx`](reports/ME5704_English_Colour.docx) - English paper with colour figures and the retained academic layout

## Run the validation system

On the existing Windows setup, double-click `run_validation.cmd`. For a new machine, follow [installation and usage](validation/README.md).

The integrated run passes 26 tests and 15 analytical benchmark rows. It also reruns 1000 quadratic Monte Carlo samples per noise level. The generated browser report is `validation/output/index.html`; exact/numerical values and errors are exported to CSV and JSON. The longer contact study is optional via `--full-contact`.

The assignment requires scripts with a simple accuracy test, all members' full names on the front page, a contribution page and a final PDF. The editable paper still has placeholders for names and percentages. Tests check numerical routines; they do not establish the actual physical pressure field from five readings.

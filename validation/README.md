# ME5704 numerical validation

The assignment asks for scripts and a simple test case demonstrating programme accuracy. It does **not** require a web application. This command-line harness generates a local browser report for demonstrations and exports numerical evidence for the paper.

## Windows quick start

In this repository, double-click `run_validation.cmd`. On the author's existing machine it detects `../_venv/Scripts/python.exe`. On another machine install Python 3.12 and run from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r validation/requirements.txt
.\.venv\Scripts\python.exe validation/run_validation.py --open
```

The normal run takes about a minute, depending on hardware. It runs the contact tests and additional validation tests, evaluates known-answer benchmarks, and executes the **actual corrected** `c/noise_robustness.py` Monte Carlo study with 1000 simulations per level. No server, account, or internet is needed after dependencies are installed.

## Results

- `validation/output/index.html`: local report, open in any browser.
- `validation/output/benchmarks.csv`: exact values, numerical values, maximum absolute error, tolerance, iteration count, measured execution time and status.
- `validation/output/results.json`: machine-readable results, package versions, source hashes and Monte Carlo counts.
- `validation/output/pytest.log` and `pytest.xml`: full automated test evidence.
- `validation/output/quadratic_mc.log`: original Monte Carlo script output.

The overall status is FAIL, and the process exits nonzero, if tests or benchmark tolerances fail. An execution exception also exits nonzero; do not interpret an older HTML report as the outcome of a failed new run. Inspect terminal output and timestamps. `validation/evidence/` contains the checked-in snapshot from the report preparation, while `output/` is regenerated locally and ignored by Git.

## Full contact-model regeneration

```powershell
.\.venv\Scripts\python.exe validation/run_validation.py --full-contact --open
```

This also runs `physical_contact_models/run_contact.py` (2000 fit starts and 1000 simulations per noise level); allow roughly 10–15 minutes or longer. It rewrites the contact study's results and figures. Progress is recorded in `validation/output/contact_full.log`. A normal run validates the contact routines with tests but **does not rerun this entire study**.

## Analytical tests and coverage

The central test is `p=40-2(x-1)^2-3(y-2)^2`. The exact maximum is 40 at (1,2), and on x=1 the two roots are `2±sqrt(40/3)`. Six full-rank synthetic observations test quadratic recovery at unseen locations; using five observations would not identify an arbitrary six-coefficient quadratic. Additional checks cover TPS affine reproduction, a boundary maximum, singular/constant quadratics, nonlinear parameter recovery, a known ring crest and the Hertz Jacobian.

The harness imports the actual fit, prediction and hull-extrema functions in `c/noise_robustness.py`, plus the actual root, optimisation and nonlinear-fitting functions under `physical_contact_models/numerics`. The Monte Carlo script now has a main guard so importing it does not run simulations or open figures; direct execution retains its original behaviour. `validation/tps.py` is an independent TPS implementation, labelled accordingly. Other standalone `c/` scripts are not all automatically validated by these tests.

The root scan now includes a root at its right endpoint. Tests also document that an off-grid double root or a close pair can be missed. These are limitations of sign-change scanning, not evidence of zero-free regions. The main quadratic model uses a global minimum calculation to establish zero-free status within the sensor hull. Passing all tests does not prove global optimality of arbitrary nonlinear fits, nor establish the real hand's pressure field or actual damage.

## Report mapping

- Part (a): synthetic quadratic recovery, TPS affine reproduction, nonlinear parameter recovery; original contact tests also check known Hertz/Ring fits.
- Part (b): analytical roots with bisection/Newton, invalid brackets, endpoint roots and documented scan limitations.
- Part (c): analytical interior/boundary maxima, exact polygon extrema and known ring crest.
- Discussion/D: error tolerances, iterations, runtime, noise sensitivity, incomplete coverage and source provenance.

The English paper is in `reports/ME5704_English_Colour.docx`. The reproducible report inputs, figure generator and translation builder are stored alongside it. Fill in all members' full names and contribution percentages before final submission; the assignment requires PDF submission.

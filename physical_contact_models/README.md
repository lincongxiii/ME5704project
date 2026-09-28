# Physical contact models for the robot-hand pressure data

Two physically motivated models for the pressure measured by five sensors on a robot hand:

- **Hertz contact model** (lead-in): a single elliptical contact,
  $p = p_0\sqrt{1-((x-x_0)/a)^2-((y-y_0)/b)^2}$.
  Shown to be incompatible with the data: no concave surface can be as low as 17 at S2, and the
  least-squares problem has no finite minimiser (the ellipse "wants" to become infinitely large).
- **Ring-shaped Gaussian contact model** (main model): pressure concentrated on a ring,
  $p = A\exp(-(r-R)^2/(2\sigma^2))$, $r = |(x,y)-(x_0,y_0)|$.
  Fits the five sensors exactly, but with four different solutions. The one with the largest $\sigma$
  is selected (rule fixed before looking at the answers).

| Sensor | x | y | p |
|---|---|---|---|
| S1 | 0.0 | 0.5 | 2 |
| S2 | 1.3 | 1.1 | 17 |
| S3 | 1.9 | 0.1 | 43 |
| S4 | 2.5 | 2.3 | 28 |
| S5 | 0.7 | 1.8 | 36 |

The written analysis is in [`report/physical_contact_models.md`](report/physical_contact_models.md).

## Setup and run

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt       # Windows (Linux/macOS: .venv/bin/python)
.venv\Scripts\python -m pytest                                # 13 tests, about 1 minute
.venv\Scripts\python run_contact.py                           # full analysis, about 11 minutes
.venv\Scripts\python run_contact.py --mc 20 --out out         # quick run into out/ (keeps results/ intact)
```

Everything is seeded, so a full run reproduces the committed `results/` and `figures/`.

## Files

```
data.py          sensor data and convex-hull geometry
models.py        Hertz and ring models (formulas, analytic Jacobians), multi-start fitting,
                 selection rule (largest sigma), maximum / minimum over the sensor hull
run_contact.py   the analysis, section by section, plus the figures:
                   A1 Hertz fit (own LM vs scipy)      A2 Hertz degenerate limit
                   B1 all exact ring fits               B2 S2 hold-out / leave-one-out
                   B3 noise Monte Carlo (2/5/10 %)      B4 zero pressure
                   B5 maximum pressure (edges, Newton, projected ascent, step size)
                   B6 test case with a known answer
plotting.py      contour / surface / Hertz figures
numerics/        own implementations of the course methods
                   rootfind.py   bracket scan, bisection, Newton-Raphson
                   optimize.py   parabolic interpolation, multidimensional Newton, projected steepest ascent
                   nlls.py       Levenberg-Marquardt (active-set bounds) with multi-start
tests/           pytest: numerical methods on a known quadratic, Jacobians, parameter recovery
results/         CSV tables and numbers.json of the full run
figures/         figures of the full run
report/          written analysis (Markdown)
```

## Working together

- Pull before you start: `git pull`.
- Work on a branch for anything larger than a typo: `git checkout -b <topic>`, then open a pull request.
- Run `pytest` before pushing. If you change the analysis, re-run `run_contact.py` and commit the
  updated `results/` and `figures/` together with the code, so that the report numbers stay consistent.

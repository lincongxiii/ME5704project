"""Run from any directory. Produces a local HTML report, CSV, JSON and test log."""
from pathlib import Path
import argparse
import contextlib
import csv
import hashlib
import html
import json
import os
import platform
import runpy
import subprocess
import sys
import time
import webbrowser
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'physical_contact_models'))
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import scipy
import matplotlib
from validation.benchmarks import run_benchmarks


def git_info(*args):
    try:
        p=subprocess.run(['git',*args],cwd=ROOT,text=True,capture_output=True)
        return p.stdout.strip() if p.returncode==0 else None
    except OSError:
        return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'validation/output')
    parser.add_argument('--open',action='store_true',help='Open the generated local HTML report')
    parser.add_argument('--full-contact',action='store_true',help='Also rerun the 2000-start/1000-sample contact study (roughly 10+ minutes)')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    # Do not leave an old success page visible if a later stage raises.
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Validation in progress</title><h1>RUNNING OR INCOMPLETE</h1><p>This run has not completed. Inspect the terminal and log files if execution stopped.</p>',encoding='utf-8')
    start=time.perf_counter()
    meta={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
          'matplotlib':matplotlib.__version__,'quadratic_mc_seed':5704,'quadratic_mc_samples_per_level':1000,
          'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    meta['git_commit']=git_info('rev-parse','HEAD') or 'unavailable'
    state=git_info('status','--porcelain')
    meta['git_worktree_dirty']=bool(state) if state is not None else None
    meta['source_sha256']={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                          for folder in ['c','physical_contact_models','validation'] for p in sorted((ROOT/folder).rglob('*.py'))}
    print('[1/3] Running the actual project tests and integration tests...',flush=True)
    junit=out/'pytest.xml'
    junit.unlink(missing_ok=True)
    proc=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
                         str(ROOT/'physical_contact_models'),str(ROOT/'validation/tests'),f'--junitxml={junit}'],
                        cwd=ROOT,text=True,encoding='utf-8',errors='replace',capture_output=True)
    (out/'pytest.log').write_text((proc.stdout+'\n'+proc.stderr).rstrip()+'\n',encoding='utf-8')
    print(proc.stdout,flush=True)
    tests={'exit_code':proc.returncode}
    if junit.exists():
        suites=ET.parse(junit).getroot().findall('.//testsuite')
        tests.update({k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']})
    print('[2/3] Known-answer error tables...',flush=True)
    rows=run_benchmarks()
    with (out/'benchmarks.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('[3/3] Re-running the corrected quadratic Monte Carlo script (1000 samples/level)...',flush=True)
    # The existing programme is executed, not replaced by a look-alike simulation.
    old=os.getcwd()
    try:
        os.chdir(out)
        with (out/'quadratic_mc.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
            ns=runpy.run_path(str(ROOT/'c/noise_robustness.py'),run_name='__main__')
    finally:os.chdir(old)
    mc=[]
    for level in [0.,.02,.05,.1]:
        lo,hi=ns['all_hull_minima'][level],ns['all_hull_maxima'][level]
        mc.append(dict(noise_level=level,damage_count=int(np.sum(lo<=0)),pain_count=int(np.sum(hi>50)),
                       samples=len(lo),maximum_interval=np.percentile(hi,[2.5,97.5]).tolist()))
    full={'requested':args.full_contact,'status':'not rerun; report uses stored contact-study results'}
    if args.full_contact:
        print('Running the full contact study; progress is recorded in contact_full.log.',flush=True)
        with (out/'contact_full.log').open('w',encoding='utf-8') as log:
            full['exit_code']=subprocess.call([sys.executable,'-u',str(ROOT/'physical_contact_models/run_contact.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        full['status']='PASS' if full['exit_code']==0 else 'FAIL'
    ok=proc.returncode==0 and all(r['status']=='PASS' for r in rows) and full.get('exit_code',0)==0
    meta['elapsed_seconds']=time.perf_counter()-start
    result=dict(status='PASS' if ok else 'FAIL',metadata=meta,tests=tests,benchmarks=rows,quadratic_mc=mc,full_contact=full)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    esc=lambda x:html.escape(str(x))
    trs=''.join('<tr>'+''.join(f'<td>{esc(r[k])}</td>' for k in ['case','exact','numerical','absolute_error','tolerance','iterations','status'])+'</tr>' for r in rows)
    mrs=''.join(f'<tr><td>{100*r["noise_level"]:.0f}%</td><td>{r["damage_count"]}/{r["samples"]}</td><td>{r["pain_count"]}/{r["samples"]}</td><td>{esc(r["maximum_interval"])}</td></tr>' for r in mc)
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>ME5704 validation report</title>
<style>body{{font:16px/1.6 system-ui;margin:3rem auto;max-width:1250px;padding:0 1rem;color:#17263a}}h1,h2{{color:#17263a}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #ccd4df;overflow-wrap:anywhere}}th{{background:#eef3f8}}.status{{font-weight:bold;font-size:24px}}code{{background:#eef3f8}}.scroll{{overflow-x:auto}}a{{color:#135fa1}}@media print{{body{{margin:0}}}}</style>
<h1>ME5704 programme validation</h1><p class="status">{result['status']}</p>
<p>{esc(tests)} · {len(rows)} known-answer benchmark rows · {meta['elapsed_seconds']:.1f} seconds</p>
<p>Tests check implementation accuracy. They do not establish the true pressure field from five sensors.</p>
<p><a href="benchmarks.csv">Download error table</a> · <a href="results.json">Results and provenance</a> · <a href="pytest.log">Test log</a> · <a href="quadratic_mc.log">Monte Carlo log</a></p>
<h2>Analytical test case</h2><p>p(x,y) = 40 − 2(x−1)² − 3(y−2)². Its unique unconstrained maximum is 40 at (1,2). On x=1, its roots are y=2±√(40/3). Full-rank synthetic fitting uses six measurements. TPS is checked with the affine field 3+2x−4y.</p>
<h2>Measured errors</h2><div class="scroll"><table><tr><th>Case</th><th>Exact</th><th>Numerical</th><th>Max absolute error</th><th>Tolerance</th><th>Iterations</th><th>Status</th></tr>{trs}</table></div>
<h2>Quadratic Monte Carlo rerun</h2><p>Independent Gaussian sensor noise, seed 5704, standard deviation = noise level × 41. Hull extrema include vertices and stationary points.</p><table><tr><th>Noise</th><th>p minimum ≤ 0</th><th>p maximum &gt; 50</th><th>95% empirical maximum interval</th></tr>{mrs}</table>
<h2>Coverage and limitations</h2><p>Root scanning now includes the right endpoint. Off-grid tangencies and two roots in one scan cell can still be missed; regression tests explicitly document this limitation. A successful scan cannot prove all roots have been found. The quadratic zero-free conclusion instead uses a positive global minimum on the specified hull.</p>
<p>TPS here is an independent implementation, not a verification of unprovided MATLAB source. Contact-boundary derivatives, ring-centre nonsmoothness and LM stopping diagnostics need care. The standalone scripts in c are not all individually covered by this harness. The suite exercises the actual quadratic Monte Carlo fit/extrema functions and the contact numerical library. Consult validation/README.md for scope.</p>
<p>Full contact study: {esc(full['status'])}. Use --full-contact to regenerate the longer stored study. No web service or account is required.</p><h2>Environment</h2><pre>{esc(json.dumps(meta,indent=2))}</pre></html>'''
    target=out/'index.html';target.write_text(page,encoding='utf-8')
    print(f"{result['status']}: {target}",flush=True)
    if args.open:webbrowser.open(target.as_uri())
    return 0 if ok else 1


if __name__=='__main__':
    raise SystemExit(main())

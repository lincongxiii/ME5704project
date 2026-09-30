"""Known-answer tests exercise the actual project functions and report errors."""
import time
import numpy as np
from c.noise_robustness import fit_minimum_norm, predict, exact_hull_extrema, hull_vertices, X, p_original
from validation.tps import tps_fit
from numerics.rootfind import bisection, newton_raphson
from numerics.optimize import newton_nd, parabolic_max, projected_steepest_ascent
from numerics.nlls import levenberg_marquardt, numerical_jacobian
from models import Hertz, Ring, ring_hull_max


def run_benchmarks():
    rows = []

    def record(name, exact, operation, tol, note):
        start = time.perf_counter()
        value, iterations, valid = operation()
        elapsed = time.perf_counter()-start
        error = float(np.max(np.abs(np.asarray(value)-np.asarray(exact))))
        rows.append(dict(case=name, exact=np.asarray(exact).tolist(), numerical=np.asarray(value).tolist(),
                         absolute_error=error, tolerance=tol, iterations=iterations,
                         elapsed_seconds=elapsed, status='PASS' if valid and np.isfinite(error) and error <= tol else 'FAIL',
                         interpretation=note))

    f = lambda q: 40-2*(q[..., 0]-1)**2-3*(q[..., 1]-2)**2
    grad = lambda q: np.array([-4*(q[0]-1), -6*(q[1]-2)])
    hess = lambda q: np.diag([-4., -6.])
    train = np.array([[0,0], [1,0], [0,1], [2,0], [1,1], [0,2]], float)
    query = np.array([[.2,.7], [1,2], [2,3]], float)
    def fit_quad():
        b,m,s = fit_minimum_norm(train, f(train))
        return predict(query,b,m,s), None, True
    record('Quadratic recovery at unseen points', f(query), fit_quad, 1e-10,
           'Six full-rank synthetic measurements identify six coefficients; five project readings cannot uniquely identify a quadratic.')
    def fit_tps():
        values=3+2*X[:,0]-4*X[:,1]
        return tps_fit(X,values)[0](query), None, True
    record('TPS affine reproduction', 3+2*query[:,0]-4*query[:,1], fit_tps, 1e-10,
           'A TPS with affine augmentation must exactly reproduce affine pressure at unseen points.')
    def clean():
        b,m,s=fit_minimum_norm(X,p_original)
        return exact_hull_extrema(b,m,s), None, True
    record('Clean project hull extrema', [2.,43.], clean, 1e-10,
           'Regression check for the Monte Carlo grid omission; fixed to include sensor vertices.')
    for side, root in [('lower', 2-np.sqrt(40/3)), ('upper', 2+np.sqrt(40/3))]:
        fn=lambda y: 40-3*(y-2)**2
        interval=(-3.,2.) if side=='lower' else (2.,7.)
        def bis(interval=interval):
            r,k=bisection(fn,*interval)
            return r,k,abs(fn(r))<1e-8
        record('Bisection '+side+' root', root,bis,1e-10,'Known slice x=1; both analytical roots are checked.')
        def newt(interval=interval):
            r,k,ok=newton_raphson(fn,lambda y:-6*(y-2),interval[0] if side=='lower' else interval[1])
            return r,k,ok and abs(fn(r))<1e-8
        record('Newton '+side+' root',root,newt,1e-10,'Convergence flag and root residual are both checked.')
    for start in [[0.,0.], [4.,-1.]]:
        def opt(start=start):
            r=newton_nd(grad,hess,start)
            return [*r['x'],f(r['x'])],r['iterations'],r['converged'] and r['kind']=='maximum'
        record('Newton maximum start '+str(start),[1.,2.,40.],opt,1e-10,'Negative-definite Hessian gives the unique unconstrained maximum.')
    def ascent():
        r=projected_steepest_ascent(lambda q:f(q),grad,lambda q:q,np.zeros(2),alpha0=.1)
        return [*r['x'],f(r['x'])],r['iterations'],r['converged']
    record('Steepest ascent maximum',[1.,2.,40.],ascent,1e-6,'A local solver is checked against the known unique maximum.')
    def para():
        x,fx,k=parabolic_max(lambda x:40-2*(x-1)**2,2.,4.)
        return [x,fx],k,True
    record('Boundary maximum',[2.,38.],para,1e-10,'Unconstrained optimum x=1 is outside [2,4]; endpoint comparison is required.')
    def quad_hull():
        # f=26+4x+12y-2x^2-3y^2; derive exact edge t from this known formula.
        return exact_hull_extrema(np.array([26.,4,12,-2,0,-3]),np.zeros(2),np.ones(2)),None,True
    candidates=[f(v) for v in hull_vertices]
    for a,b in zip(hull_vertices,np.roll(hull_vertices,-1,axis=0)):
        d=b-a
        t=np.clip(-(4*(a[0]-1)*d[0]+6*(a[1]-2)*d[1])/(4*d[0]**2+6*d[1]**2),0,1)
        candidates.append(f(a+t*d))
    record('Synthetic quadratic constrained extrema',[min(candidates),max(candidates)],quad_hull,1e-10,
           'Analytical edge derivatives provide the reference on the actual sensor polygon; (1,2) is outside it.')
    xx=np.linspace(0,2,12)
    def nonlinear():
        yy=3*np.exp(-.7*xx)+.5
        r=levenberg_marquardt(lambda t:t[0]*np.exp(-t[1]*xx)+t[2]-yy,[1.,.1,0.],[-10]*3,[10]*3)
        return r['theta'],r['iterations'],r['converged'] and r['cost']<1e-12
    record('Nonlinear parameter recovery',[3.,.7,.5],nonlinear,1e-7,'Known exponential parameters; both parameter error and residual cost are checked.')
    def jac_check():
        theta=np.array([40.,1.3,1.,1.2,1.3])
        j=Hertz.jac(theta,X[:,0],X[:,1])
        ref=numerical_jacobian(lambda t:Hertz.residual(t,X[:,0],X[:,1],p_original),theta)
        return float(np.max(np.abs(j-ref))),None,True
    record('Hertz Jacobian cross-check',0.,jac_check,1e-4,
           'Central differences away from the contact boundary; outside-contact data derivatives must be zero.')
    def crest():
        v,x,y,_=ring_hull_max(np.array([50.,1.2,1.1,.5,.3]))
        return [v,np.hypot(x-1.2,y-1.1)],None,True
    record('Known ring maximum',[50.,.5],crest,1e-8,'Crest circle intersects the hull; p<=A proves the maximum is 50.')
    return rows

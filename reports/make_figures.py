"""Regenerate the English colour comparison figures from full-precision results."""
from pathlib import Path
import sys
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'physical_contact_models'))
from c.noise_robustness import X,p_original,fit_minimum_norm,predict,hull_vertices
from validation.tps import tps_fit
from models import Hertz,Ring
OUT=ROOT/'reports/figures';OUT.mkdir(exist_ok=True)
nums=json.loads((ROOT/'physical_contact_models/results/numbers.json').read_text())
data=json.loads((ROOT/'reports/integrated_data.json').read_text())
th=np.array([nums['hertz'][k] for k in Hertz.names])
tr=np.array([nums['ring_solutions']['selected'][k] for k in Ring.names])
b,m,s=fit_minimum_norm(X,p_original)
funcs=[lambda q:predict(q,b,m,s),tps_fit(X,p_original)[0],lambda q:Hertz.f(th,q[:,0],q[:,1]),lambda q:Ring.f(tr,q[:,0],q[:,1])]
names=['Quadratic','TPS','Hertz','Ring (largest width)']
eq=ConvexHull(hull_vertices).equations
poly=np.vstack([hull_vertices,hull_vertices[:1]])
def grid(x0,x1,y0,y1,n):
    xx,yy=np.meshgrid(np.linspace(x0,x1,n),np.linspace(y0,y1,n))
    return xx,yy,np.c_[xx.ravel(),yy.ravel()]
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
xx,yy,q=grid(0,2.5,.1,2.3,251)
inside=np.all(q@eq[:,:2].T+eq[:,2]<=1e-10,axis=1)
fig,axs=plt.subplots(2,2,figsize=(9,7),layout='constrained')
for ax,f,name in zip(axs.flat,funcs,names):
    z=f(q).reshape(xx.shape);z[~inside.reshape(xx.shape)]=np.nan
    im=ax.contourf(xx,yy,z,levels=np.linspace(0,70,29),cmap='viridis',extend='max')
    if np.nanmax(z)>50:ax.contour(xx,yy,z,levels=[50],colors='white',linestyles='--',linewidths=1.5)
    ax.plot(*poly.T,c='black',lw=1);ax.scatter(*X.T,c='white',edgecolors='black',s=25)
    for i,v in enumerate(X):ax.annotate(f'S{i+1}',v,xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set(title=name,xlabel='x',ylabel='y',aspect='equal')
fig.colorbar(im,ax=axs,label='Pressure (assignment units)',shrink=.85)
fig.savefig(OUT/'comparison.png',dpi=220);plt.close(fig)
xx,yy,q=grid(-.6,4.5,-.5,3.5,401)
fig,ax=plt.subplots(figsize=(9,4.2),layout='constrained')
for f,name,ls,color in zip(funcs[:2],names[:2],['-','--'],['#0072B2','#D55E00']):
    ax.contour(xx,yy,f(q).reshape(xx.shape),levels=[0],colors=[color],linestyles=ls,linewidths=1.7)
    ax.plot([],[],c=color,ls=ls,label=f'{name}: p=0')
ang=np.linspace(0,2*np.pi,3000)
ax.plot(th[1]+th[3]*np.cos(ang),th[2]+th[4]*np.sin(ang),c='#009E73',ls=':',label='Hertz contact boundary')
ax.fill(*poly.T,alpha=.12,color='#56B4E9');ax.plot(*poly.T,c='black',lw=1,label='Sensor hull')
ax.scatter(*X.T,c='black',s=18)
ax.set(xlim=(-.6,4.5),ylim=(-.5,3.5),xlabel='x',ylabel='y',aspect='equal');ax.legend(fontsize=8,loc='upper left',ncols=2)
fig.savefig(OUT/'zero_contours.png',dpi=220);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
axs[0].bar(['Hull','Rectangle'],[data['quadratic_hull'][1],data['quadratic_box'][1]],label='Quadratic',width=.35,align='edge',color='#0072B2',edgecolor='black')
axs[0].bar(['Hull','Rectangle'],[data['tps_hull_numeric'][1],data['tps_box_numeric'][1]],label='TPS',width=-.35,align='edge',color='#E69F00',edgecolor='black',hatch='///')
axs[0].axhline(50,color='#b2182b',ls='--');axs[0].set(ylabel='Maximum pressure',title='Domain changes the conclusion');axs[0].legend(fontsize=8)
rs=list(csv.DictReader((ROOT/'physical_contact_models/results/b1_ring_exact_fits.csv').open()))
axs[1].bar(['1','2','3','4'],[float(r['A']) for r in rs],color=['#0072B2','#009E73','#E69F00','#CC79A7'],edgecolor='black')
axs[1].axhline(50,color='#b2182b',ls='--');axs[1].set(xlabel='Exact ring fit',ylabel='Maximum in hull',title='Four fits to the same five readings')
fig.savefig(OUT/'maxima.png',dpi=220);plt.close(fig)
print('Wrote three colour figures to',OUT)

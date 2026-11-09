"""Independent small-grid distribution and covariance checks."""
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import roots_hermitenorm
from scipy.stats import t as student_t
from run_rate_test import (HERE,filters,drift_weights,coefficients,native,simulate,
                           coarse_filters,MODELS)

def normal_payoffs(mean,var):
    return np.array([np.cos(.7*mean)*np.exp(-.49*var/2),
                     np.sin(.7*mean)*np.exp(-.49*var/2),
                     np.exp(-(mean-.4)**2/(2*(1+var)))/np.sqrt(1+var),
                     np.cos(mean)*np.exp(-var/2)])

def two_step(H,model,order):
    _,C,_=filters(H,2)
    d=drift_weights(H,2); x0=0. if model==4 else .3
    b0,s0=coefficients(np.array([x0]),model);b0,s0=b0[0],s0[0]
    nodes,weights=roots_hermitenorm(order); weights/=math.sqrt(2*math.pi)
    z=nodes*math.sqrt(C[0,0]); y=x0+d[0]*b0+s0*z
    b,s=coefficients(y,model)
    mean=x0+d[1]*b0+d[0]*b+s0*C[0,1]/C[0,0]*z
    variance=s0*s0*(C[1,1]-C[0,1]**2/C[0,0])+s*s*C[0,0]
    value=normal_payoffs(mean,variance)@weights
    one=normal_payoffs(x0+drift_weights(H,1)[0]*b0,s0*s0)
    return value-one

def main():
    (HERE/'results').mkdir(parents=True,exist_ok=True)
    lib=native();rows=[];covariances=[]
    for H in (.1,.25,.35):
        values,diag=simulate(lib,H,2,[1,3,4],12,16,8850000+round(H*1e4),128)
        for j,m in enumerate((1,3,4)):
            reference=two_step(H,m,256); earlier=two_step(H,m,128)
            mean=values[:,j].mean(axis=0);se=values[:,j].std(axis=0,ddof=1)/4
            for k,F in enumerate(('cos07','sin07','gaussian','cos1')):
                rows.append(dict(H=H,model=MODELS[m],payoff=F,
                                 deterministic_difference=float(reference[k]),
                                 quadrature_order_difference=float(abs(reference[k]-earlier[k])),
                                 estimate=float(mean[k]),standard_error=float(se[k]),
                                 standardized_discrepancy=float((mean[k]-reference[k])/se[k]) if se[k]>0 else 0.))
        for n in (32,64,128,256):
            K,C,d=filters(H,n);K2,C2,d2=filters(H,n,tol=1e-16,order=128)
            coarse=coarse_filters(K);_,Cc,_=filters(H,n//2)
            d.update(coarse_covariance_error=float(np.max(abs(coarse.T@coarse-Cc))),
                     tighter_covariance_difference=float(np.max(abs(K.T@K-K2.T@K2))),
                     tighter_channels=len(K2))
            covariances.append(d)
    out=HERE/'results'/'reference_checks.json'
    result=dict(two_step_checks=rows,covariance_checks=covariances,
                note='Deterministic quadrature estimates and roundoff checks, not interval certificates. No diagonal jitter added.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print('Max quadrature order difference:',max(r['quadrature_order_difference'] for r in rows))
    print('Max absolute standardized discrepancy:',max(abs(r['standardized_discrepancy']) for r in rows))
    print('Max covariance perturbation:',max(r['tighter_covariance_difference'] for r in covariances))
    print(out)

if __name__=='__main__':main()

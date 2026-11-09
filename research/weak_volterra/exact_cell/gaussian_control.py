"""Absolute weak bias in the affine-drift, additive Gaussian subclass.

This is a control, not numerical evidence for nonlinear variance.
Mittag-Leffler power series plus two quadrature orders supply the continuous
mean and variance. No Monte Carlo and no extrapolated continuous reference.
"""
import csv
import json
import math
import numpy as np
from scipy.special import gamma, roots_legendre
from scipy.signal import lfilter
from run_rate_test import HERE, filters, drift_weights
from check_references import normal_payoffs

def continuum(H,order,terms=80):
    alpha,beta=H+.5,2*H;a=math.sqrt(beta)*gamma(alpha);k=.7
    coeff1=1/gamma(alpha*np.arange(terms)+1)
    coeff2=1/gamma(alpha*(np.arange(terms)+1))
    m=.2+.1*np.polynomial.polynomial.polyval(-k*a,coeff1)
    nodes,weights=roots_legendre(order);y=(nodes+1)/2;weights=weights/2
    f=np.polynomial.polynomial.polyval(-k*a*y**(alpha/beta),coeff2)
    v=a*a/beta*np.dot(weights,f*f)
    return float(m),float(v)

def main():
    (HERE/'results').mkdir(parents=True,exist_ok=True)
    rows=[];references=[]
    for H in (.1,.25,.35):
        m,v=continuum(H,512);m2,v2=continuum(H,1024)
        ref=normal_payoffs(m2,v2)
        references.append(dict(H=H,mean=m2,variance=v2,quadrature_difference=abs(v-v2)))
        for n in (16,32,64,128,256,512):
            K,C,diag=filters(H,n);d=drift_weights(H,n)
            R=lfilter([1.],np.r_[1.,.7*d[:-1]],K,axis=1)
            mean=np.empty(n+1);mean[0]=.3
            for i in range(1,n+1):mean[i]=.3+.7*np.dot(d[:i][::-1],.2-mean[:i])
            variance=float(np.sum(R*R));value=normal_payoffs(mean[-1],variance)
            for k,F in enumerate(('cos07','sin07','gaussian','cos1')):
                rows.append(dict(H=H,N=n,payoff=F,discrete=float(value[k]),continuous=float(ref[k]),
                                 bias=float(value[k]-ref[k]),mean=float(mean[-1]),variance=variance))
    out=HERE/'results'
    with (out/'gaussian_control.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'gaussian_control_metadata.json').write_text(json.dumps(dict(references=references,
        scope='Affine drift/additive noise absolute weak bias; stronger subclass rates are known.'),indent=2)+'\n')
    print(references)

if __name__=='__main__':main()

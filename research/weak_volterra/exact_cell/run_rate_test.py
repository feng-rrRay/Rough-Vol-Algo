"""Fixed nonlinear mesh-difference test of exact-cell Euler weak upper bounds.

Numerical Gaussian covariance factorization, not the randomized Haar algorithm.
Nested fine-cell Gaussian sources couple Y_N and Y_{N/2} on the same Brownian
motion. No continuous nonlinear oracle or fixed-model asymptotic is assumed.
Randomized QMC uncertainty is computed across INDEPENDENT scrambles.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/weak_volterra_matplotlib')
import argparse
import csv
import ctypes
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import ndtri
from scipy.stats import qmc, t as student_t

HERE=Path(__file__).resolve().parent
PTR=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
MODELS={0:'G0',1:'N1',2:'N2',3:'N3',4:'R',5:'G1'}

def native():
    src=HERE/'rate_kernels.c'; dst=HERE/'rate_kernels.dylib'
    if not dst.exists() or src.stat().st_mtime>dst.stat().st_mtime:
        subprocess.run(['clang','-O3','-shared','-fPIC',str(src),'-o',str(dst)],check=True)
    lib=ctypes.CDLL(str(dst))
    lib.rate_leaf.argtypes=[ctypes.c_int]*6+[PTR]*5
    lib.rate_leaf.restype=None
    return lib

def drift_weights(H,n):
    alpha=H+.5
    l=np.arange(1,n+1,dtype=float)
    increments=np.ones(n)
    increments[1:]=l[1:]**alpha*(-np.expm1(alpha*np.log1p(-1/l[1:])))
    return np.sqrt(2*H)*n**(-alpha)*increments/alpha

def residual_covariance(H,n,order):
    gamma,alpha,beta=H-.5,H+.5,2*H
    nodes,ws=leggauss(order); u=(nodes+1)/2;ws=ws/2
    centers=np.arange(2,n+1,dtype=float)-.5
    values=centers[:,None]**gamma*np.expm1(gamma*np.log1p((u[None,:]-.5)/centers[:,None]))
    means=values@ws; centered=values-means[:,None]
    B=np.empty((n,n)); B[0,0]=1/beta-1/alpha**2
    B[1:,1:]=(centered*ws)@centered.T
    reported=0.
    for k,c in enumerate(centers):
        val,err=quad(lambda x:c**gamma*math.expm1(gamma*math.log1p((x-.5)/c))-means[k],
                     0.,1.,weight='alg',wvar=(gamma,0.),epsabs=2e-14,epsrel=2e-13)
        B[0,k+1]=B[k+1,0]=val; reported=max(reported,err)
    return B*(2*H)*n**(-2*H),reported*(2*H)*n**(-2*H)

def filters(H,n,tol=1e-14,order=96):
    B,err=residual_covariance(H,n,order)
    B2,err2=residual_covariance(H,n,order+32)
    vals,vecs=np.linalg.eigh(B2); keep=vals>tol
    v=vals[keep][::-1]; V=vecs[:,keep][:,::-1]
    # Fix signs for reproducibility; all omitted eigenvalues are recorded.
    V*=np.where(V[0]>=0,1.,-1.)
    a=drift_weights(H,n)*math.sqrt(n)
    K=np.ascontiguousarray(np.vstack((a,(V*np.sqrt(v)).T)))
    C=np.outer(a,a)+B2
    diag=dict(H=H,N=n,channels=len(K),threshold=tol,
              covariance_quadrature_max_difference=float(np.max(abs(B-B2))),
              covariance_quadrature_reported_error=max(err,err2),
              negative_eigenvalue_mass=float(-vals[vals<0].sum()),
              discarded_positive_trace=float(vals[(vals>0)&~keep].sum()),
              covariance_reconstruction_max_error=float(np.max(abs(K.T@K-C))),
              terminal_variance_error=float(np.sum(K*K)-1),eigenvalues=v.tolist())
    return K,C,diag

class Solver:
    def __init__(self,lib,K,H,model,leaf=16):
        self.lib,self.K,self.H,self.model,self.leaf=lib,np.ascontiguousarray(K),H,model,leaf
        self.r,self.n=K.shape; self.x0=0. if model==4 else .3
        self.d=np.ascontiguousarray(drift_weights(H,self.n))
        self.ffts={}
        def setup(w):
            if w<=leaf:return
            if w not in self.ffts:
                self.ffts[w]=np.fft.rfft(np.vstack((self.K[:,:w-1],self.d[:w-1])),2*w,axis=1)
            setup(w//2);setup(w-w//2)
        setup(self.n)

    def solve(self,z,direct=False):
        z=np.ascontiguousarray(z);r,p,n=z.shape
        assert (r,n)==(self.r,self.n)
        x=np.full((p,n+1),self.x0); source=np.zeros((r+1,p,n))
        def rec(left,right):
            w=right-left
            if direct or w<=self.leaf:
                self.lib.rate_leaf(p,n,r,left,right,self.model,self.K,self.d,z,x,source)
                return
            mid=(left+right)//2;rec(left,mid)
            transformed=np.fft.rfft(source[:,:,left:mid],2*w,axis=2)
            combined=np.einsum('rpf,rf->pf',transformed,self.ffts[w])
            value=np.fft.irfft(combined,2*w,axis=1)
            x[:,mid:right]+=value[:,mid-left-1:right-left-1]
            rec(mid,right)
        rec(0,n)
        x[:,n]+=np.einsum('rpn,rn->p',source,np.vstack((self.K,self.d))[:,::-1])
        return x

def coarse_filters(K):
    r,n=K.shape
    assert n%2==0
    # Source i has half-cell a=0,1, and lag on the fine grid 2*l-a.
    return np.ascontiguousarray(np.concatenate((K[:,1::2],K[:,0::2]),axis=0))

def coarse_noise(z):
    return np.ascontiguousarray(np.concatenate((z[:,:,0::2],z[:,:,1::2]),axis=0))

def coefficients(x,model):
    q=np.ones_like(x);b=np.zeros_like(x)
    if model in (1,2):q=1+.8*np.sin(x)
    if model in (2,5):b=.7*(.2-x)
    if model==3:q=1+.6*np.tanh(x)+.2*np.sin(2*x);b=-.4*x+.2*np.sin(x)
    if model==4:q=1+.5*np.cos(x)
    return b,np.sqrt(q)

def python_direct(K,d,z,model):
    r,p,n=z.shape;x=np.full((p,n+1),0. if model==4 else .3)
    for i in range(n):
        b,s=coefficients(x[:,i],model)
        noise=np.einsum('rp,rl->pl',z[:,:,i],K[:,:n-i])
        x[:,i+1:]+=b[:,None]*d[:n-i]+s[:,None]*noise
    return x

def validate(lib):
    rng=np.random.default_rng(881710);result={};maximum=0.;coupling=0.
    for n in (1,2,3,8,16,31,32,33):
        H=.1;K,C,diag=filters(H,n)
        z=rng.normal(size=(len(K),5,n))
        for model in MODELS:
            s=Solver(lib,K,H,model,leaf=4)
            a=s.solve(z);b=python_direct(K,s.d,z,model)
            maximum=max(maximum,float(np.max(abs(a-b))))
            assert np.max(abs(a-b))<3e-12,(n,model)
        if n%2==0:
            kc=coarse_filters(K)
            _,cc,_=filters(H,n//2)
            coupling=max(coupling,float(np.max(abs(kc.T@kc-cc))))
            fine=Solver(lib,K,H,0).solve(z)[:,::2]
            coarse=Solver(lib,kc,H,0).solve(coarse_noise(z))
            assert np.max(abs(fine-coarse))<3e-12
            # Independent source aggregation for genuinely nonlinear coefficients.
            model=3; nc=n//2; xx=np.full((5,nc+1),.3)
            for i in range(nc):
                b,s=coefficients(xx[:,i],model)
                for j in range(i+1,nc+1):
                    noise=sum(np.einsum('rp,r->p',z[:,:,2*i+a],K[:,2*(j-i)-a-1]) for a in (0,1))
                    xx[:,j]+=b*drift_weights(H,nc)[j-i-1]+s*noise
            check=Solver(lib,kc,H,model,leaf=4).solve(coarse_noise(z))
            assert np.max(abs(xx-check))<3e-12
    assert coupling<3e-12
    result.update(cdq_vs_python_max=maximum,coarse_covariance_max=coupling,
                  source_aggregation='passed',additive_grid_equality='passed')
    return result

def payoffs(x):
    return np.column_stack((np.cos(.7*x),np.sin(.7*x),np.exp(-.5*(x-.4)**2),np.cos(x)))

def terminal_rotation(z,K):
    # An orthogonal Householder reflection puts the common additive terminal
    # state in Sobol coordinate zero. This changes no Gaussian marginal law.
    p=len(z); v=K[:,::-1].reshape(-1).copy();v/=np.linalg.norm(v);v[0]-=1
    v/=np.linalg.norm(v)
    return z-2*np.outer(z@v,v)

def simulate(lib,H,n,models,power,reps,seed,batch,method='sobol',tol=1e-14):
    K,C,diag=filters(H,n,tol=tol);r=len(K);kc=coarse_filters(K)
    fs={m:Solver(lib,K,H,m) for m in models};cs={m:Solver(lib,kc,H,m) for m in models}
    estimates=np.empty((reps,len(models),4)); start=time.perf_counter()
    for rep in range(reps):
        gen=qmc.Sobol(d=r*n,scramble=True,bits=52,seed=seed+rep) if method=='sobol' else np.random.default_rng(seed+rep)
        total=np.zeros((len(models),4))
        for first in range(0,2**power,batch):
            size=min(batch,2**power-first)
            # Use midpoints of 52-bit dyadic bins: strictly inside (0,1),
            # without clipping tails or accepting inverse-normal infinities.
            raw=ndtri(gen.random(size)+2.**(-53)) if method=='sobol' else gen.normal(size=(size,r*n))
            assert np.isfinite(raw).all(), 'Non-finite Gaussian draw'
            raw=terminal_rotation(raw,K)
            z=np.ascontiguousarray(raw.reshape(size,r,n).transpose(1,0,2));zc=coarse_noise(z)
            for k,m in enumerate(models):
                for sign in (1.,-1.):
                    yf=fs[m].solve(sign*z)[:,-1];yc=cs[m].solve(sign*zc)[:,-1]
                    assert np.isfinite(yf).all() and np.isfinite(yc).all(), 'Non-finite state'
                    total[k]+=.5*(payoffs(yf)-payoffs(yc)).sum(axis=0)
        estimates[rep]=total/(2**power)
        if rep%4==3 or rep==reps-1:
            print('H={} fineN={} {} rep={}/{} elapsed={:.1f}s'.format(H,n,method,rep+1,reps,time.perf_counter()-start),flush=True)
    diag.update(seconds=time.perf_counter()-start,seed=seed,power=power,reps=reps,method=method,
                sobol_bits=52 if method=='sobol' else None,sobol_bin_midpoint_shift=2.**(-53) if method=='sobol' else None,
                independent_unit='one independently scrambled net' if method=='sobol' else 'one independent pseudorandom batch')
    return estimates,diag

def save(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hurst',type=float,nargs='+',default=[.1,.25,.35])
    p.add_argument('--grids',type=int,nargs='+',default=[32,64,128,256])
    p.add_argument('--models',type=int,nargs='+',default=[1,3,4])
    p.add_argument('--power',type=int,default=10);p.add_argument('--reps',type=int,default=8)
    p.add_argument('--batch',type=int,default=128);p.add_argument('--seed',type=int,default=202611060)
    p.add_argument('--tag',default='pilot');p.add_argument('--method',choices=['sobol','mc'],default='sobol')
    p.add_argument('--tol',type=float,default=1e-14);p.add_argument('--validate-only',action='store_true')
    a=p.parse_args();out=HERE/'results'/a.tag;out.mkdir(parents=True,exist_ok=True)
    lib=native();checks=validate(lib);print('Validation:',checks,flush=True)
    meta=dict(settings=vars(a),validation=checks,platform=platform.platform(),numpy=np.__version__,
              scope='Fixed nonlinear exact-cell mesh differences, not continuous bias or Haar algorithm accuracy.',
              statistical_note='Pointwise t intervals across independent replicate means; empirical, not deterministic certificates.',
              kernel='sqrt(2H)*t**(H-0.5)',T=1.,diagnostics=[],
              source_hashes={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ('run_rate_test.py','rate_kernels.c')})
    rows=[]
    if not a.validate_only:
        for H in a.hurst:
            for n in a.grids:
                seed=a.seed+round(H*1e4)*10000+n*32
                values,diag=simulate(lib,H,n,a.models,a.power,a.reps,seed,a.batch,a.method,a.tol)
                np.savez(out/('H{}_N{}.npz'.format(H,n)),replicate_means=values,models=a.models)
                meta['diagnostics'].append(diag)
                for j,m in enumerate(a.models):
                    for k,F in enumerate(('cos07','sin07','gaussian','cos1')):
                        mean=float(values[:,j,k].mean());se=float(values[:,j,k].std(ddof=1)/math.sqrt(a.reps))
                        radius=float(student_t.ppf(.975,a.reps-1)*se)
                        rows.append(dict(H=H,N_coarse=n//2,N_fine=n,model=MODELS[m],payoff=F,
                                         difference=mean,standard_error=se,ci95_low=mean-radius,ci95_high=mean+radius,
                                         independent_replicates=a.reps,points_per_replicate=2**a.power,
                                         antithetic_paths=2*a.reps*2**a.power))
                save(out/'differences.csv',rows)
                (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
                for row in rows[-4*len(a.models):]:
                    if row['payoff'] in ('cos07','cos1'):
                        print(row,flush=True)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':main()

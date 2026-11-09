"""Fixed-maturity compression comparisons with a numerical Gaussian reference.

Uses a direct C solver for accuracy experiments, NOT for the FFT cost claim.
Covariance quadrature and spectral truncation are checked and recorded, not
certified as an exact reference. Standard errors are from independent paths.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name]='1'
import argparse
import csv
import ctypes
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from verify_randomized_haar_correction import weights, mode_weights, solve_antithetic_paths

HERE=Path(__file__).resolve().parent
PTR=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')


def native():
    source=HERE/'haar_benchmark_kernels.c'
    library=HERE/'results'/'haar_benchmark_kernels.dylib'
    library.parent.mkdir(parents=True,exist_ok=True)
    if not library.exists() or library.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(['clang','-O3','-shared','-fPIC',str(source),'-o',str(library)],check=True)
    c=ctypes.CDLL(str(library))
    c.haar_reference_batch.argtypes=[ctypes.c_int]*3+[ctypes.c_double]*2+[PTR]*6
    c.haar_reference_batch.restype=None
    return c


def covariance(H,n,order=64):
    """Integrate residual products directly, avoiding C-aa^T cancellation."""
    gamma,alpha,beta=H-.5,H+.5,2*H
    nodes,ws=leggauss(order)
    u=(nodes+1)/2;ws=ws/2
    centers=np.arange(2,n+1,dtype=float)-.5
    values=centers[:,None]**gamma*np.expm1(gamma*np.log1p((u[None,:]-.5)/centers[:,None]))
    means=values@ws
    centered=values-means[:,None]
    B=np.empty((n,n));B[0,0]=1/beta-1/alpha**2
    B[1:,1:]=(centered*ws)@centered.T
    maximum_quad_error=0.
    for k,center in enumerate(centers):
        # x=1-u, so (lag-u)^gamma=(lag-1+x)^gamma.
        def integrand(x):
            return center**gamma*math.expm1(gamma*math.log1p((x-.5)/center))-means[k]
        val,err=quad(integrand,0.,1.,weight='alg',wvar=(gamma,0.),
                     epsabs=2e-14,epsrel=2e-13)
        B[0,k+1]=B[k+1,0]=val
        maximum_quad_error=max(maximum_quad_error,err)
    B*=n**(-beta)/math.gamma(alpha)**2
    return B,maximum_quad_error*n**(-beta)/math.gamma(alpha)**2


def reference(H,n):
    started=time.perf_counter()
    B,e1=covariance(H,n,64);B2,e2=covariance(H,n,96)
    vals,vecs=np.linalg.eigh(B)
    negative_mass=float(-vals[vals<0].sum())
    keep=vals>1e-14
    dropped=float(vals[(vals>0)&~keep].sum())
    vals=vals[keep][::-1];vecs=vecs[:,keep][:,::-1]
    for k in range(len(vals)):
        if vecs[0,k]<0:vecs[:,k]*=-1
    factors=np.ascontiguousarray((vecs*np.sqrt(vals)).T)
    diag=dict(H=H,n=n,rank=len(vals),quadrature_order_difference=float(np.max(abs(B-B2))),
              quad_reported_error=max(e1,e2),negative_eigenvalue_mass=negative_mass,
              discarded_positive_mass=dropped,retained_eigenvalues=vals.tolist(),
              preprocessing_seconds=time.perf_counter()-started)
    return B,factors,diag


def ensemble(H,n):
    # Enumeration is only for a reproducible accuracy experiment. The actual
    # algorithm uses rejection sampling and avoids this O(N^2) setup.
    cap=int(math.ceil(math.log2(n)))
    raw=np.array([mode_weights(H,n,j,d,normalized=False)[1:]
                  for j in range(cap+1) for d in range(1,2**j+1)])
    energies=np.sum(raw*raw,axis=1)
    a,s=weights(H,n);total=float(np.dot(s,s))
    normalized=raw*np.sqrt(total/energies)[:,None]
    return np.ascontiguousarray(normalized),energies/energies.sum()


class EnergySampler:
    """Cached rejection sampler; no enumeration of levels or positions."""
    def __init__(self,H,n,cap):
        self.H,self.n,self.cap=H,n,cap
        self.beta,self.r=2*H,2-2*H
        _,s=weights(H,n);self.target=float(np.dot(s,s))
        first=mode_weights(H,n,0,1,normalized=False)
        self.envelope=float(np.dot(first,first))/((1-2**(-self.beta))*(1-2**(-self.r)))
        self.cache={};self.proposals=0;self.accepted=0

    def draw(self,paths,count,rng):
        needed=paths*count;accepted=[]
        while len(accepted)<needed:
            batch=max(32,3*(needed-len(accepted)))
            levels=np.floor(-np.log(rng.uniform(size=batch))/(self.beta*math.log(2))).astype(int)
            distances=np.floor(np.exp(-np.log(rng.uniform(size=batch))/self.r)).astype(np.int64)
            uniform=rng.uniform(size=batch)
            self.proposals+=batch
            for j,d,u in zip(levels,distances,uniform):
                j,d=int(j),int(d)
                if j>self.cap or d>2**j:continue
                key=(j,d)
                if key not in self.cache:
                    proposed=mode_weights(self.H,self.n,j,d)[1:]
                    energy=float(np.dot(proposed,proposed))
                    assert energy<=self.envelope*(1+1e-10)
                    self.cache[key]=(proposed*math.sqrt(self.target/energy),energy/self.envelope)
                mode,probability=self.cache[key]
                if u<probability:
                    accepted.append(mode)
                    self.accepted+=1
                    if len(accepted)==needed:break
        return np.array(accepted).reshape(paths,count,self.n)/math.sqrt(count)


def solve(c,a,d,filters,xi,eta,epsilon,drift):
    p,r,n=filters.shape
    out=np.empty((p,4))
    c.haar_reference_batch(p,n,r,epsilon,drift,*[np.ascontiguousarray(v)
                            for v in (a,d,filters,xi,eta)],out)
    return out


def controls(out):
    # Constant-diffusion terminal laws coincide for energy-normalized modes.
    # Return raw antithetic payoff and its centered additive Gaussian control.
    result={}
    for label,fun in [('sin',np.sin),('cos',np.cos)]:
        y=fun(.7*out)
        raw=.5*(y[:,0]+y[:,1])
        result[label]=(raw,raw-.5*(y[:,2]+y[:,3]))
    return result


def validate(c):
    rng=np.random.RandomState(71);n=16
    a,s=weights(.1,n);d=a[1:]/math.sqrt(n)
    xi=rng.normal(size=(3,n));eta=rng.normal(size=(3,1,n))
    filters=np.broadcast_to(s[None,None,1:],(3,1,n))
    result=solve(c,a[1:],d,filters,xi,eta,.8,.2)
    for p in range(3):
        check=solve_antithetic_paths(a,s,np.zeros(n+1),xi[p],eta[p,0],
                                    np.zeros(n),drift_strength=.2,h=1./n)
        assert np.max(abs(check[:2,-1]-result[p,:2])) < 2e-12
    gaussian=solve(c,a[1:],d,filters,xi,eta,0.,0.)
    assert np.max(abs(gaussian[:,:2]-gaussian[:,2:])) < 2e-12
    print('PASS: independent native/CDQ and additive-control checks',flush=True)


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hurst',type=float,default=.1)
    parser.add_argument('--grids',type=int,nargs='+',default=[16,32,64])
    parser.add_argument('--modes',type=int,nargs='+',default=[1,4])
    parser.add_argument('--paths',type=int,default=32768)
    parser.add_argument('--batch',type=int,default=256)
    parser.add_argument('--epsilon',type=float,default=.8)
    parser.add_argument('--drift',type=float,default=.2)
    parser.add_argument('--tag',default='pilot')
    parser.add_argument('--cap',type=int,default=None,
                        help='Use cached rejection sampling with this fixed maximum Haar level.')
    args=parser.parse_args();c=native();validate(c)
    rng=np.random.RandomState(20261105)
    diagnostics=[];rows=[]
    started=time.perf_counter()
    for n in args.grids:
        B,L,diag=reference(args.hurst,n);diagnostics.append(diag)
        if args.cap is None:
            modes,probs=ensemble(args.hurst,n);sampler=None
        else:
            sampler=EnergySampler(args.hurst,n,args.cap)
        a,s=weights(args.hurst,n);a=a[1:];s=s[1:]
        d=a/math.sqrt(n);r=len(L)
        labels=['original','spectral1','spectral2']+['haar{}'.format(m) for m in args.modes]
        sums={(label,payoff):np.zeros(4) for label in labels for payoff in ('sin','cos')}
        counts=0
        for begin in range(0,args.paths,args.batch):
            p=min(args.batch,args.paths-begin)
            xi=rng.normal(size=(p,n))
            common=rng.normal(size=(p,n))
            other=rng.normal(size=(p,max(0,r-1),n))
            ref_eta=np.concatenate((common[:,None,:],other),axis=1)
            ref_filters=np.ascontiguousarray(np.broadcast_to(L,(p,r,n)))
            ref=controls(solve(c,a,d,ref_filters,xi,ref_eta,args.epsilon,args.drift))
            candidates={
                'original':(np.broadcast_to(s[None,None,:],(p,1,n)),common[:,None,:]),
                'spectral1':(np.broadcast_to(L[:1],(p,1,n)),ref_eta[:,:1,:]),
                'spectral2':(np.broadcast_to(L[:2],(p,2,n)),ref_eta[:,:2,:])}
            for m in args.modes:
                if sampler is None:
                    chosen=rng.choice(len(probs),size=(p,m),p=probs)
                    filt=modes[chosen]/math.sqrt(m)
                else:
                    filt=sampler.draw(p,m,rng)
                # Orthogonal decomposition: the average of these m independent
                # Gaussian streams is exactly the reference's leading stream.
                noise=rng.normal(size=(p,m,n))
                noise-=noise.mean(axis=1)[:,None,:]
                noise+=common[:,None,:]/math.sqrt(m)
                candidates['haar{}'.format(m)]=(filt,noise)
            for label,(filters,eta) in candidates.items():
                out=controls(solve(c,a,d,filters,xi,eta,args.epsilon,args.drift))
                # Gaussian control means are zero for sin. For cos, account
                # for any deterministic spectral truncation variance loss.
                candidate_variances=np.sum(filters*filters,axis=(1,2))+np.dot(a,a)
                reference_variance=float(np.sum(L*L)+np.dot(a,a))
                for payoff in ('sin','cos'):
                    raw=out[payoff][0]-ref[payoff][0]
                    difference=out[payoff][1]-ref[payoff][1]
                    if payoff=='cos':
                        difference+=np.exp(-.245*candidate_variances)-math.exp(-.245*reference_variance)
                    sums[label,payoff]+=[difference.sum(),np.dot(difference,difference),
                                        raw.sum(),np.dot(raw,raw)]
            counts+=p
        for (label,payoff),z in sums.items():
            mean=z[0]/counts
            se=math.sqrt(max(0.,(z[1]-z[0]**2/counts)/(counts-1))/counts)
            rawse=math.sqrt(max(0.,(z[3]-z[2]**2/counts)/(counts-1))/counts)
            rows.append(dict(H=args.hurst,n=n,epsilon=args.epsilon,drift=args.drift,
                             haar_cap=args.cap if args.cap is not None else int(math.ceil(math.log2(n))),
                             method=label,payoff=payoff,paths=counts,bias_estimate=mean,
                             standard_error=se,raw_standard_error=rawse,
                             elapsed_seconds=time.perf_counter()-started))
        print('n={} paths={} elapsed={:.2f}s'.format(n,counts,time.perf_counter()-started),flush=True)
        for row in rows[-len(sums):]:print(row,flush=True)
    destination=HERE/'results'/('haar_compression_'+args.tag+'.csv')
    with destination.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (HERE/'results'/('haar_reference_'+args.tag+'.json')).write_text(json.dumps(diagnostics,indent=2)+'\n')
    print('Saved {}'.format(destination),flush=True)


if __name__=='__main__':run()

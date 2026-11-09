"""Matched batched CDQ timings for fixed and per-path residual filters.

All methods use q=1+.8*sin(x), b=-.2*tanh(x), the same generic native leaf,
the same CDQ recursion, and one full nonlinear path per sample. This is a
cost comparison, not an accuracy/variance comparison or a production claim.
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
import platform
import subprocess
import time
import numpy as np
from benchmark_haar_compression import EnergySampler, reference
from verify_randomized_haar_correction import weights

HERE=Path(__file__).resolve().parent
PTR=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')


def native():
    source=HERE/'fft_cost_generic_kernels.c'
    library=HERE/'results'/'fft_cost_generic_kernels.dylib'
    library.parent.mkdir(parents=True,exist_ok=True)
    if not library.exists() or library.stat().st_mtime<source.stat().st_mtime:
        subprocess.run(['clang','-O3','-shared','-fPIC',str(source),'-o',str(library)],check=True)
    lib=ctypes.CDLL(str(library))
    lib.fft_cost_leaf.argtypes=[ctypes.c_int]*6+[ctypes.c_double]*2+[PTR]*7
    lib.fft_cost_leaf.restype=None
    return lib


class GenericCDQ:
    def __init__(self,lib,a,d,filters,leaf=32,epsilon=.8,drift=.2):
        self.lib,self.n,self.leaf=lib,len(a),leaf
        self.a=np.ascontiguousarray(a)
        self.d=np.ascontiguousarray(d)
        self.filters=np.ascontiguousarray(filters)
        assert self.filters.ndim==3 and self.filters.shape[2]==self.n
        self.filter_paths,self.residuals,_=self.filters.shape
        self.epsilon,self.drift=epsilon,drift
        self.ffts={}
        width=self.n
        while width>leaf:
            self.ffts[width]=(
                np.fft.rfft(self.a[:width-1],2*width),
                np.fft.rfft(self.d[:width-1],2*width),
                np.fft.rfft(self.filters[:,:,:width-1],2*width,axis=2).transpose(1,0,2))
            width//=2

    def solve(self,xi,eta,direct=False):
        xi=np.ascontiguousarray(xi);eta=np.ascontiguousarray(eta)
        paths,n=xi.shape
        assert n==self.n and eta.shape==(paths,self.residuals,n)
        assert self.filter_paths in (1,paths)
        x=np.zeros((paths,n+1))
        source=np.zeros((self.residuals+2,paths,n))

        def recurse(left,right):
            width=right-left
            if direct or width<=self.leaf:
                self.lib.fft_cost_leaf(paths,n,self.residuals,left,right,
                    self.filter_paths,self.epsilon,self.drift,
                    self.a,self.d,self.filters,xi,eta,x,source)
                return
            mid=(left+right)//2
            recurse(left,mid)
            transformed=np.fft.rfft(source[:,:,left:mid],2*width,axis=2)
            fa,fd,fr=self.ffts[width]
            combined=transformed[0]*fa+transformed[-1]*fd
            combined+=np.sum(transformed[1:-1]*fr,axis=0)
            values=np.fft.irfft(combined,2*width,axis=1)
            x[:,mid:right]+=values[:,mid-left-1:right-left-1]
            recurse(mid,right)

        recurse(0,n)
        x[:,n]=np.sum(source[0]*self.a[::-1],axis=1)
        x[:,n]+=np.sum(source[-1]*self.d[::-1],axis=1)
        if self.filter_paths==1:
            x[:,n]+=np.einsum('rpn,rn->p',source[1:-1],self.filters[0,:,::-1])
        else:
            x[:,n]+=np.einsum('rpn,prn->p',source[1:-1],self.filters[:,:,::-1])
        return x


def python_direct(a,d,filters,xi,eta,epsilon=.8,drift=.2):
    paths,n=xi.shape
    x=np.zeros((paths,n+1))
    f=np.broadcast_to(filters,(paths,eta.shape[1],n))
    for i in range(n):
        sigma=np.sqrt(1+epsilon*np.sin(x[:,i]))
        noise=a[None,:n-i]*xi[:,i,None]
        noise+=np.sum(f[:,:,:n-i]*eta[:,:,i,None],axis=1)
        x[:,i+1:]+=sigma[:,None]*noise
        x[:,i+1:]+=-drift*np.tanh(x[:,i,None])*d[None,:n-i]
    return x


def validate(lib):
    rng=np.random.RandomState(2026110501)
    n,paths=32,5
    a,s=weights(.1,n);a,s=a[1:],s[1:];d=a/math.sqrt(n)
    cases=[('shared1',s[None,None,:]),
           ('shared2',rng.normal(size=(1,2,n))*.04),
           ('perpath1',rng.normal(size=(paths,1,n))*.04),
           ('perpath4',rng.normal(size=(paths,4,n))*.04)]
    results={}
    for label,filt in cases:
        xi=rng.normal(size=(paths,n));eta=rng.normal(size=(paths,filt.shape[1],n))
        solver=GenericCDQ(lib,a,d,filt,leaf=4)
        fast=solver.solve(xi,eta)
        direct=solver.solve(xi,eta,direct=True)
        independent=python_direct(a,d,filt,xi,eta)
        errors=(float(np.max(abs(fast-direct))),float(np.max(abs(fast-independent))))
        assert max(errors)<2e-12,(label,errors)
        results[label]={'cdq_native_direct_max_error':errors[0],
                        'cdq_python_direct_max_error':errors[1]}
    print('PASS: all generic CDQ cases match native and independent Python direct recursions',flush=True)
    return results


def benchmark(lib,H,n,batch,leaf,repeats,warmup,spectral,max_n,cap_override=None):
    rng=np.random.RandomState(119+n+batch)
    a,s=weights(H,n);a,s=a[1:],s[1:];d=a/math.sqrt(n)
    # Fixed spectral basis computed on max_n once; restrict lag coordinates
    # and adjust h scaling. These remain valid deterministic Gaussian filters.
    fixed_spectral=spectral[:2,:n]*(max_n/n)**H
    rows=[]
    for label,count in [('original2',1),('spectral2',2),('haar1',1),('haar4',4)]:
        random=label.startswith('haar')
        cap=int(math.ceil(math.log2(n)))+int(math.ceil(math.log2(4)/(2*H)))
        if cap_override is not None:cap=cap_override
        start=time.perf_counter()
        if random:
            sampler=EnergySampler(H,n,cap)
            solver=None
        else:
            filt=s[None,None,:] if label=='original2' else fixed_spectral[None,:,:]
            solver=GenericCDQ(lib,a,d,filt,leaf=leaf)
            sampler=None
        initial=time.perf_counter()-start
        times=[];cold=None;checksum=0.
        for repetition in range(warmup+repeats):
            start=time.perf_counter()
            xi=rng.normal(size=(batch,n))
            eta=rng.normal(size=(batch,count,n))
            gaussian=time.perf_counter()-start
            sample_time=fft_setup=0.
            if random:
                start=time.perf_counter()
                filt=sampler.draw(batch,count,rng)
                sample_time=time.perf_counter()-start
                start=time.perf_counter()
                solver=GenericCDQ(lib,a,d,filt,leaf=leaf)
                fft_setup=time.perf_counter()-start
            start=time.perf_counter()
            x=solver.solve(xi,eta)
            solve_time=time.perf_counter()-start
            assert np.isfinite(x).all()
            checksum+=float(np.cos(.7*x[:,-1]).sum())
            values=[gaussian,sample_time,fft_setup,solve_time]
            if cold is None:cold=values
            if repetition>=warmup:times.append(values)
        times=np.array(times)
        med=np.median(times,axis=0)*1000/batch
        total=np.median(times.sum(axis=1))*1000/batch
        rows.append(dict(H=H,n=n,batch=batch,leaf=leaf,method=label,
            gaussian_streams=count+1,residual_streams=count,
            per_path_filters=random,haar_cap=cap if random else '',
            repeats=repeats,warmup_batches=warmup,
            one_time_sampler_or_fft_setup_ms=1000*initial,
            cold_first_batch_sampling_ms=1000*cold[1],
            cold_first_batch_fft_setup_ms=1000*cold[2],
            gaussian_ms_per_path=med[0],sampling_ms_per_path=med[1],
            kernel_fft_setup_ms_per_path=med[2],solver_ms_per_path=med[3],
            total_hot_ms_per_path=total,
            total_hot_min_ms_per_path=float(times.sum(axis=1).min()*1000/batch),
            total_hot_max_ms_per_path=float(times.sum(axis=1).max()*1000/batch),
            sampler_cached_modes=len(sampler.cache) if random else 0,
            sampler_proposals=sampler.proposals if random else 0,
            sampler_accepted=sampler.accepted if random else 0,
            payoff_checksum=checksum))
        print('N={} batch={} {} solver={:.4f}ms/path total={:.4f}ms/path'.format(
            n,batch,label,med[3],total),flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grids',nargs='+',type=int,default=[256,512,1024])
    parser.add_argument('--batches',nargs='+',type=int,default=[32,64])
    parser.add_argument('--hurst',type=float,default=.1)
    parser.add_argument('--leaf',type=int,default=32)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--cap',type=int,default=None)
    parser.add_argument('--tag',default='')
    args=parser.parse_args()
    lib=native();checks=validate(lib)
    max_n=max(args.grids)
    print('Building one covariance reference at N={}'.format(max_n),flush=True)
    _,spectral,diagnostics=reference(args.hurst,max_n)
    print('Spectral reference setup {:.3f}s; retained rank {}'.format(
        diagnostics['preprocessing_seconds'],len(spectral)),flush=True)
    rows=[]
    for n in args.grids:
        assert n&(n-1)==0
        for batch in args.batches:
            rows.extend(benchmark(lib,args.hurst,n,batch,args.leaf,args.repeats,
                                  args.warmup,spectral,max_n,args.cap))
    suffix='_'+args.tag if args.tag else ''
    destination=HERE/'results'/('fft_residual_cost'+suffix+'.csv')
    with destination.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metadata=dict(settings=vars(args),platform=platform.platform(),
        python=platform.python_version(),numpy=np.__version__,validation=checks,
        spectral_reference=diagnostics,
        scope='One full nonlinear path, same sin diffusion and nonlinear drift, same native leaf/CDQ; no accuracy comparison.',
        spectral_convention='Two leading residual eigenfilters from the largest grid; restricted and h-rescaled at smaller grids.',
        setup_convention='Fixed filters reuse FFTs across batches. Random filters are redrawn and transformed each batch; sampler cache persists. Common kernel weights and spectral preprocessing are separate.',
        sampler_note='Proposal counters include unused batched draws; their ratio is not the literal rejection acceptance probability.')
    (HERE/'results'/('fft_residual_cost'+suffix+'_metadata.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    print('Saved {}'.format(destination),flush=True)


if __name__=='__main__':main()

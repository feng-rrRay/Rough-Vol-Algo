#include <math.h>
#include <stddef.h>

/* Shared native direct leaf for every timed method.
   filters: (filter_paths, residual_streams, n), filter_paths is 1 or paths.
   eta: (paths, residual_streams, n); source: (residual_streams+2, paths, n).
   q(x)=1+epsilon*sin(x), b(x)=-drift*tanh(x). */
void fft_cost_leaf(int paths, int n, int residual_streams, int left, int right,
                   int filter_paths, double epsilon, double drift,
                   const double *a, const double *d, const double *filters,
                   const double *xi, const double *eta,
                   double *x, double *source) {
    const size_t channel_stride=(size_t)paths*n;
    for (int p=0;p<paths;++p) {
        const size_t off=(size_t)p*n, out=(size_t)p*(n+1);
        const size_t filter_off=(size_t)(filter_paths==1?0:p)*residual_streams*n;
        for (int j=left;j<right;++j) {
            const double sigma=sqrt(1.0+epsilon*sin(x[out+j]));
            source[off+j]=sigma*xi[off+j];
            source[(residual_streams+1)*channel_stride+off+j]
                =-drift*tanh(x[out+j]);
            for (int r=0;r<residual_streams;++r)
                source[(r+1)*channel_stride+off+j]
                    =sigma*eta[((size_t)p*residual_streams+r)*n+j];
            for (int k=j+1;k<right;++k) {
                const int lag=k-j-1;
                double value=a[lag]*source[off+j]
                    +d[lag]*source[(residual_streams+1)*channel_stride+off+j];
                for (int r=0;r<residual_streams;++r)
                    value+=filters[filter_off+(size_t)r*n+lag]
                        *source[(r+1)*channel_stride+off+j];
                x[out+k]+=value;
            }
        }
    }
}

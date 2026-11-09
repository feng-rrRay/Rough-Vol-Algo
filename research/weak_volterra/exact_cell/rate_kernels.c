#include <math.h>
#include <stddef.h>

/* All normals at source i are independent of its finalized state. */
void rate_leaf(int paths, int n, int channels, int left, int right,
               int model, const double *filters, const double *drift,
               const double *normal, double *state, double *source) {
    size_t stride = (size_t)paths*n;
    for (int p=0; p<paths; ++p) {
        size_t off=(size_t)p*n, xo=(size_t)p*(n+1);
        for (int i=left; i<right; ++i) {
            double x=state[xo+i], q=1., b=0.;
            if (model==1 || model==2) q=1.+.8*sin(x);
            if (model==2 || model==5) b=.7*(.2-x);
            if (model==3) { q=1.+.6*tanh(x)+.2*sin(2*x); b=-.4*x+.2*sin(x); }
            if (model==4) q=1.+.5*cos(x);
            double s=sqrt(q);
            for (int r=0;r<channels;++r)
                source[(size_t)r*stride+off+i]=s*normal[(size_t)r*stride+off+i];
            source[(size_t)channels*stride+off+i]=b;
            for (int k=i+1;k<right;++k) {
                int lag=k-i-1;
                double value=b*drift[lag];
                for (int r=0;r<channels;++r)
                    value+=filters[(size_t)r*n+lag]*source[(size_t)r*stride+off+i];
                state[xo+k]+=value;
            }
        }
    }
}

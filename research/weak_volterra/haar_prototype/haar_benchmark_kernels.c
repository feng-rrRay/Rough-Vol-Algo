#include <math.h>
#include <stdlib.h>
#include <stddef.h>

/* Direct coupled reference solver, not the FFT timing implementation.
   Residual filters can differ by path. All source normals are independent;
   sign reversal affects only the residual channels. q=1+eps*sin(x). */
void haar_reference_batch(int paths, int n, int channels, double eps, double drift,
                          const double *a, const double *d, const double *filters,
                          const double *xi, const double *eta, double *out) {
    double *work=(double*)calloc((size_t)5*(n+1),sizeof(double));
    for(int p=0;p<paths;++p) {
        for(int j=0;j<5*(n+1);++j) work[j]=0.;
        double *xp=work, *xm=work+n+1, *gp=work+2*(n+1);
        double *gm=work+3*(n+1), *noise=work+4*(n+1);
        const double *wp=filters+(size_t)p*channels*n;
        const double *ep=eta+(size_t)p*channels*n;
        for(int i=0;i<n;++i) {
            int remaining=n-i;
            for(int l=0;l<remaining;++l) noise[l]=0.;
            for(int r=0;r<channels;++r) {
                const double z=ep[(size_t)r*n+i];
                const double *wr=wp+(size_t)r*n;
                for(int l=0;l<remaining;++l) noise[l]+=wr[l]*z;
            }
            double sp=sqrt(1.+eps*sin(xp[i]));
            double sm=sqrt(1.+eps*sin(xm[i]));
            double bp=-drift*tanh(xp[i]), bm=-drift*tanh(xm[i]);
            const double z=xi[(size_t)p*n+i];
            for(int l=0;l<remaining;++l) {
                int k=i+l+1;
                double zp=a[l]*z+noise[l], zm=a[l]*z-noise[l];
                xp[k]+=sp*zp+d[l]*bp;
                xm[k]+=sm*zm+d[l]*bm;
                gp[k]+=zp; gm[k]+=zm;
            }
        }
        out[4*p]=xp[n];out[4*p+1]=xm[n];
        out[4*p+2]=gp[n];out[4*p+3]=gm[n];
    }
    free(work);
}

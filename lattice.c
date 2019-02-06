//-------------------------------
//-- Date: 3 Oct 2016 ----------
//-- Engineer: Dmitry Gulevich --
//-----------------------------------------------------------------------------------
// To set OpenMP environment variable:
// $ export OMP_NUM_THREADS=6
// $ export OMP_NUM_THREADS=8
// $ echo $OMP_NUM_THREADS
//
#include <stdio.h>
#include <stdlib.h> // malloc, rand
#include <time.h> // srand(time(NULL));
#include <complex.h> // complex numbers
#include <math.h> // sqrt
#include <string.h> // memcpy
//#include <stdbool.h> // boolean datatype
#include <time.h> // timing
#include <omp.h>
//#include <ctype.h> // character handling for getopt routine
//#include <unistd.h> // getopt routine
#include <fftw3.h> // should stay after complex.h for type converion
//#include <lapacke.h>

#define LIEB
#define NPKY 6
#define PARABOLIC

int M; // global

inline int mod(int m) {
    return ((m+M)%M);
}


inline int row(int i, int j) {
    return M*i+j;
}


void lattice(double complex *psi_py, double complex *fpsi_array, int Ncells, int Mhalf, double tstart, double dt, int Nframes, int countout, double complex *U)
{
srand(time(NULL));
printf("############################################################\n");
printf("# Running C Extension Module\n");
printf("############################################################\n");

// --- Initialization ---
int P=Mhalf*Ncells;
M=2*P;
int M2=M*M, i, j, m, n, count, frame;
double invM2=1./M2;
double Lx=Ncells; // Lieb
double Ly=Ncells; // Lieb
double dx=Lx/M;
double dy=Ly/M;
double mk, nk, k2;
double dkx=2.*M_PI/Lx;
double dky=2.*M_PI/Ly;
double dkx2=dkx*dkx;
double dky2=dky*dky;
double complex cfactor;

double Ec, Ex, Elp, OmegaRabi;
double ky[NPKY]={0., 0.15e6, 0.44e6, 0.74e6, 1.03e6, 1.31e6}; // m^-1
double d=3.0e-6;
int Pky[NPKY];
for(i=0;i<NPKY;i++)
    Pky[i]=lround(d*ky[i]/dky);

fftw_complex *psi, *fpsi;
psi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * M2);
fpsi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * M2);
fftw_plan fftpsi, ifftpsi; // Warning: plans should be created before initialization of the input
fftpsi = fftw_plan_dft_2d(M, M, psi, fpsi, FFTW_FORWARD, FFTW_MEASURE); 
ifftpsi = fftw_plan_dft_2d(M, M, fpsi, psi, FFTW_BACKWARD, FFTW_MEASURE);
memcpy(psi, psi_py, M2*sizeof(double complex));

// --- Display info, tests ---

// --- Calculation ---
double t0=omp_get_wtime();
clock_t c0 = clock();

double t=0.;
for(count=0,frame=0;frame<Nframes;count++) {

    // half-step
    for(i=0;i<M;i++)
        for(j=0;j<M;j++)
            psi[row(i,j)]=psi[row(i,j)]*cexp(-0.5*dt*I*(U[row(i,j)]));

    // full FFT step
    fftw_execute(fftpsi); // psi -> fpsi
    for(m=0;m<M;m++)
        for(n=0;n<M;n++) {
            mk=(m+P-1)%M-P+1; // M-even
            nk=(n+P-1)%M-P+1; // M-even
            k2=(mk*mk*dkx2+nk*nk*dky2);

#ifdef PARABOLIC
            fpsi[m*M+n]*=cexp(-dt*I*k2);
#else
            Ec=k2;
            Ex=30+0.1*k2;
            OmegaRabi=30;
            Elp=0.5*(Ec+Ex-sqrt((Ec-Ex)*(Ec-Ex)+OmegaRabi*OmegaRabi));
            fpsi[m*M+n]*=cexp(-dt*I*Elp);
#endif

        }

/*    for(m=0;m<M;m++)
        fpsi[m*M+P]=0;
    for(n=0;n<M;n++)
        fpsi[P*M+n]=0;*/


    fftw_execute(ifftpsi); // fpsi -> psi*M2

    // half-step
    for(i=0;i<M;i++)
        for(j=0;j<M;j++)
            psi[row(i,j)]=invM2*psi[row(i,j)]*cexp(-0.5*dt*I*(U[row(i,j)]));

    t += dt;

    if(t>=tstart && count%countout==0) {

        printf("# frame=%d, t=%f\n",frame,t);
        fftw_execute(fftpsi); // psi -> fpsi
        for(i=0;i<NPKY;i++)
            memcpy(fpsi_array+frame*NPKY*M+M*i, fpsi+M*Pky[i], M*sizeof(double complex));
        frame++;
    }

            
}

double t1=omp_get_wtime();
clock_t c1 = clock();
printf ("#\n");
printf ("# CPU time:        %f\n", (float) (c1 - c0)/CLOCKS_PER_SEC);
printf ("# Wall clock time: %f\n", t1-t0);

// --- Return ---
memcpy(psi_py, psi, M2*sizeof(double complex));

// --- Clear ---
fftw_destroy_plan(fftpsi);
fftw_destroy_plan(ifftpsi);
fftw_free(psi);
fftw_free(fpsi);  
}


/*
int main() 
{
int P=5;
int MM2=2*(2*P+1)*(2*P+1);
int Niter=1;
double mu=1.;

double complex *psi_py = malloc( (2*P+1)*(2*P+1) * sizeof(double complex) );
double complex *rhs = malloc(  MM2* sizeof(double complex) );
double complex *matrix = malloc( MM2*MM2 * sizeof(double complex) );
solid(psi_py, P, 10., mu, matrix, Niter, rhs);
return 0;
}
*/

// For Reference:
/*    
//memcpy(fpsinew, fpsi, MM2*sizeof(double complex));
//printf("# test %f%+fi\n", creal(psi_py[0]),cimag(psi_py[0]));
//double complex *psi = malloc( M2 * sizeof(double complex) );
*/

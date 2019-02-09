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

int My; // global

inline int row(int i, int j) {
    return My*i+j;
}



void evolve(double complex *psi_py, double complex *fpsi_array, int Ncells, int Mhalf, double dt, int Nframes, int countout, double complex *U, double *ky, int Nky)
{
srand(time(NULL));
printf("############################################################\n");
printf("# Running C Extension Module\n");
printf("############################################################\n");

// --- Initialization ---
int P=Mhalf*Ncells;
int M=2*P;
My=M;
int M2=M*M, i, j, m, n, count, frame;
double invM2=1./M2;
double mk, nk, k2;
double dkx=2.*M_PI/Ncells;
double dky=2.*M_PI/Ncells;
double dkx2=dkx*dkx;
double dky2=dky*dky;

int *Pky = malloc(Nky*sizeof(int));

for(i=0;i<Nky;i++)
    Pky[i]=lround(ky[i]/dky);

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
            fpsi[m*M+n]*=cexp(-dt*I*k2);
        }

    fftw_execute(ifftpsi); // fpsi -> psi*M2

    // half-step
    for(i=0;i<M;i++)
        for(j=0;j<M;j++)
            psi[row(i,j)]=invM2*psi[row(i,j)]*cexp(-0.5*dt*I*(U[row(i,j)]));

    t += dt;

    if(count%countout==0) {

        printf("# frame=%d, t=%f\n",frame,t);
        fftw_execute(fftpsi); // psi -> fpsi
        for(i=0;i<Nky;i++)
            memcpy(fpsi_array+frame*Nky*M+M*i, fpsi+M*Pky[i], M*sizeof(double complex));
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



void lattice(double complex *psi1_py, double complex *psi2_py, double complex *psi1data, double complex *psi2data, double Lcell, 
    int Nxcells, int Nycells, int Mhalf, double dt, int Nframes, int countout, double complex *U, double *nR, double alpha, double beta, double Omega,
    double A0, double omegadrive, double R, double xrel, double yrel)
{
//double gr=2.; // debug
//double Rs=100.0; // debug
//double P0=10.0; // debug
double gr=0.; // debug
double Rs=0.; // debug
double P0=0.; // debug

        
srand(time(NULL));
printf("############################################################\n");
printf("# Running C Extension Module\n");
printf("############################################################\n");

// --- Initialization ---
int Px=Mhalf*Nxcells;
int Py=Mhalf*Nycells;
int Mx=2*Px;
My=2*Py;
int M2=Mx*My, i, j, m, n, mk, nk, count, frame;
double invM2=1./M2;

double Lx=Lcell*Nxcells;
double Ly=Lcell*Nycells;
double kx, ky, k2;
double dkx=2.*M_PI/Lx;
double dky=2.*M_PI/Ly;
double dkx2=dkx*dkx;
double dky2=dky*dky;

double complex cfactor, offdiag, fpsi1new, fpsi2new;
double cosfactor, sinfactor, rho1, rho2;

fftw_complex *psi1, *fpsi1;
fftw_complex *psi2, *fpsi2;
psi1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * M2);
fpsi1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * M2);
psi2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * M2);
fpsi2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * M2);
fftw_plan fftpsi1, fftpsi2, ifftpsi1, ifftpsi2; // Warning: plans should be created before initialization of the input
fftpsi1 = fftw_plan_dft_2d(Mx, My, psi1, fpsi1, FFTW_FORWARD, FFTW_MEASURE);
fftpsi2 = fftw_plan_dft_2d(Mx, My, psi2, fpsi2, FFTW_FORWARD, FFTW_MEASURE);  
ifftpsi1 = fftw_plan_dft_2d(Mx, My, fpsi1, psi1, FFTW_BACKWARD, FFTW_MEASURE);
ifftpsi2 = fftw_plan_dft_2d(Mx, My, fpsi2, psi2, FFTW_BACKWARD, FFTW_MEASURE);
memcpy(psi1, psi1_py, M2*sizeof(double complex));
memcpy(psi2, psi2_py, M2*sizeof(double complex));

// --- Display info, tests ---

printf("psi1_py[0]=%f%+fi\n",creal(psi1_py[0]),cimag(psi1_py[0]));
printf("psi1[0]=%f%+fi\n",creal(psi1[0]),cimag(psi1[0]));
printf("Mx=%d, My=%d, Lcell=%.1f, alpha=%.2f, beta=%.2f\n",Mx,My,Lcell,alpha,beta);

double complex cfactor1, cfactor2;
double dr = Lcell/(2.*Mhalf);
 
double *A = malloc( M2 * sizeof(double) );
//int i0 = 22 + (2*Mhalf)*7;
//int j0 = 28 + (2*Mhalf)*0;
//int i0 = 11 + (2*Mhalf)*7;
//int j0 = 27 + (2*Mhalf)*0;

int i0 = Px + round(xrel*2*Mhalf);
int j0 = Py + round(yrel*2*Mhalf);

//printf("# i0: %d, j0: %d\n",i0,j0);

double dr2=dr*dr;

for(i=0;i<Mx;i++)
    for(j=0;j<My;j++)
        A[row(i,j)]=A0*exp(-((i-i0)*(i-i0)+(j-j0)*(j-j0))*dr2/(R*R));


// --- Calculation ---
double t0=omp_get_wtime();
clock_t c0 = clock();

double t=0.;
for(count=0,frame=0;frame<Nframes;count++) {

    // half-step
    for(i=0;i<Mx;i++)
        for(j=0;j<My;j++) {
            cfactor1=A[row(i,j)]/(omegadrive-U[row(i,j)]-Omega);
            cfactor2=A[row(i,j)]/(omegadrive-U[row(i,j)]+Omega);                            

            rho1=psi1[row(i,j)]*conj(psi1[row(i,j)]);
            rho2=psi2[row(i,j)]*conj(psi2[row(i,j)]);

//            psi1[row(i,j)]=psi1[row(i,j)]*cexp(-0.5*dt*I*(U[row(i,j)]+Omega)) + cfactor1*cexp(-I*omegadrive*t)*( cexp(-0.5*dt*I*omegadrive)-cexp(-0.5*dt*I*(U[row(i,j)]+Omega)) );
//            psi2[row(i,j)]=psi2[row(i,j)]*cexp(-0.5*dt*I*(U[row(i,j)]-Omega)) + cfactor2*cexp(-I*omegadrive*t)*( cexp(-0.5*dt*I*omegadrive)-cexp(-0.5*dt*I*(U[row(i,j)]-Omega)) );

            psi1[row(i,j)]=psi1[row(i,j)]*cexp(-0.5*dt*I*(U[row(i,j)]+rho1+alpha*rho2+Omega)) + cfactor1*cexp(-I*omegadrive*t)*( cexp(-0.5*dt*I*omegadrive)-cexp(-0.5*dt*I*(U[row(i,j)]+rho1+alpha*rho2+Omega)) );
            psi2[row(i,j)]=psi2[row(i,j)]*cexp(-0.5*dt*I*(U[row(i,j)]+rho2+alpha*rho1-Omega)) + cfactor2*cexp(-I*omegadrive*t)*( cexp(-0.5*dt*I*omegadrive)-cexp(-0.5*dt*I*(U[row(i,j)]+rho2+alpha*rho1-Omega)) );

//            psi1[row(i,j)]=psi1[row(i,j)]*cexp(-0.5*dt*I*(U[row(i,j)]+Omega+rho1+alpha*rho2+gr*nR[row(i,j)]+0.5*I*Rs*nR[row(i,j)])) + cfactor1*cexp(-I*omegadrive*t)*( cexp(-0.5*dt*I*omegadrive)-cexp(-0.5*dt*I*(U[row(i,j)]+Omega)) );
//            psi2[row(i,j)]=psi2[row(i,j)]*cexp(-0.5*dt*I*(U[row(i,j)]-Omega+rho2+alpha*rho1+gr*nR[row(i,j)]+0.5*I*Rs*nR[row(i,j)])) + cfactor2*cexp(-I*omegadrive*t)*( cexp(-0.5*dt*I*omegadrive)-cexp(-0.5*dt*I*(U[row(i,j)]-Omega)) );
//            printf("i=%d, j=%d, rho1=%f, rho2=%f, U=%f\n",i,j,rho1,rho2,U[row(i,j)]);
        }

    // full FFT step
    fftw_execute(fftpsi1); // psi1 -> fpsi1
    fftw_execute(fftpsi2); // psi2 -> fpsi2
    for(m=0;m<Mx;m++)
        for(n=0;n<My;n++) {
            mk=(m+Px-1)%Mx-Px+1; // Mx - even
            nk=(n+Py-1)%My-Py+1; // My - even
            kx=mk*dkx;
            ky=nk*dky;
            k2=kx*kx+ky*ky;
            cfactor=cexp(-dt*I*k2);
            cosfactor=cos(dt*k2*beta);
            sinfactor=sin(dt*k2*beta);

            if(mk!=0 || nk!=0) { /// DEBUG
                offdiag=(kx-I*ky)*(kx-I*ky)/k2;
                fpsi1new=cfactor*(cosfactor*fpsi1[m*My+n] + I*sinfactor*fpsi2[m*My+n]*offdiag);
                fpsi2new=cfactor*(I*sinfactor*fpsi1[m*My+n]*conj(offdiag) + cosfactor*fpsi2[m*My+n]);
                fpsi1[m*My+n]=fpsi1new;
                fpsi2[m*My+n]=fpsi2new;
            }
            
        }
    fftw_execute(ifftpsi1); // fpsi1 -> psi1*M2
    fftw_execute(ifftpsi2); // fpsi2 -> psi2*M2

    // half-step
    for(i=0;i<Mx;i++)
        for(j=0;j<My;j++) {
            psi1[row(i,j)]*=invM2;
            psi2[row(i,j)]*=invM2;
            cfactor1=A[row(i,j)]/(omegadrive-U[row(i,j)]-Omega);
            cfactor2=A[row(i,j)]/(omegadrive-U[row(i,j)]+Omega);                            

            rho1=psi1[row(i,j)]*conj(psi1[row(i,j)]);
            rho2=psi2[row(i,j)]*conj(psi2[row(i,j)]);

//            psi1[row(i,j)]=psi1[row(i,j)]*cexp(-0.5*dt*I*(U[row(i,j)]+Omega)) + cfactor1*cexp(-I*omegadrive*t)*( cexp(-0.5*dt*I*omegadrive)-cexp(-0.5*dt*I*(U[row(i,j)]+Omega)) );
//            psi2[row(i,j)]=psi2[row(i,j)]*cexp(-0.5*dt*I*(U[row(i,j)]-Omega)) + cfactor2*cexp(-I*omegadrive*t)*( cexp(-0.5*dt*I*omegadrive)-cexp(-0.5*dt*I*(U[row(i,j)]-Omega)) );

            psi1[row(i,j)]=psi1[row(i,j)]*cexp(-0.5*dt*I*(U[row(i,j)]+rho1+alpha*rho2+Omega)) + cfactor1*cexp(-I*omegadrive*t)*( cexp(-0.5*dt*I*omegadrive)-cexp(-0.5*dt*I*(U[row(i,j)]+rho1+alpha*rho2+Omega)) );
            psi2[row(i,j)]=psi2[row(i,j)]*cexp(-0.5*dt*I*(U[row(i,j)]+rho2+alpha*rho1-Omega)) + cfactor2*cexp(-I*omegadrive*t)*( cexp(-0.5*dt*I*omegadrive)-cexp(-0.5*dt*I*(U[row(i,j)]+rho2+alpha*rho1-Omega)) );

//            psi1[row(i,j)]=psi1[row(i,j)]*cexp(-0.5*dt*I*(U[row(i,j)]+Omega+rho1+alpha*rho2+gr*nR[row(i,j)]+0.5*I*Rs*nR[row(i,j)])) + cfactor1*cexp(-I*omegadrive*t)*( cexp(-0.5*dt*I*omegadrive)-cexp(-0.5*dt*I*(U[row(i,j)]+Omega)) );
//            psi2[row(i,j)]=psi2[row(i,j)]*cexp(-0.5*dt*I*(U[row(i,j)]-Omega+rho2+alpha*rho1+gr*nR[row(i,j)]+0.5*I*Rs*nR[row(i,j)])) + cfactor2*cexp(-I*omegadrive*t)*( cexp(-0.5*dt*I*omegadrive)-cexp(-0.5*dt*I*(U[row(i,j)]-Omega)) );
        }

	// reservoir polaritons
   /* for(i=0;i<Mx;i++)
        for(j=0;j<My;j++) {
			double gammar = 3.*cimag(U[row(i,j)]);
			nR[row(i,j)] = (nR[row(i,j)]+P0*dt)/(1. + gammar*dt + Rs*(rho1+rho2)*dt);
			}*/
			

    t += dt;

    if(count%countout==0) {
        printf("# frame=%d, t=%f\n",frame,t);
        memcpy(psi1data + frame*M2, psi1, M2*sizeof(double complex));
        memcpy(psi2data + frame*M2, psi2, M2*sizeof(double complex));
        frame++;
    }

}

double t1=omp_get_wtime();
clock_t c1 = clock();
printf ("#\n");
printf ("# CPU time:        %f\n", (float) (c1 - c0)/CLOCKS_PER_SEC);
printf ("# Wall clock time: %f\n", t1-t0);

// --- Return ---
memcpy(psi1_py, psi1, M2*sizeof(double complex));
memcpy(psi2_py, psi2, M2*sizeof(double complex));

// --- Clear ---
fftw_destroy_plan(fftpsi1);
fftw_destroy_plan(fftpsi2);
fftw_destroy_plan(ifftpsi1);
fftw_destroy_plan(ifftpsi2);
fftw_free(psi1);
fftw_free(psi2);
fftw_free(fpsi1);
fftw_free(fpsi2);  
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


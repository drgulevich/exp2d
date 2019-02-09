import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import dia_matrix
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation

### Define a custom cmap
cdict = {'red':   ((0.0, 1.0, 1.0),
                   (0.3, 0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 1.0, 1.0),
                   (0.3, 0.0, 0.0),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.8, 0.8)),

         'blue':  ((0.0, 1.0, 1.0),
                   (0.3, 0.5, 0.5),
                   (0.75, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }
mycmap = LinearSegmentedColormap('mycmap', cdict)

### Square lattice
# Assume square MxM cell of unit length 
# M: number of nodes along one direction
# R: radius of the pillar (relative to the cell length)
# Returns: complex potential in the range [0,1]
def U_Square(M, R, Uin, Uout):
    assert M%2==0
    Mhalf=round(M/2)
    dr=1./M

    Rbound=(R/dr)**2
    U=Uout*np.ones((M,M),dtype=np.complex128)
    for i in range(M):
        for j in range(M):
            for cx,cy in [(Mhalf,Mhalf)]:
                di=i-cx
                dj=j-cy
                if(di*di+dj*dj <= Rbound):
                    U[i,j] = Uin

    return U

### Lieb lattice
# Assume square MxM cell of unit length 
# M: number of nodes along one direction
# R: radius of the pillar (relative to the cell length)
# Returns: complex potential in the range [0,1]
def U_Lieb(M, R, Uin, Uout):
    assert M%2==0
    Mhalf=round(M/2)
    dr=1./M

    Rbound=(R/dr)**2
    U=Uout*np.ones((M,M),dtype=np.complex128)
    for i in range(M):
        for j in range(M):
            for cx,cy in [(0,Mhalf),(Mhalf,0),(Mhalf,Mhalf),(Mhalf,M),(M,Mhalf)]:
                di=i-cx
                dj=j-cy
                if(di*di+dj*dj <= Rbound):
                    U[i,j] = Uin

    return U


### Calculate bands at fixed ky in the U.real
# M: number of nodes along one direction
# U: complex potential (only real part is used)
# kxrange: range of kx
# ky: fixed value of ky
# Nbands: number of bands
def bands_at_ky(M, U, kxrange, ky, Nbands):

    dr=1./M
    dr2=dr*dr

    ### Construct matrix
    Imat=np.ones(M*M,dtype=np.complex128)
    Itop=np.ones(M,dtype=np.complex128)
    Itop[0]=0.
    Itopmat=np.array([Itop]*M).ravel()
    Ibot=np.ones(M,dtype=np.complex128)
    Ibot[-1]=0.
    Ibotmat=np.array([Ibot]*M).ravel()

    evlist=[]
    for kx in kxrange:
    
        ekx=np.exp(-1j*kx)
        eky=np.exp(-1j*ky)
        ekytop=np.zeros(M,dtype=np.complex128)
        ekybot=np.zeros(M,dtype=np.complex128)
        ekytop[-1]=-eky
        ekybot[0]=-np.conjugate(eky)
        topmat=np.array([ekytop]*M).ravel()
        botmat=np.array([ekybot]*M).ravel()

        data=[botmat,-Imat*np.conjugate(ekx),-Imat,-Ibotmat,4.*Imat + dr2*U.real.ravel(),-Itopmat,-Imat,-Imat*ekx,topmat]
        offset=[-(M-1),-M*(M-1),-M,-1,0,1,M,M*(M-1),M-1]
        dmatrix=dia_matrix((data, offset), shape=(M*M, M*M))
        spmatrix=csc_matrix(dmatrix)
        evals=np.sort(linalg.eigsh(spmatrix, return_eigenvectors=False, k=Nbands, sigma=0))/dr2
        evlist.append(evals)

    return np.array(evlist)


### Calculate evals,evecs for a range of kx,ky values in Re[U(x,y)]
# M: number of nodes along one direction
# U: complex potential (only real part is used)
# kxrange: range of kx values
# kyrange: range of ky values
# Nbands: number of bands
def eigsystem(M, U, kxrange, kyrange, Nbands):

    dr=1./M
    dr2=dr*dr

    ### Constructing matrix
    Imat=np.ones(M*M,dtype=np.complex128)
    Itop=np.ones(M,dtype=np.complex128)
    Itop[0]=0.
    Itopmat=np.array([Itop]*M).ravel()
    Ibot=np.ones(M,dtype=np.complex128)
    Ibot[-1]=0.
    Ibotmat=np.array([Ibot]*M).ravel()
    offset=[-(M-1),-M*(M-1),-M,-1,0,1,M,M*(M-1),M-1]

    evalslist=[]
    evecslist=[]

    for ky in kyrange: 

        eky=np.exp(-1j*ky)
        ekytop=np.zeros(M,dtype=np.complex128)
        ekybot=np.zeros(M,dtype=np.complex128)
        ekytop[-1]=-eky
        ekybot[0]=-np.conjugate(eky)
        topmat=np.array([ekytop]*M).ravel()
        botmat=np.array([ekybot]*M).ravel()
    
        for kx in kxrange:
    
            ekx=np.exp(-1j*kx)
            data=[botmat,-Imat*np.conjugate(ekx),-Imat,-Ibotmat,4.*Imat + dr2*U.real.ravel(),-Itopmat,-Imat,
                  -Imat*ekx,topmat]
            dmatrix=dia_matrix((data, offset), shape=(M*M, M*M))
            spmatrix=csc_matrix(dmatrix)
            evalsunsorted,Tevecs=linalg.eigsh(spmatrix, return_eigenvectors=True, k=Nbands, sigma=0)
            evalsunsorted/=dr2
            evecsunsorted=Tevecs.T
        
            inds=np.argsort(evalsunsorted)
            evals=evalsunsorted[inds]
            evecs=evecsunsorted[inds]
    
            evalslist.append(evals.ravel())
            evecslist.append(evecs.ravel())

    evalsarr=np.array(evalslist).reshape(kyrange.size,kxrange.size,Nbands)        
    evecsarr=np.array(evecslist).reshape(kyrange.size,kxrange.size,Nbands,M*M)

    return evalsarr,evecsarr


### Display wavefunction Psi in real space
def cpsi(Psi,interpolation=None):
    data=(Psi*np.conjugate(Psi)).real
    img=plt.imshow(data.T, interpolation=interpolation,cmap = plt.cm.Blues_r, origin='lower')
    plt.colorbar(img)
    plt.show()

### Display two wavefunctions in real space
def cpsi12(Psi1,Psi2,interpolation=None):
    fig,ax=plt.subplots(1,2,figsize=(12,5))

    data1=(Psi1*np.conjugate(Psi1)).real
    img1=ax[0].imshow(data1.T, interpolation=interpolation,cmap = plt.cm.Blues_r, origin='lower')
    plt.colorbar(img1,ax=ax[0])

    data2=(Psi2*np.conjugate(Psi2)).real
    img2=ax[1].imshow(data2.T, interpolation=interpolation,cmap = plt.cm.Reds_r, origin='lower')
    plt.colorbar(img2,ax=ax[1])

    plt.show()

### Display wavefunction Psi in real space
### Animation example: http://matplotlib.org/examples/animation/dynamic_image.html
def vipsi(Psidata,interpolation=None):
    rhomax=np.max(abs(Psidata*np.conjugate(Psidata)))
    fig, ax = plt.subplots(figsize=(18,6))
    plt.axis('off')
    rhodata=(Psidata[0,:,:]*np.conjugate(Psidata[0,:,:])).real
    im=plt.imshow(rhodata.T, interpolation=interpolation, cmap = plt.cm.Blues_r, origin='lower',vmin=0,vmax=rhomax);
    def updatefig(frame):
        rhodata=(Psidata[frame,:,:]*np.conjugate(Psidata[frame,:,:])).real
        im.set_array(rhodata.T)
        return im,
    def init():
        return updatefig(0)
    anim = animation.FuncAnimation(fig, updatefig, np.arange(Psidata.shape[0]), init_func=init, interval=100, blit=True, repeat=False)
    plt.close(fig)
    return anim


### Display two wavefunctions Psi1, Psi2 in real space
### Animation example: http://matplotlib.org/examples/animation/dynamic_image.html
def vipsi12(Psi1data,Psi2data,interpolation=None):
    rho1max=np.max(abs(Psi1data*np.conjugate(Psi1data)))
    rho2max=np.max(abs(Psi2data*np.conjugate(Psi2data)))
    fig, ax = plt.subplots(1,2,figsize=(12,5))
    ax[0].axis('off')
    ax[1].axis('off')
    rho1data=(Psi1data[0,:,:]*np.conjugate(Psi1data[0,:,:])).real
    rho2data=(Psi2data[0,:,:]*np.conjugate(Psi2data[0,:,:])).real
    im1=ax[0].imshow(rho1data.T, interpolation=interpolation, cmap = plt.cm.Blues_r, origin='lower',vmin=0,vmax=rho1max);
    im2=ax[1].imshow(rho2data.T, interpolation=interpolation, cmap = plt.cm.Reds_r, origin='lower',vmin=0,vmax=rho2max);
    def updatefig(frame):
        rho1data=(Psi1data[frame,:,:]*np.conjugate(Psi1data[frame,:,:])).real
        rho2data=(Psi2data[frame,:,:]*np.conjugate(Psi2data[frame,:,:])).real
        im1.set_array(rho1data.T)
        im2.set_array(rho2data.T)
        return im1,im2
    def init():
        return updatefig(0)
    anim = animation.FuncAnimation(fig, updatefig, np.arange(Psi1data.shape[0]), init_func=init, interval=100, blit=True, repeat=False)
    plt.close(fig)
    return anim

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import dia_matrix
from scipy.sparse import linalg
import time

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

#NBzones=2
#Nvals=30
#Nkpoints=70
def calc_bands(M, U, ky, NBzones, Nvals, Nkpoints):

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

    kxrange=np.linspace(-NBzones*np.pi, NBzones*np.pi,Nkpoints)

    start_time = time.time()
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

        data=[botmat,-Imat*np.conjugate(ekx),-Imat,-Ibotmat,4.*Imat + dr2*U.ravel(),-Itopmat,-Imat,-Imat*ekx,topmat]
        offset=[-(M-1),-M*(M-1),-M,-1,0,1,M,M*(M-1),M-1]
        dmatrix=dia_matrix((data, offset), shape=(M*M, M*M))
        spmatrix=csc_matrix(dmatrix)
        evals=np.sort(linalg.eigsh(spmatrix, return_eigenvectors=False, k=Nvals, sigma=0))/dr2
        evlist.append(evals)

    evlist=np.array(evlist)

#np.savez('U'+str(Ulist[nu])+'_Pky_'+str(Pky),kxrange=kxrange,evlist=evlist)
#    print('#')
#    print("# Timing: %.f seconds" % (time.time()-start_time))
#    print('#')

    return kxrange,evlist


### Source: lieb-vectors.py
#    kx=math.pi
#    ky=math.pi
#    Nvals=30
def calc_vectors(M, U, kx, ky, Nvals):

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

    start_time = time.time()

    ekx=np.exp(-1j*kx)
    eky=np.exp(-1j*ky)
    ekytop=np.zeros(M,dtype=np.complex128)
    ekybot=np.zeros(M,dtype=np.complex128)
    ekytop[-1]=-eky
    ekybot[0]=-np.conjugate(eky)
    topmat=np.array([ekytop]*M).ravel()
    botmat=np.array([ekybot]*M).ravel()
    
    data=[botmat,-Imat*np.conjugate(ekx),-Imat,-Ibotmat,4.*Imat + dr2*U.ravel(),-Itopmat,-Imat,-Imat*ekx,topmat]
    offset=[-(M-1),-M*(M-1),-M,-1,0,1,M,M*(M-1),M-1]
    dmatrix=dia_matrix((data, offset), shape=(M*M, M*M))
    spmatrix=csc_matrix(dmatrix)
    #evals=sort(eigsh(spmatrix, return_eigenvectors=False, k=Nvals, sigma=0))/dr2
    evalsunsorted,Tevecs=linalg.eigsh(spmatrix, return_eigenvectors=True, k=Nvals, sigma=0)
    evalsunsorted/=dr2
    evecsunsorted=Tevecs.T
    
    inds=np.argsort(evalsunsorted)
    evals=evalsunsorted[inds]
    evecs=evecsunsorted[inds]
    
    # np.savez('temp',temp=temp)
    
    #print('#')
    #print("# Timing: %.f seconds" % (time.time()-start_time))
    #print('#')
    
    #els=np.where((ens>2) & (ens<2.5))
    #print('Selected values (sparse):')
    #print(ens[els])

    return evals,evecs



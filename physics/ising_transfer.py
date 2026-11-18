# physics/ising_transfer.py
import numpy as np

def logZ_transfer(N,beta):
    """
    Periodic uniform-J square Ising: trace of the Nth row-transfer power.
    """
    if N<3:
        raise ValueError('This transfer convention requires N>=3')
    row_codes=np.arange(1<<N,dtype=np.uint64)
    rows=((row_codes[:,None]>>np.arange(N,dtype=np.uint64))&1).astype(float)*2-1
    horizontal=(rows*np.roll(rows,-1,axis=1)).sum(1)
    logW=beta*(rows@rows.T+(horizontal[:,None]+horizontal[None,:])/2)
    shift=float(logW.max())
    W=np.exp(logW-shift)
    # N=5 here; all entries are nonnegative, so no eigenvalue cancellation.
    power=np.linalg.matrix_power(W,N)
    return float(N*shift+np.log(np.trace(power)))


def validate_transfer():
    N=3
    D=N*N
    codes=np.arange(1<<D,dtype=np.uint64)
    X=(((codes[:,None]>>np.arange(D,dtype=np.uint64))&1).astype(float)*2-1).reshape(-1,N,N)
    E=-(X*np.roll(X,-1,axis=1)+X*np.roll(X,-1,axis=2)).sum(axis=(1,2))

    errors=[]
    for T in (1.5,2.27,3.2):
        brute=float(np.logaddexp.reduce(-E/T))
        errors.append(abs(brute-logZ_transfer(N,1/T)))

    if max(errors)>1e-9:
        raise RuntimeError('Transfer-matrix vs brute-force validation failed')
    
    return max(errors)

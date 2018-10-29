import numpy as np
import matplotlib.pyplot as plt
from fci import FCI
import warnings


def density_matr(C, F = 2):
    # D = np.einsum('ij,ik->jk',C[:F,:].conj() ,C[:F,:])
    D = (C.conj().T[:,:F]) @ (C[:F, :]) # slightly faster (1.6 µs < 2.5 µs)
    return D

def solve_hfock(L, hf, tol = 1e-6, max_itr = 20):
    
    hfocks = []
    energies = []
    
    F = hf.Z
    C = np.eye(L)


    hfock_diff = np.inf
    itr = 1
    
    D = density_matr(C, F=F)
    energies.append(calc_hf_energy(D, hf))
    print('Reference energy: ', energies[0])

    print('{:<12}  {:<12}   {:<25}   {:<12}'.format("Iteration" ,'energy:',
        '∑_i |ϵ^n_i−ϵ^{n−1}_i| /n', 'ionization energies'))
    while hfock_diff > tol and  itr < max_itr:
        # hfock = np.zeros((L,L))
        
        hfock = hf.h + np.einsum('gd,agbd->ab', D, hf.v, optimize = True)
        e, Cdagger = np.linalg.eigh(hfock)

        C = Cdagger.T.conj()
        D = density_matr(C, F=F)
        
        try:
            assert np.all(np.isclose(C @ C.T, np.eye(L)))
        except AssertionError:
            warnings.warn('C is not unitary')
            
        hfocks.append(e)
        energies.append(calc_hf_energy(D, hf))
        
        if len(hfocks) > 1:
            hfock_diff = np.mean(np.abs(hfocks[-1] - hfocks[-2]))
        np.set_printoptions(precision=3)
        print('{:^12}  {:<12.6g}   {:<25.6g}   {}'.format(itr,
            energies[-1], hfock_diff, e))
        itr += 1
    return energies, hfocks

    
def calc_hf_energy(D, hf):
    F = hf.Z
    L = D.shape[0]

    E = np.einsum('ab,ab->', D, hf.h) + 0.5* np.einsum('ag,bd,abgd->', D, D, hf.v)
    
    return E

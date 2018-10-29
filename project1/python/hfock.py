import numpy as np
import matplotlib.pyplot as plt
from fci import FCI
import warnings


class HartreeFock(FCI):

    """Docstring for HartreeFock. """

    def __init__(self, Z):
        """TODO: to be defined1.

        :Z: TODO

        """
        FCI.__init__(self, Z)
        self.Z = Z

    def Hamiltonian(self):
        print('Not implemented')

        
    def energy_states(self):
        print('Not implemented')
        return [None]
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
    
    print(F)
    D = density_matr(C, F=F)
    energies.append(calc_hf_energy(D, hf))
    print(energies[0])

    print('{:<12}  {:<12}   {:<25}   {:<12}'.format("Iteration" ,'energy:',
        '∑_i |ϵ^n_i−ϵ^{n−1}_i| /n', 'ionization energies'))
    while hfock_diff > tol and  itr < max_itr:
        hfock = np.zeros((L,L))
        
        for alpha in range(L):
            for beta in range(L):
                hfock[alpha,beta] += hf.matrix[alpha,beta]

                for gamma in range(L):
                    for delta in range(L):
                        hfock[alpha,beta] += D[gamma,delta]*hf.matrix[alpha, gamma, beta, delta]

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
        print('{:^12}  {:<12.6f}   {:<25.6f}   {}'.format(itr,
            energies[-1], hfock_diff, e))
        itr += 1
    return C

    
def calc_hf_energy(D, hf):
    F = hf.Z
    L = D.shape[0]

    E = 0
    for alpha in range(L):
        for beta in range(L):
            E += D[alpha,beta] * hf.matrix[alpha, beta]

            for gamma in range(L):
                for delta in range(L):
                    E += 0.5*D[alpha,gamma]*D[beta,delta]*hf.matrix[alpha, beta, gamma, delta]
    
    return E

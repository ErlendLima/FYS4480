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

def solve_hfock(n, hf):
    
    hfocks = []
    energies = []
    
    F = hf.Z
    C = np.eye(n)


    hfock_diff = np.inf
    tol = 1e-2
    max_itr = 20
    itr = 0
    
    D = density_matr(C)

    while hfock_diff > tol and  itr < max_itr:
        hfock = np.zeros((n,n))
        
        for alpha in range(n):
            for beta in range(n):
                hfock[alpha,beta] += hf.matrix[alpha,beta]

                for gamma in range(n):
                    for delta in range(n):
                        hfock[alpha,beta] += D[gamma,delta]*hf.matrix[alpha, gamma, beta, delta]

        e, Cdagger = np.linalg.eigh(hfock)

        C = Cdagger.conj().T
        D = density_matr(C)
        
        try:
            assert np.all(np.isclose(C @ C.T, np.eye(n)))
        except AssertionError:
            warnings.warn('C is not unitary')
            
        hfocks.append(e)
        energies.append(calc_hf_energy(C, D, hf))
        
        if len(hfocks) > 1:
            hfock_diff = np.mean(np.abs(hfocks[-1] - hfocks[-2]))
        print(72*'=')
        print('{:<12} {:<12}'.format('energy:', 'diff'))
        print('{:<12.9f} {:<12.9f}'.format(energies[-1], hfock_diff))
        print()
        print('ionization energies:')
        print(e)
    return C

    
def calc_hf_energy(C, D, hf):
    F = hf.Z
    n = C.shape[0]

    # _, [ax1,ax2] = plt.subplots(1,2)
    # ax1.imshow(np.einsum('ji,ki->jk',C,C))
    # ax2.imshow(D)
    # plt.show()

    E = 0
    for alpha in range(n):
        for beta in range(n):
            E += D[alpha,beta]*hf.matrix[alpha, beta]

            for gamma in range(n):
                for delta in range(n):
                    E += 0.5*D[alpha,gamma]*D[beta,delta]*hf.matrix[alpha, beta, gamma, delta]
    
    return E

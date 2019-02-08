import numpy as np
import warnings


def density_matr(C, F=2):
    """ Computes the density matrix ρ

        Uses the expression ρ_γδ = Σ_j C*_jγ C_jδ
    """
    D = (C.conj().T[:, :F]) @ (C[:F, :])  # slightly faster (1.6 µs < 2.5 µs)
    return D


def solve_hfock(hf, tol=1e-6, max_itr=20, verbose=True):
    """ Finds the HarteeFock energies for the system hf

    Computes the HartreeFock Hamiltonian as
        hHF = ⟨ϕ_α|h0|ϕ_β⟩ + Σ_γδ ρ_γδ ⟨ϕ_α ϕ_γ|v| ϕ_β ϕ_δ⟩
    Solving the eigenvalue problem
        hHf C = ϵC
    using np.linalg.eigh and repeating the process until
    convergence criteria are satisfied.

    Args:
        hf: A FCI instance describing the physical system
        tol: Tolerance for when to stop the iteration
        max_itr: Maximum number of iterations

    Returns:
        Two arrays containing the Hartree-Fock energies and
        energy eigenvalues (ionization energies) for each iteration
    """

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    np.set_printoptions(precision=3)

    hfocks = []
    energies = []

    F = hf.Z
    C = np.eye(hf.n)

    hfock_diff = np.inf
    itr = 1

    D = density_matr(C, F=F)
    energies.append(calc_hf_energy(D, hf))
    vprint('Reference energy: ', energies[0])

    vprint('{:<12}  {:<12}   {:<25}   {:<12}'.format("Iteration", 'energy:',
                                                     '∑_i |ϵ^n_i−ϵ^{n−1}_i| /n',
                                                     'ionization energies'))
    while hfock_diff > tol and itr < max_itr:
        hfock = hf.h + np.einsum('gd,agbd->ab', D, hf.v, optimize=True)
        e, Cdagger = np.linalg.eigh(hfock)
        C = Cdagger.T.conj()
        D = density_matr(C, F=F)

        if not np.all(np.isclose(C @ C.T, np.eye(hf.n))):
            warnings.warn('C is not unitary')

        hfocks.append(e)
        energies.append(calc_hf_energy(D, hf))

        if len(hfocks) > 1:
            hfock_diff = np.mean(np.abs(hfocks[-1] - hfocks[-2]))

        vprint(f'{itr:^12}  {energies[-1]:<12.6g}   {hfock_diff:<25.6g}   {e}')
        itr += 1
    return energies, hfocks, C


def calc_hf_energy(D, hf):
    """ Computes the expectation value for Hartree-Fock energy

        Uses the expression
        E[Φ] = Σ_iΣ_αβ C*_αβ C_iβ ⟨ϕ_α |h_0| ϕ_β⟩
             + 1/2 ∑_ij∑_αβγδ C*_iα C*_jβ C_iγ C_jδ ⟨ϕ_αϕ_β |v| ϕ_γϕ_δ⟩
    """
    E = np.einsum('ab,ab->', D, hf.h) + 0.5 * \
        np.einsum('ag,bd,abgd->', D, D, hf.v)

    return E

"""
Solves the amplitude equations of Coupled Cluster Doubles
"""


import numpy as np
import sys
sys.path.append("../../project1/python/")
from fci import FCI
from fci import cartesian_product
from numpy import einsum
from itertools import product
from hfock import solve_hfock


def es(*args, **kwargs):
    return einsum(*args, optimize=True, **kwargs)


class CCD(FCI):
    """ Class to solve CCD equations

    Uses the framework created by FCI to set up the matrices of F and V.
    Method solve() solves the amplitude and energy equations for a given
    Z, returning the energy computed in each step of the iterative process.

    Attributes:
        Same as FCI.
        t: Array containing the amplitudes
        include_v: Whether to include the term ⟨ab|v|ij⟩. Defaults to True.
        include_L: Whether to include the L diagrams. Defaults to True.
        include_Q: Whether to include the Q diagrams. Defaults to True.
        D_ijab: The energy divisor in the iterative scheme.
    """
    def __init__(self, Z: int):
        """ Initalizes the instance

        Args:
            Z: The proton number of the target nucleus.
        """
        super().__init__(Z)
        self.t = self.initial_guess()

        # Flags to exclude some terms from the CCD expansion
        self.include_v = True
        self.include_L = True
        self.include_Q = True

    def initial_guess(self):
        """ Create an initial t_ij^ab

        Also creates the D_ijab tensor and stores it
        for later used. Should be called by __init__
        Result of t and D has been crosschecked with
        independently written code for Z=2 and Z=4.

        Returns:
            A numpy array of shape (M, M, N, N)
              where M is number of holes and N
              number of particles.
        """
        ϵ = np.einsum('pp->p', self.f)
        F, M, N = self.fermi, self.num_holes, self.num_particles
        H, P = self.holes, self.particles
        t = np.zeros((M, M, N, N))
        self.D_ijab = np.zeros_like(t)
        for i, j, a, b in self.hhpp():  # Validated using an independent source
            self.D_ijab[i, j, a-F, b-F] = ϵ[i] + ϵ[j] - ϵ[a] - ϵ[b]
        t = self.v[H, H, P, P]/self.D_ijab
        return t

    def Ecorr(self):
        # E_corr = 1/4Σ_abij ⟨ij|v|ab⟩t_ij^ab
        H, P = self.holes, self.particles
        return 1/4*es('ijab,ijab', self.v[H, H, P, P], self.t)

    def energy(self):
        # E_CCD = E_ref + E_corr = Eref + 1/4Σ_abij ⟨ij|v|ab⟩t_ij^ab
        return self.Eref + self.Ecorr()

    def iterate(self, t: np.ndarray) -> np.ndarray:
        """ Compute one step of a CCD iterative calculation

        Args:
            t: The amplitudes t^k. Is not mutated.
        Returns:
            The new amplitudes t^(k+1).
        Algorithm:

        Compute ⟨Φ_ij^ab|̄H|0⟩ = 0. Splitting up the operator, we have
        Ĥ =  Eref + F̂ + V̂, H̄ = exp(T)Ĥ exp(-T) so
        ⟨Φ_ij^ab|F|c⟩ = 0
        ⟨Φ_ij^ab|[[F, T], T]|c⟩ = 0
        ⟨Φ_ij^ab|̂V|c⟩ = ⟨ab|v̂|ij⟩                                       (1)
        ⟨Φ_ij^ab|[F,T]|c⟩ = P(ij)Σ_k f_i^k t_jk^ab                      (2.a)
                          - P(ab)Σ_c f_c^a t_ij^bc  (***)               (2.b)
        ⟨Φ_ij^ab|[V,T]|c⟩ = ½Σ_kl⟨kl|v̂|ij⟩t_kl^ab                       (3.a)
                          + P(ab)P(ij)Σ_kc⟨ak|v̂|ic⟩t_jk^bc              (3.b)
                          + ½Σ_cd⟨ab|v̂|cd⟩t_ij^cd                       (3.c)
        ½⟨Φ_ij^ab|[[V,T], T]|c⟩ = ¼     Σ_klcd⟨kl|v|cd⟩t_kl^ab t_ij^cd  (4.a)
                                - ½P(ij)Σ_klcd⟨kl|v|cd⟩t_il^ab t_kj^cd  (4.b)
                                -  P(ab)Σ_klcd⟨kl|v|cd⟩t_ik^ac t_lj^bd  (4.c)
                                - ½P(ab)Σ_klcd⟨kl|v|cd⟩t_ij^ac t_kl^bd  (4.d)
        (***) For the iterative scheme we change these into
        ⟨Φ_ij^ab|[F,T]|c⟩** = P(ij)Σ_k≠i f_i^k t_jk^ab                  (2.a)
                            - P(ab)Σ_c≠a f_c^a t_ij^bc  (***)
        The amplitudes is found iteratively by computing
        (t_ij^ab)^(k+1) = g((t_ij^ab)^(k))/(ϵ_i + ϵ_j - ϵ_a - ϵ_b)
        where g(...) is the modified sum shown above.
        """
        H, P = self.holes, self.particles
        v = self.v
        vpppp = v[P, P, P, P]
        vhhhh = v[H, H, H, H]
        vhhpp = v[H, H, P, P]
        vphph = v[P, H, P, H]

        # Take care of special case of k≠i and c≠a by removing diagonal
        f0 = self.f.copy()
        f0[np.diag_indices_from(f0)] = 0
        T1 = (es('ki,jkab->ijab', f0[H, H], t)               # (2.a) L1b
             - es('kj,ikab->ijab', f0[H, H], t)             # (2.a)
             - es('ac,ijbc->ijab', f0[P, P], t)             # (2.b) L1a
             + es('bc,ijac->ijab', f0[P, P], t)             # (2.b)
        )

        HN = np.copy(vhhpp)                                 # (1)
        # Developer note: Sign errors cause insane oscillations
        L = (es('ki,jkab->ijab', f0[H, H], t)               # (2.a) L1b
             - es('kj,ikab->ijab', f0[H, H], t)             # (2.a)
             - es('ac,ijbc->ijab', f0[P, P], t)             # (2.b) L1a
             + es('bc,ijac->ijab', f0[P, P], t)             # (2.b)
             + 1/2*es('klij,klab->ijab', vhhhh, t)          # (3.a) L2b
             + 1/2*es('abcd,ijcd->ijab', vpppp, t)          # (3.c) L2a
             - es('akcj,ikcb->ijab', vphph, t)              # (3.b) L2c
             + es('bkcj,ikca->ijab', vphph, t)              # (3.b)
             + es('akci,jkcb->ijab', vphph, t)              # (3.b)
             - es('bkci,jkca->ijab', vphph, t)              # (3.b)
            )
        Q = (1/4*es('klcd,klab,ijcd->ijab', vhhpp, t, t)    # (4.a) Qa
             - 1/2*es('klcd,ilab,kjcd->ijab', vhhpp, t, t)  # (4.b) Qc
             + 1/2*es('klcd,jlab,kicd->ijab', vhhpp, t, t)  # (4.b) Qc
             - es('klcd,ikac,ljbd->ijab', vhhpp, t, t)      # (4.c) Qb
             + es('klcd,ikbc,ljad->ijab', vhhpp, t, t)      # (4.c) Qb
             - 1/2*es('klcd,ijac,klbd->ijab', vhhpp, t, t)  # (4.d) Qd
             + 1/2*es('klcd,ijbc,klad->ijab', vhhpp, t, t)  # (4.d) Qd
        )

        if self.include_L:
            HN += L
        if self.include_Q:
            HN += Q
        return HN/self.D_ijab

    def solve(self, tol: float = 1e-10, maxiter: int = 1e4) -> ([float], [float]):
        """ Solves the CCD equations iteratively.

        Args:
            tol: Absolute difference between the energy of the
                last two iterations below which the iteration
                will terminate.
            maxiter: The maximum number of iterations above which
                the iteration will terminate.
        Returns:
            Two lists containing the total energy and correlation energy
            of each iteration. The final index corresponds to the final
            iteration.
        """
        self.t = self.iterate(self.t)
        E = [self.energy()]
        Ecorr = [self.Ecorr()]
        counter = 1
        diff = np.inf
        while counter < maxiter and diff > tol:
            self.t = self.iterate(self.t)
            E.append(self.energy())
            Ecorr.append(self.Ecorr())
            diff = abs((E[-1]-E[-2])/E[-1])
            counter += 1

        return E, Ecorr

    def transform_to_hfock(self):
        _, _, C = solve_hfock(self, verbose=False)
        # ⟨p|f|q⟩ = εₚδ_pq
        self.f[:] = np.eye(len(np.diag(self.f)))*self.f
        Cd = C.conj()
        # ⟨pq|v|rs⟩ = Σ_αβγδ C_αp^*  C_βq^* Cγr Cδs ⟨αβ|v|γδ⟩
        self.v[:] = np.einsum("ap,bq,gr,ds,abgd->pqrs", Cd, Cd, C, C, self.v)

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
import matplotlib.pyplot as plt


def es(*args, **kwargs):
    return einsum(*args, optimize=True, **kwargs)


class CCD(FCI):
    def __init__(self, Z):
        super().__init__(Z)
        self.t = self.initial_guess()
        self.include_v = True
        self.include_L = True
        self.include_Q = True

    def initial_guess(self):
        ϵ = np.einsum('pp->p', self.f)
        F, M, N = self.fermi, self.num_holes, self.num_particles
        H, P = self.holes, self.particles
        t = np.zeros((M, M, N, N))
        self.D_ijab = np.zeros_like(t)
        for i, j, a, b in self.hhpp():  # Validated with independent source
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

    def iterate(self, t):
        """
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
        # Special case of k≠i and c≠a                     (2.a)
        f0 = self.f.copy()
        f0[np.diag_indices_from(f0)] = 0

        # Sign errors cause insane oscillations
        HN = np.copy(vhhpp)
        L = (
            + es('ki,jkab->ijab', f0[H, H], t)             # (2.a) L1b
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
        Q = (
            + 1/4*es('klcd,klab,ijcd->ijab', vhhpp, t, t)  # (4.a) Qa
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

    def solve(self, tol=1e-10, maxiter=1e4):
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

        return E, Ecorr

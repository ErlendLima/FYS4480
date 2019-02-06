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

    def initial_guess(self):
        ϵ = np.einsum('pp->p', self.f)
        F, M, N = self.fermi, self.num_holes, self.num_particles
        H, P = self.holes, self.particles
        t = np.zeros((M, M, N, N))
        self.D_ijab = np.zeros_like(t)
        for i, j, a, b in self.hhpp():
            self.D_ijab[i, j, a-F, b-F] = ϵ[i] + ϵ[j] - ϵ[a] - ϵ[b]
        t = self.v[H, H, P, P]/self.D_ijab
        return t

    def energy(self):
        # E_CCD = E_ref + E_corr = Eref + 1/4Σ_abij ⟨ij|v|ab⟩t_ij^ab
        H, P = self.holes, self.particles
        return self.Eref + 1/4*es('ijab,ijab', self.v[H, H, P, P], self.t)

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
        HN = (
            vhhpp                                          # (1)
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
            + 1/4*es('klcd,klab,ijcd->ijab', vhhpp, t, t)  # (4.a) Qa
            - 1/2*es('klcd,ilab,kjcd->ijab', vhhpp, t, t)  # (4.b) Qc
            + 1/2*es('klcd,jlab,kicd->ijab', vhhpp, t, t)  # (4.b) Qc
            - es('klcd,ikac,ljbd->ijab', vhhpp, t, t)      # (4.c) Qb
            + es('klcd,ikbc,ljad->ijab', vhhpp, t, t)      # (4.c) Qb
            - 1/2*es('klcd,ijac,klbd->ijab', vhhpp, t, t)  # (4.d) Qd
            + 1/2*es('klcd,ijbc,klad->ijab', vhhpp, t, t)  # (4.d) Qd
        )
        return HN/self.D_ijab

    def run(self):
        self.t = self.initial_guess()
        N = 100
        E = np.zeros(N)
        for counter in range(N):
            self.t = self.iterate(self.t)
            E[counter] = self.energy()
        plt.plot(E, '-o')
        plt.show()

    def explicit_iterate(self):
        t = self.initial_guess()
        v = self.v
        f = self.h
        for counter in range(10):
            tk = np.copy(t)
            for i, j, a, b in self.hhpp():
                t[a, b, i, j] = v[a, b, i, j]
                # for k in self.core:       # (2.a)  #ALWAYS 0 for HF. Gives roundoff error
                #     if k == i:
                #         continue
                #     t[a, b, i, j] += f[i, k]*tk[a, b, j, k] - f[j, k]*tk[a, b, i, k]
                # for c in self.virtual:   # (2.b)
                #     if c == a:
                #         continue
                #     t[a, b, i, j] -= f[c, a]*tk[b, c, i, j] - f[c, b]*tk[a, c, i, j]
                #     print(f[c, a]*tk[b, c, i, j], - f[c, b]*tk[a, c, i, j])
                # for k, l in self.hh():  # Causes oscillations
                    # t[a, b, i, j] += 1/2*v[k, l, i, j]*tk[a, b, k, l]
                # for k, c in self.hp():
                    # t[a, b, i, j] += v[a, k, i, c]*tk[b, c, j, k] - v[b, k, j, c]*tk[a, c, i, k]
                for c, d in self.pp():
                    t[a, b, i, j] += 1/2*v[a, b, c, d]*tk[c, d, i, j]
                for k, l, c, d in self.hhpp():
                    t[a, b, i, j] += 1/4*v[k, l, c, d]*tk[a, b, k, l]*tk[c, d, i, j]
                    t[a, b, i, j] -= 1/2*(v[k, l, c, d]*tk[a, b, i, l]*tk[c, d, k, j] -
                                          v[k, l, c, d]*tk[a, b, j, l]*tk[c, d, k, i])
                    t[a, b, i, j] -= (v[k, l, c, d]*tk[a, c, i, k]*tk[b, d, l, j] -
                                      v[k, l, c, d]*tk[b, c, i, k]*tk[a, d, l, j])
                    t[a, b, i, j] -= 1/2*(v[k, l, c, d]*tk[a, c, i, j]*tk[b, d, k, l] -
                                          v[k, l, c, d]*tk[b, c, i, j]*tk[a, d, k, l])
                t[a, b, i, j] /= self.D_abij[a, b, i, j]
            self.t = t
            # print(t.sum())
            print(counter, ":", self.energy())



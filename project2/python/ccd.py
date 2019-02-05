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


def es(*args, **kwargs):
    return einsum(*args, optimize=True, **kwargs)


class CCD(FCI):
    def __init__(self, Z):
        super().__init__(Z)
        self.t = self.initial_guess()

    def initial_guess(self):
        ϵ = np.einsum('ii->i', self.h)
        M, N = self.num_particles, self.num_holes
        # t = np.zeros((M, M, N, N))
        t = np.zeros_like(self.v)
        self.D_abij = np.zeros_like(self.v)
        indicies = tuple(zip(*self.pphh()))
        mask = np.full(t.shape, False)
        mask[indicies] = True
        for i, j, a, b in self.hhpp():
            self.D_abij[a, b, i, j] = ϵ[i] - ϵ[j] + ϵ[a] + ϵ[b]
        t[mask] = self.v[mask]/self.D_abij[mask]
        self.mask = mask
        return t

    def energy(self):
        # e = 0
        # for a, b, i, j in self.pphh():
        #     e += 1/4*self.v[a, b, i, j]*self.t[a, b, i, j]
        # return e
        return self.Eref + 1/4*es('ijab,abij', self.v, self.t)

    def iterate(self):
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
                            - P(ab)Σ_c≠a f_c^a t_ij^bc  (***)           (2.b)

        The amplitudes is found iteratively by computing
        (t_ij^ab)^(k+1) = g((t_ij^ab)^(k))/(ϵ_i + ϵ_j - ϵ_a - ϵ_b)
        where g(...) is the modified sum shown above.
        """
        t = self.initial_guess()

        v = self.v
        f = self.h
        ϵ = np.einsum('ii->i', self.h)
        mask = self.mask
        holes = self.virtual
        part = self.core
        for counter in range(10):
            H = (
                v                                           # (1)
                # - es('ca,bcij->abij', f, t)                 # (2.b)
                # + es('cb,acij->abij', f, t)                 # (2.b)
                # + 1/2*es('klij,abkl->abij', v, t)           #
                # + es('klij,abkl->abij', v, t)               # (3.b)
                # - es('klji,bakl->abij', v, t)               # (3.b)
                + 1/2*es('abcd,cdij->abij', v, t)           # (3.c)
                + 1/4*es('klcd,abkl,cdij->abij', v, t, t)   # (4.a)
                - 1/2*es('klcd,abil,cdkj->abij', v, t, t)   # (4.b)
                + 1/2*es('klcd,abjl,cdki->abij', v, t, t)   # (4.b) i→j
                - es('klcd,acik,bdlj->abij', v, t, t)   # (4.c)
                + es('klcd,bcik,adlj->abij', v, t, t)   # (4.c) a→b
                - 1/2*es('klcd,acij,bdkl->abij', v, t, t)   # (4.d)
                + 1/2*es('klcd,bcij,adkl->abij', v, t, t)   # (4.d) a→b
                )
            # Special case of k≠i and c≠a                     (2.a)
            # for i, j, a, b in product(holes, holes, part, part):
            #     for k in range(self.n):
            #         if k == i:
            #             continue
            #         H += f[i, k]*t[a, b, j, k] - f[j, k]*t[a, b, i, k]
            #     for c in range(self.n):
            #         if c == a:
            #             continue
            #         H -= f[c, a]*t[b, c, i, j] - f[c, b]*t[a, c, i, j]

            t[mask] = H[mask]/self.D_abij[mask]
            # print("Waste: ", H[~mask].sum())
            self.t = t
            print(counter, ":", self.energy())
        self.t = t

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



import numpy as np
from matrixelementparser import MatrixElementParser
from itertools import product


def cartesian_product(*args):
    """ Yields tuples over every integer coordinate in ⨉(n, m, ...) """
    for element in product(*[range(n) for n in args]):
        yield element


def pureproperty(func):
    """ Same as property, but the function's body will only execute _once_

    Is equivalent to rewriting a function to the form
    @property
    def foo(self):
       if not hastattr(self, _foo):
           self._foo = self.original_foo()
       return self._foo
    """
    name = '_' + func.__name__

    def wrapper(self=None):
        if not hasattr(self, name):
            setattr(self, name, func(self))
        return getattr(self, name)
    return property(wrapper)


class FCI:
    def __init__(self, Z):
        self.Z = Z
        self.matrix = MatrixElementParser(Z=Z)
        # n is set by the modelspace, 1s 2s 3s with degeneracy 2
        # total number of single particle states
        self.n = 2+2+2
        # Set Fermi level equal to the element's proton number
        self.fermi = Z
        self.num_particles = self.n - self.fermi
        self.num_holes = self.fermi
        self.core = np.arange(self.fermi)
        self.virtual = np.arange(self.num_particles) + self.fermi
        self.hole_ind = slice(0, self.fermi)           # i, j, k, ...
        self.particle_ind = slice(self.fermi, self.n)  # a, b, c, ...

    @pureproperty
    def Eref(self):
        """ The reference energy """
        Eref = 0
        for i in range(self.Z):
            for j in range(self.Z):
                Eref += 0.5*self.matrix[i, j, i, j]
            Eref += self.matrix[i, i]
        return Eref

    @pureproperty
    def h(self):
        n = self.n
        h = np.zeros((n, n))
        for α, β in cartesian_product(n, n):
            h[α, β] = self.matrix[α, β]
        return h

    @pureproperty
    def v(self):
        n = self.n
        v = np.zeros((n, n, n, n))
        for α, β, γ, δ in cartesian_product(n, n, n, n):
            v[α, β, γ, δ] = self.matrix[α, β, γ, δ]
        return v

    def excitations(self):
        """ Returns a list of all possible single excitations

            The spin is encoded as n % 2 and state as n // 2 + 1
            where spin ∈ [0, 1] and state ∈ 1s, 2s, 3s.
            Each tuple has the form (initial, final).
            >>> [(0, 2), (0, 4), (1, 3), (1, 5)]
        """
        excitations = []
        for i in self.core:
            for a in self.virtual:
                if (i % 2) == (a % 2):
                    excitations.append((i, a))

        return excitations

    def shifted_excitations(self):
        """ Same as excitations but with particle indices shifted by M """
        M = self.num_particles
        return [(i, a-M) for (i, a) in self.excitations()]

    def double_excitations(self):
        for i, a in self.excitations():
            for j, b in self.excitations():
                yield i, j, a, b

    def shifted_double_excitations(self):
        M = self.num_particles
        for i, a in self.excitations():
            for j, b in self.excitations():
                yield i, j, a-M, b-M

    def hhpp(self):
        return product(self.core, self.core, self.virtual, self.virtual)

    def pphh(self):
        return product(self.virtual, self.virtual, self.core, self.core)

    def hh(self):
        return product(self.core, self.core)

    def pp(self):
        return product(self.virtual, self.virtual)

    def hp(self):
        return product(self.core, self.virtual)

    def Hamiltonian(self):
        """ Constructs the Hamiltonian """
        nsingles = len(self.excitations())

        ref_ref = np.zeros((1, 1))
        ref_sing = np.zeros((nsingles, 1))
        sing_sing = np.zeros((nsingles, nsingles))

        ref_ref[0, 0] = self.Eref
        for ind1, (i, a) in enumerate(self.excitations()):
            # Evaluate ⟨c|H|Φ_i^a⟩ = ⟨i|h|a⟩ + Σ⟨ij|v|aj⟩
            ref_sing[ind1] = self.h[i, a]  # = 0 always
            ref_sing[ind1] += sum(self.v[i, j, a, j] for j in self.core)

            # Evaluate <Φ_i^a|H|Φ_j^b> = <aj|v|ib>_AS + <a|h|b>δij
            #                          - <j|h|i>δab   + Erefδijδab
            for ind2, (j, b) in enumerate(self.excitations()):
                E = self.v[a, j, i, b]
                E += self.h[a, b] if i == j else 0
                E -= self.h[j, i] if a == b else 0
                E += self.Eref if ((i == j) and (a == b)) else 0
                for k in range(self.fermi):
                    E += self.v[a, k, b, k] if i == j else 0
                    E -= self.v[j, k, i, k] if a == b else 0
                sing_sing[ind1, ind2] = E

        H = np.block([[ref_ref, ref_sing.T],
                      [ref_sing, sing_sing]])
        return H

    def energy_states(self):
        H = self.Hamiltonian()
        e, v = np.linalg.eigh(H)
        return e, v


if __name__ == '__main__':
    system = FCI(Z=2)
    print(system.Hamiltonian())
    print(FCI(Z=2).energy_states()[0])
    # print(FCI(Z=4).energy_states()[0])

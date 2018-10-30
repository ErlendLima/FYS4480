import numpy as np
from matrixelementparser import MatrixElementParser
from itertools import product


def cartesian_product(*args):
    """ Yields tuples over every integer coordinate in ⨉(n, m, ...) """
    for element in product(*[range(n) for n in args]):
        yield element


class FCI:
    def __init__(self, Z, n=6):
        """
            TODO: Find a better way of defining n
        """
        self.Z = Z
        self.matrix = MatrixElementParser(Z=Z)
        self.n = n

    @property
    def Eref(self):
        """ The reference energy """
        try:
            return self._Eref
        except AttributeError:
            Eref = 0
            for i in range(self.Z):
                for j in range(self.Z):
                    Eref += 0.5*self.matrix[i, j, i, j]
                Eref += self.matrix[i, i]
            self._Eref = Eref
            return Eref

    @property
    def h(self):
        try:
            return self._h
        except AttributeError:
            n = self.n
            self._h = np.zeros((n, n))
            for α, β in cartesian_product(n, n):
                self._h[α, β] = self.matrix[α, β]
            return self._h

    @property
    def v(self):
        try:
            return self._v
        except AttributeError:
            n = self.n
            self._v = np.zeros((n, n, n, n))
            for α, β, γ, δ in cartesian_product(n, n, n, n):
                self._v[α, β, γ, δ] = self.matrix[α, β, γ, δ]
            return self._v

    def Hamiltonian(self):
        """ Constructs the Hamiltonian """

        # Fix these lines later to generalize
        n_states = 5

        F = self.Z
        core = np.arange(F)
        n_virtual = 6 - self.Z
        virtual = np.arange(n_virtual) + F

        refs = [0, 1]
        # this code needs a bit of rework if the ground state is degenerate
        nref = len(refs)

        singles = []  # [(0,2),(1,3),(0,4),(1,5)]

        for i in core:
            for a in virtual:
                if (i % 2) == (a % 2):
                    singles.append((i, a))

        nsingles = len(singles)

        ref_ref = np.zeros((1, 1))
        ref_sing = np.zeros((nsingles, 1))
        sing_sing = np.zeros((nsingles, nsingles))

        ref_ref[0, 0] = self.Eref
        for ind1, (i, a) in enumerate(singles):
            # Evaluate <c|H|Φ_i^a> = <i|h|a> + Σ<ij|a|aj>
            ref_sing[ind1] = self.matrix[i, a]  # = 0 always
            ref_sing[ind1] += sum(self.matrix[i, j, a, j] for j in core)

            # Evaluate <Φ_i^a|H|Φ_j^b> = <aj|v|ib>_AS + <a|h|b>δij
            #                          - <j|h|i>δab + Erefδijδab
            for ind2, (j, b) in enumerate(singles):
                E = self.matrix[a, j, i, b]
                E += self.matrix[a, b] if i == j else 0
                E -= self.matrix[j, i] if a == b else 0
                E += self.Eref if ((i == j) and (a == b)) else 0
                for k in range(F):
                    E += self.matrix[a, k, b, k] if i == j else 0
                    E -= self.matrix[j, k, i, k] if a == b else 0
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

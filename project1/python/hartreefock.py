import numpy as np
from matrixelementparser import MatrixElementParser


class HartreeFock:
    def __init__(self, Z):
        self.Z = Z
        self.matrix = MatrixElementParser(Z=Z)

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
                Eref += self.matrix.onebody(i, i)
            self._Eref = Eref
            return Eref

    def Hamiltonian(self):
        """ Constructs the Hamiltonian """

        # Fix these lines later to generalize
        n_states = 5

        F = 2
        core = np.arange(F)
        n_virtual = 4
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
            ref_sing[ind1] = self.matrix[i, a]
            ref_sing[ind1] = sum(self.matrix[i, j, a, j] for j in core)

            # Evaluate <Φ_i^a|H|Φ_j^b> = <aj|v|ib>_AS + <a|h|b>δij
            #                          - <j|h|i>δab + Erefδijδab
            for ind2, (j, b) in enumerate(singles):
                E = self.matrix[a, j, i, b]
                E += self.matrix[a, b] if i == j else 0
                E -= self.matrix[j, i] if a == b else 0
                E += self.Eref if (i == j and a == b) else 0
                for k in range(nsingles):
                    E += self.matrix[a, k, b, k] if i == j else 0
                    E -= self.matrix[j, k, i, k] if a == b else 0
                sing_sing[ind1, ind2] = E

        H = np.block([[ref_ref, ref_sing.T],
                      [ref_sing, sing_sing]])
        return H

    def energy_states(self):
        H = self.Hamiltonian()
        e, v = np.linalg.eig(H)
        return e, v


if __name__ == '__main__':
    system = HartreeFock(Z=2)
    print(system.Hamiltonian())
    system.energy_states()[0]

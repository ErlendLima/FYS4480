from fci import FCI

class HartreeFock(FCI):

    """Docstring for HartreeFock. """

    def __init__(self, Z):
        """TODO: to be defined1.

        :Z: TODO

        """
        FCI.__init__(self, Z)

        self._Z = Z
    def Hamiltonian(self):
        print('Not implemented')

        
    def energy_states(self):
        print('Not implemented')
        return [None]

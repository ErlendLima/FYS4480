from sympy.parsing import sympy_parser
import sympy as sp
import numpy as np
import pandas as pd


class MatrixElementParser:

    """
    Reads matrix element expressions from file and stores as a pandas
    dataframe of sympy expressions, which can be evaluated for
    different Z with i.e self.eval_data(Z=2).
    """

    def __init__(self, filename='../data/matrix_data.txt', Z=1):
        self.filename = filename
        self.sympy_data = self.read_data(filename)
        self.Z = Z  # invokes property

    def __getitem__(self, ind):
        """
        Gets matrix element <pq|V|rs>_AS from ind = [p,q,r,s], assuming
        p,q,r,s ∈ {0,...,5} with odd and even integers corresponding to
        opposite spins, and (p//2 + 1) as the energy levels 1s, 2s, 3s.
        """
        ind = np.array(ind)
        if np.any(ind < 0) or np.any(5 < ind):
            raise IndexError("Index out of range.")

        if ind.shape[0] == 2:
            return self.onebody(*ind)
        elif ind.shape[0] != 4:
            raise IndexError("Expected 2 or 4 indices.")

        # spin indices
        s1, s2, s3, s4 = ind % 2
        # spatial indices
        r1, r2, r3, r4 = ind // 2 + 1

        matr = self.matrix
        mel = matr.loc[(r1, r2), (r3, r4)] * ((s1 == s3) and (s2 == s4)) \
            - matr.loc[(r1, r2), (r4, r3)] * ((s1 == s4) and (s2 == s3))
        return mel

    def onebody(self, p, q):
        n1 = p//2 + 1
        n2 = q//2 + 1
        return -self.Z**2/(2*n1**2) * (n1 == n2)

    @staticmethod
    def read_data(filename):
        """
        Reads matrix element data from file, structured with lines
        containing 'index1 index2 expression', separated by spaces.
        """
        sympy_data = pd.DataFrame()
        with open(filename) as infile:
            for line in infile.read().split('\n'):
                if not line:
                    continue
                spl = line.split()
                i1, i2 = spl[:2]
                s = sympy_parser.parse_expr(spl[-1])
                sympy_data.loc[i1, i2] = s
        return sympy_data

    def eval_data(self, Z=None, index_type='multiindex'):
        """
        Evaluates matrix elements for a given Z, defaulting to given Z. Returns pandas DataFrame.
        """
        Z = Z or self.Z
        Z_symb = sp.symbols('Z')

        m = self.sympy_data.applymap(lambda s: s.evalf(subs={Z_symb: Z}))

        # m = matr_parse.sympy_data.applymap(lambda s:s.evalf(subs = {Z_symb:Z}))
        if index_type == 'multiindex':
            col1, col2 = m.columns.str
            col1, col2 = col1.astype('int'), col2.astype('int')
            row1, row2 = m.index.str
            row1, row2 = row1.astype('int'), row2.astype('int')

            row_ind = pd.MultiIndex.from_arrays([row1, row2])
            col_ind = pd.MultiIndex.from_arrays([col1, col2])
            m.index = row_ind
            m.columns = col_ind
        return m.astype('float')

    @property
    def matrix(self):
        """Returns matrix"""
        try:
            return self._matrix
        except AttributeError:
            self._matrix = self.eval_data()
            return self._matrix

    @property
    def Z(self):
        """Very redundant :)"""
        return self._Z

    @Z.setter
    def Z(self, Z):
        self._Z = Z

    def show(self):
        """
        Plots the values of the matrix, for Z = 1.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        matr = self.eval_data(1)
        fig, ax = plt.subplots()

        m = ax.matshow(np.array(matr.values, dtype=np.float64))

        ax.set_yticks(np.arange(matr.shape[0]), minor=False)
        ax.set_xticks(np.arange(matr.shape[1]), minor=False)
        ax.set_yticklabels(matr.columns, minor=False)
        ax.set_xticklabels(matr.index, minor=False)

        cax = fig.colorbar(m, fraction=0.046, pad=0.04)
        cax.set_label('Units of $Z$')
        plt.show()


if __name__ == "__main__":
    a = MatrixElementParser()
    print(a[2, 2, 2, 2])

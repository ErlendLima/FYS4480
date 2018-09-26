from sympy.parsing import sympy_parser
from sympy.core.function import Lambda
from sympy import symbols
import pandas as pd

class MatrixElementParser(object):

    """Reads matrix element expressions from file and stores as a pandas
    dataframe of sympy expressions, which can be evaluated for
    different Z with i.e self.eval_data(Z=2). """

    def __init__(self, filename = '../data/matrix_data.txt'):
        self.filename = filename
        self.sympy_data = self.read_data(filename)

    def read_data(self, filename):
        sympy_data = pd.DataFrame()
        with open(filename) as infile:
            lines = infile.read()
            for line in lines.split('\n'):
                if not line:
                    continue
                # print(line)
                spl = line.split()
                i1, i2 = spl[:2]
                s = sympy_parser.parse_expr(spl[-1])
                sympy_data.loc[i1,i2] = s
        return sympy_data
    
    def eval_data(self, Z):
        Z = symbols('Z')
        return self.sympy_data.applymap(lambda s:s.evalf(subs = {Z:1}))

    def show_matrix(self, Z):
        import numpy as np
        import matplotlib.pyplot as plt
        matr = self.eval_data(1)
        fig,ax = plt.subplots(1)

        m = plt.imshow(np.array(matr.values, dtype = np.float64))

        ax.set_yticks(np.arange(matr.shape[0]), minor = False)
        ax.set_xticks(np.arange(matr.shape[1]), minor = False)
        ax.set_yticklabels(matr.columns, minor = False)
        ax.set_xticklabels(matr.index, minor = False)
        ax.xaxis.tick_top()

        cax = plt.colorbar(m)#, orientation='horizontal')
        cax.set_label('Units of $Z$')
        plt.show()

if __name__ == "__main__":
    a = MatrixElementParser()
    print(a.eval_data(1))
    a.show_matrix(1)

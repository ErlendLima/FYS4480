import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-Z', '--Z', default = [2,4], type = int,nargs = '*')
    parser.add_argument('-m', '--solver',default = 'fci',choices =
            ['fci','hfock','all'])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    from fci import FCI
    from hfock import HartreeFock
    args = get_args()

    solvers = {'fci':FCI, 'hfock':HartreeFock}

    for solver in solvers:
        if solver == args.solver or args.solver == 'all':
            print(60*'=')
            print('{:^60}'.format(solver.upper()))
            print(60*'=')

            for Z in args.Z:
                print()
                print('Z = {}'.format(Z))
                system = solvers[solver](Z=Z)
                print(system.Hamiltonian())
                print(system.energy_states()[0])

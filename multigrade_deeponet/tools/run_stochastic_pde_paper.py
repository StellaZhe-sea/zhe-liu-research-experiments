"""Pathwise stochastic elliptic PDE with the paper's sigma=0.1."""
import sys
from run_stochastic_ode_paper import main

if __name__ == '__main__':
    main(['--problem', 'pde'] + sys.argv[1:])

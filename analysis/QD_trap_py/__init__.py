"""
Quantum dot trap state analysis module for Python
=================================================

QD_trap_py is a Python module that analyzes the relationship between QD structure and trap state
distribution based on a range of scientific packages (numpy, scipy, matplotlib, scikit-learn). 
Geometry optimization and electronic structure prediction were performed within the DFT framework
using the VASP packgage.

"""

import sys

__all__ = ['load_data', 
           'build_feature',
           'trap_classify',
           'trap_regress',
          ]


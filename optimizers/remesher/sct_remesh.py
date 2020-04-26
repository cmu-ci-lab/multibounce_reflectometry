# sct_remesh.py

# Simple remeshing script that takes a mesh with perturbed normals
# and reconstructs the surface with it and writes it back to the given mesh

import remesh
import sys
import integrate
import algorithms.poisson.integrator

integrator = None
if len(sys.argv) > 2:
    if sys.argv[3] == "frankot":
        integrator = integrate.integrate
    elif sys.argv[3] == "poisson":
        integrator = algorithms.poisson.integrator.integrate
else:
    integrator = integrate.integrate

assert(sys.argv[1].endswith(".ply"))
print(integrator)
remesh.remesh(sys.argv[1], sys.argv[2], integrator=integrator)
print("###RUN###")
import numpy as np

import ufl
import dolfinx
from dolfinx import mesh, fem, io, plot
from dolfinx.mesh import locate_entities_boundary, meshtags, locate_entities
from ufl import ds, dx, grad, inner, Measure
# import dolfinx_mpc

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

n=10 # discretization number
msh = mesh.create_box(MPI.COMM_WORLD,[[0.0,0.0,0.0], [1.0, 1.0, 1.0]], [n, n, n], mesh.CellType.hexahedron)
V = fem.FunctionSpace(msh, ("CG", 1)) # the space of functions
 
in_neumann_boundary = lambda x: np.isclose(x[0],0) | np.isclose(x[0],1)


# Variational Problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = ufl.sin(5 * x[0]) + ufl.sin(3 * x[1]) + ufl.sin(2 * x[2])
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds


problem = fem.petsc.LinearProblem(a, L, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()


# plot option 1:
try:
    import pyvista
    cells, types, x = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    plotter.show(screenshot='demo_poisson.png')
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")


""" unaswered questions:
- how to create functions in function space from data
    -> see https://fenicsproject.discourse.group/t/mapping-2d-numpy-array-into-dolfinx-function/7487/4

"""

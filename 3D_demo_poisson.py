""" https://fenicsproject.org/olddocs/dolfin/1.4.0/python/demo/documented/periodic/python/documentation.html """
print("###RUN###")
import numpy as np

import ufl
import dolfinx
from dolfinx import fem, io, mesh, plot
from dolfinx.mesh import locate_entities_boundary, meshtags
from ufl import ds, dx, grad, inner
# import dolfinx_mpc

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

n=10 # discretization number
msh = mesh.create_box(MPI.COMM_WORLD,[[0.0,0.0,0.0], [1.0, 1.0, 1.0]], [n, n, n], mesh.CellType.hexahedron)
V = fem.FunctionSpace(msh, ("CG", 1)) # the space of functions
 
"""
# Periodic condition:
def PeriodicBoundary(x):
        return np.isclose(x[0], 1)

    facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, PeriodicBoundary)
    arg_sort = np.argsort(facets)
    mt = meshtags(mesh, mesh.topology.dim - 1, facets[arg_sort], np.full(len(facets), 2, dtype=np.int32))

    def periodic_relation(x):
        out_x = np.zeros(x.shape)
        out_x[0] = 1 - x[0]
        out_x[1] = x[1]
        out_x[2] = x[2]
        return out_x
    with Timer("~~Periodic: Compute mpc condition"):
        mpc = dolfinx_mpc.MultiPointConstraint(V)
        mpc.create_periodic_constraint_topological(V.sub(0), mt, 2, periodic_relation, bcs, 1)
        mpc.finalize()
"""

# Dirichlet:
def boundary_D(x):
    return np.isclose(np.min(x,axis=0),0) | np.isclose(np.max(x,axis=0),1)
dofs_D = fem.locate_dofs_geometrical(V, boundary_D)
# dofs = fem.locate_dofs_topological(V=V, entity_dim=2, entities=facets)
uD = fem.Function(V)
uD.interpolate(lambda x:  x[0]**2 + 2 * x[1]**2 + 3*x[2]**2)
bc = fem.dirichletbc(uD, dofs=dofs_D)

# Variational Problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
g = ufl.sin(5 * x[0]) + ufl.sin(3 * x[1]) + ufl.sin(2 * x[2])
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds

problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
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

"""
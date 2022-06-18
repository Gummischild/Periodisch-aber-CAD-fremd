"""
periodische RBs (links/rechts) funktionieren
!!! Dieser Code benötigt das Python-Paket mshr !!!

"""

from __future__ import print_function
import dolfin as dlfn
from dolfin import dot, inner, outer, grad, div #....extra definieren, damit etwaige Gleichung übersichtlich bleiben.
from vedo.dolfin import * # ....damit die Simulation sofort in einem Plot sichtbar ist.
import matplotlib.pyplot as plt
import mshr as mshr
import numpy as np

Rohrumfang =dlfn.Constant(dlfn.pi*0.1)

# Create mesh
channel = mshr.Rectangle(dlfn.Point(0, 0), dlfn.Point(1, 1))
Rohr_1 = mshr.Circle(dlfn.Point(0.25, 0.5), 0.1)
Rohr_2 = mshr.Circle(dlfn.Point(0.75, 0.5), 0.1)
domain = channel - Rohr_1 - Rohr_2 
mesh = mshr.generate_mesh(domain, 60)

mesh=dlfn.refine(mesh)
#plot(mesh)
#mesh=dlfn.refine(mesh)
#mesh=dlfn.refine(mesh)
#space_dim = mesh.geometry().dim()
#n_cells = mesh.num_cells()
# class for periodic boundary conditions
class PeriodicBoundary(dlfn.SubDomain):
    	# Left boundary is "target domain" G
    def inside(self, x, on_boundary):
    	return bool(x[0] <=0.0  and on_boundary)
    
    	# Map right boundary (H) to left boundary (G)
    def map(self, x, y):
    	y[0] = x[0] - 1
    	y[1] = x[1]

# instance for periodic boundary conditions
pbc = PeriodicBoundary()

# == PeriodicBoundary zuweisen ================================================
c = mesh.ufl_cell()
p_deg = 2 			# polynomial degree 
T_elem = dlfn.FiniteElement("CG", c, p_deg - 1)   
V = dlfn.FunctionSpace(mesh, T_elem, constrained_domain = pbc)

K_1= dlfn.Expression('x[1] <= 0.2  ? 1 : 0', degree=1)
K_2= dlfn.Expression('x[1] > 0.2 && x[1] <= 0.25 ? .11 : 0', degree=1)
K_3= dlfn.Expression('x[1] > 0.25 && x[1] <= 0.85 ? 1 : 0', degree=1)
K_4= dlfn.Expression('x[1] > 0.85  ? 0.36 : 0', degree=1)
K=K_1+K_2+K_3+K_4
plot(dlfn.project( K,V),lw=0)

# Create boundary markers
tdim = mesh.topology().dim()
boundary_parts = dlfn.MeshFunction("size_t", mesh, tdim - 1)

unten = dlfn.CompiledSubDomain("near(x[1], 0.0) && on_boundary")
oben = dlfn.CompiledSubDomain("near(x[1], 1.0) && on_boundary")
Kreis_Vorlauf = dlfn.CompiledSubDomain("near(pow(pow(x[0]-.25,2)+pow(x[1]-0.50,2),0.5)<.101, 1) && on_boundary")
Kreis_Nachlauf = dlfn.CompiledSubDomain("near(pow(pow(x[0]-.75,2)+pow(x[1]-0.50,2),0.5)<.101, 1) && on_boundary")
# mark the planar gammas
unten.mark(boundary_parts, 1)
oben.mark(boundary_parts, 2)
# mark the circle boundary
Kreis_Vorlauf.mark(boundary_parts, 3)
Kreis_Nachlauf.mark(boundary_parts, 4)    

dsN_1 = dlfn.Measure("ds", subdomain_id=1, subdomain_data=boundary_parts)
dsN_2 = dlfn.Measure("ds", subdomain_id=2, subdomain_data=boundary_parts)
dsN_3 = dlfn.Measure("ds", subdomain_id=3, subdomain_data=boundary_parts)
dsN_4 = dlfn.Measure("ds", subdomain_id=4, subdomain_data=boundary_parts)

bc = (dlfn.DirichletBC(V,dlfn.Expression('29',Rohrumfang=Rohrumfang,degree = 1), boundary_parts,2),
      dlfn.DirichletBC(V,dlfn.Expression('45',Rohrumfang=Rohrumfang,degree = 1), boundary_parts,3),
      dlfn.DirichletBC(V,dlfn.Expression('40',Rohrumfang=Rohrumfang,degree = 1), boundary_parts,4))
 
# Define variational problem
# https://fenicsproject.org/olddocs/dolfin/1.4.0/python/demo/documented/periodic/python/documentation.html
u = dlfn.TrialFunction(V)
v = dlfn.TestFunction(V)

a = K*dot(grad(u), grad(v))*dlfn.dx
L = ( dlfn.Constant(0.0)*v*dsN_1
    + dlfn.Constant(0.0)*v*dsN_2
    + dlfn.Constant(0.0)*v*dsN_3
    + dlfn.Constant(0.0)*v*dsN_4)


u = dlfn.Function(V)
dlfn.solve(a == L, u,bc)

plot(u,
     isolines={"n": 12, "lw":1, "c":'black', "alpha":0.1},
     lw=0)# no mesh edge lines


from ngsolve import *
from netgen.geom2d import SplineGeometry
#from ngsolve.webgui import Draw
from math import pi
import matplotlib.pyplot as plt
import numpy as np


'''
This code corresponds to Section 6.2 (Figure 5) of the [arXiv preprint](https://arxiv.org/abs/2504.19250)

Tested with NGSolve version 6.2.2404

Wave propagation a homogeneous, isotropic thermoelastic medium
A vertically oriented force source in a square domain
Free boundary conditions are applied on the boundary
Paraview 2D graph representations for temperature and solid velocity
'''

# ********* Parameters ****** #


t = Parameter(0.0)

tend = 0.5001 # end time 1.2
dt = 5e-4 # time step

h = 50#75#mesh size
k = 5 # order of the finite element space

#Biot-Willis coefficient
alpha = 79200

#solid density
rho = 2650

    
# permeability*(dynamic fluid viscosity)**{-1}
beta = 1/10.5
    
#Constrained specific storage (Table 1)
s = 117
    
chi = 1.49e-8/10.5
    
#Lamé coef. corresponding to C
mu  = 6e9
lam = 4e9

    
#needed for A = C**{-1}
a1E = 0.5 / mu
a2E = lam / (4.0 * mu * (lam + mu))

# *** Explosive source *** #

f0 = 5
t0 = 0.3
A0 = 1e4
S = A0*cos(2*pi*f0*(t-t0))*exp(-2*(f0*(t - t0))**2)

sig = h/3 #h/3
ge = exp(-((x - 1155)**2 + (y - 1155)**2)/(2*sig**2))*CoefficientFunction((0, 1.0))

source =  S * ge

# ********* Mesh ****** #

geometry = SplineGeometry()
pnts     = [ (0, 0), (2310,0), (2310, 2310), (0, 2310)]
pnums    = [geometry.AppendPoint(*p) for p in pnts]
#start-point, end-point, boundary-condition, domain on left side, domain on right side:
lines = [ (pnums[0], pnums[1], "open",  1, 0),
          (pnums[1], pnums[2], "open",  1, 0),
          (pnums[2], pnums[3], "open",   1, 0),
          (pnums[3], pnums[0], "open",  1, 0)]

for p1, p2, bc, left, right in lines:
    geometry.Append(["line", p1, p2], bc=bc, leftdomain=left, rightdomain=right)


mesh = Mesh( geometry.GenerateMesh(maxh=h) )

# ********* Finite element spaces ********* #

S = L2(mesh, order =k)
W = VectorL2(mesh, order =k+1)
hatU = VectorFacetFESpace(mesh, order=k+1, dirichlet="open")
hatP = VectorFacetFESpace(mesh, order=k+1)
fes = FESpace([S, S, S, S, W, W, hatU, hatP])

# ********* test and trial functions for product space ****** #

sigmaE1, sigmaE12, sigmaE2,  psi,    u, p, uhat, phat = fes.TrialFunction()
tauE1,   tauE12,   tauE2,    varphi, v, q, vhat, qhat = fes.TestFunction()
    
sigmaE  = CoefficientFunction(( sigmaE1,  sigmaE12,  sigmaE12,  sigmaE2), dims = (2,2) )
sigma = sigmaE - alpha*psi*Id(mesh.dim)

tauE   = CoefficientFunction(( tauE1,   tauE12,   tauE12,   tauE2),   dims = (2,2) )
tau    = tauE  - alpha*varphi*Id(mesh.dim)

    
AsigmaE  = a1E * sigmaE  - a2E * Trace(sigmaE) *  Id(mesh.dim)
    
    
n = specialcf.normal(mesh.dim)
h = specialcf.mesh_size
    
dS = dx(element_boundary=True)
    
jump_u = u - uhat
jump_p = p - phat
    
jump_v = v - vhat
jump_q = q - qhat
    
    
# ********* Bilinear forms ****** #
    
a = BilinearForm(fes, condense=True)
a += (1/dt)*InnerProduct(rho*u, v)*dx      +  (1/dt)*InnerProduct(chi*p, q)*dx
a += (1/dt)*InnerProduct(AsigmaE, tauE)*dx
a += (1/dt)*InnerProduct(s*psi, varphi)*dx
    
a +=   0.5*InnerProduct(beta*p, q)*dx 
    
a +=   0.5*InnerProduct(sigma, grad(v))*dx    - 0.5*InnerProduct(psi, div(q))*dx
a += - 0.5*InnerProduct(sigma*n, jump_v)*dS   + 0.5*InnerProduct(jump_q, psi*n)*dS
    
a += - 0.5*InnerProduct(tau, grad(u))*dx      + 0.5*InnerProduct(varphi, div(p))*dx
a +=   0.5*InnerProduct(tau*n, jump_u)*dS     - 0.5*InnerProduct(jump_p, varphi*n)*dS
    
a +=   0.5*((k+1)**2/h)*jump_u*jump_v*dS + 0.5*((k+1)**2/h)*jump_p*jump_q*dS
    
a.Assemble()

inv_A = a.mat.Inverse(freedofs=fes.FreeDofs(coupling=True))
    
    
M = BilinearForm(fes)
M += (1/dt)*InnerProduct(rho*u, v)*dx +      (1/dt)*InnerProduct(chi*p, q)*dx
M += (1/dt)*InnerProduct(AsigmaE, tauE)*dx 
M += (1/dt)*InnerProduct(s*psi, varphi)*dx
    
M +=  - 0.5*InnerProduct(beta*p, q)*dx 
    
M +=  - 0.5*InnerProduct(sigma, grad(v))*dx    + 0.5*InnerProduct(psi, div(q))*dx
M +=    0.5*InnerProduct(sigma*n, jump_v)*dS   - 0.5*InnerProduct(jump_q, psi*n)*dS
    
M +=   0.5*InnerProduct(tau, grad(u))*dx      - 0.5*InnerProduct(varphi, div(p))*dx
M += - 0.5*InnerProduct(tau*n, jump_u)*dS     + 0.5*InnerProduct(jump_p, varphi*n)*dS
    
M +=  - 0.5*((k+1)**2/h)*jump_u*jump_v*dS - 0.5*((k+1)**2/h)*jump_p*jump_q*dS
    
M.Assemble()
    
# Right-hand side
    
ft = LinearForm(fes)
ft += source * v * dx

# ********* Initial condition ****** #

u0 = GridFunction(fes)
u0.vec[:] = 0.0
#Draw(np.norm(u0.components[3]), mesh, "norm")
Draw( u0.components[3], mesh, "pressure")
Draw( u0.components[4], mesh, "Svelocity")



ft.Assemble()

res = u0.vec.CreateVector()
b0  = u0.vec.CreateVector()
b1  = u0.vec.CreateVector()
    
b0.data = ft.vec

# ********* Time loop ****** #

t_intermediate = dt

while t_intermediate < tend:

    t.Set(t_intermediate)
    ft.Assemble()
    b1.data = ft.vec
     
    res.data = M.mat*u0.vec + 0.5*(b0.data + b1.data)
    
    u0.vec[:] = 0.0

    res.data = res - a.mat * u0.vec
    res.data += a.harmonic_extension_trans * res
    u0.vec.data += inv_A * res
    u0.vec.data += a.inner_solve * res
    u0.vec.data += a.harmonic_extension * u0.vec
    
    b0.data = b1.data
    
    # *** Export data for a graphical representation in Paraview at time steps 0.1, 0.3 and 0.5 *** #
    
    VS = VectorL2(mesh, order=k+1)
    velo = GridFunction(VS)
    velo.Set( u0.components[4])
    
        
    if abs(t_intermediate - 0.1) < 1e-5:
        vtk = VTKOutput(ma=mesh, coefs=[u0.components[3], velo, sqrt( (u0.components[0] - alpha*u0.components[3])**2 + (u0.components[2] - alpha*u0.components[3])**2 \
            -(u0.components[0] - alpha*u0.components[3])*(u0.components[2] - alpha*u0.components[3]) + \
                3*u0.components[1]**2)], names = ["pressure", "velocity", "stress"], filename="thermoE01", subdivision=3)
        vtk.Do()
            
    if abs(t_intermediate - 0.3) < 1e-5:
        vtk = VTKOutput(ma=mesh, coefs=[u0.components[3], velo, sqrt( (u0.components[0] - alpha*u0.components[3])**2 + (u0.components[2] - alpha*u0.components[3])**2 \
            -(u0.components[0] - alpha*u0.components[3])*(u0.components[2] - alpha*u0.components[3]) + \
                3*u0.components[1]**2)], names = ["pressure", "velocity", "stress"], filename="thermoE03", subdivision=3)
        vtk.Do()
    
    if abs(t_intermediate - 0.5) < 1e-5:
        vtk = VTKOutput(ma=mesh, coefs=[u0.components[3], velo, sqrt( (u0.components[0] - alpha*u0.components[3])**2 + (u0.components[2] - alpha*u0.components[3])**2 \
            -(u0.components[0] - alpha*u0.components[3])*(u0.components[2] - alpha*u0.components[3]) + \
                3*u0.components[1]**2)], names = ["pressure", "velocity", "stress"], filename="thermoE05", subdivision=3)
        vtk.Do()
    
    t_intermediate += dt
    
    print("\r",t_intermediate,end="")
    Redraw(blocking=True)
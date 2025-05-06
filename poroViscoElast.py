from ngsolve import *
from netgen.geom2d import SplineGeometry
#from ngsolve.webgui import Draw
from math import pi
import matplotlib.pyplot as plt
import numpy as np


'''
This code corresponds to Section 6.2 (Figures 6 & 7) of the [arXiv preprint](https://arxiv.org/abs/2504.19250)

Tested with NGSolve version 6.2.2404

wave propagation within a hybrid medium comprising both poro-viscoelastic and purely poroelastic regions
An explosive source located at the center of a square domain
Free boundary conditions are applied on the boundary
Paraview 2D graph representations for pressure (Figure 6) and solid velocity (Figure 7)
'''


# ********* Geometry and mesh ********* #

geo = SplineGeometry()
L = 1155
pnts = [ (-L,0), (0,0), (L,0), (L,2*L), (0,2*L), (-L,2*L) ]
pind = [ geo.AppendPoint(*pnt) for pnt in pnts ]

geo.Append(['line',pind[0],pind[1]],leftdomain=1,rightdomain=0,bc="bottom")
geo.Append(['line',pind[1],pind[2]],leftdomain=2,rightdomain=0,bc="bottom")
geo.Append(['line',pind[2],pind[3]],leftdomain=2,rightdomain=0,bc="right")
geo.Append(['line',pind[3],pind[4]],leftdomain=2,rightdomain=0,bc="top")
geo.Append(['line',pind[4],pind[5]],leftdomain=1,rightdomain=0,bc="top")
geo.Append(['line',pind[5],pind[0]],leftdomain=1,rightdomain=0,bc="left")
geo.Append(['line',pind[1],pind[4]],leftdomain=1,rightdomain=2,bc="interface")

geo.SetMaterial(1, "left_dom")
geo.SetMaterial(2, "right_dom")

h = 50
k = 5
mesh = Mesh( geo.GenerateMesh(maxh=h) )

# ********* Parameters ****** #


t = Parameter(0.0)

tend = 0.6001 # end time 1.2
dt = 5e-4 # time step

#Biot-Willis coefficient
alpha = 1

phi = 0.1
rhoS = 2650
rhoF = 1025

#solid density
rho = phi*rhoF + (1-phi)*rhoS
    
#relaxation time
omega = 0.9
    
# permeability*(dynamic fluid viscosity)**{-1}
kappa = 1e-14 

nu = 0.001
beta = nu/kappa
    
#Constrained specific storage (Table 1)
s = 1e-9
    
chi = rhoF/phi
    
#Lamé coef. corresponding to C
mu  = 1e9
lam = 4e8

#EE = mu*(3*lam + 2*mu)/(lam + mu)
#omegaV = omega/EE
    
#Lamé coef. corresponding to D
muV  = 4e9#4.3738e9
lamV = 7e9#7.2073e9
    
muD  = muV - mu
lamD = lamV - lam
    
#needed for A = C**{-1}
a1E = 0.5 / mu
a2E = lam / (4.0 * mu * (lam + mu))
    
#needed for V = (D - C)**{-1}
a1D = 0.5 / muD
a2D = lamD / (4.0 * muD * (lamD + muD))

# *** Explosive source *** #

f0 = 5
t0 = 0.3
A0 = 1e4
ss = A0*cos(2*pi*f0*(t-t0))*exp(-2*(f0*(t - t0))**2)

dil = 2*h
rad = CoefficientFunction(sqrt(x**2 + (y - 1155)**2))
ge = (1 - rad**2/dil**2)*CoefficientFunction((x/rad, (y - 1155)/rad))*IfPos( dil - rad, 1.0, 0.0 )

BB = sqrt(2)*CoefficientFunction((0, 1, 1, 0), dims=(2, 2))/2

source =  ss * (BB*ge)



# ********* Finite element spaces ********* #

S = L2(mesh, order =k)
SV = L2(mesh, order =k, definedon = "left_dom")
W = VectorL2(mesh, order =k+1)
hatU = VectorFacetFESpace(mesh, order=k+1, dirichlet="top|left|bottom|right")
hatP = VectorFacetFESpace(mesh, order=k+1)
fes = FESpace([S, S, S, SV, SV, SV, S, W, W, hatU, hatP])

# ********* test and trial functions for product space ****** #

sigmaE1, sigmaE12, sigmaE2, sigmaV1, sigmaV12, sigmaV2, psi,    u, p, uhat, phat = fes.TrialFunction()
tauE1,   tauE12,   tauE2,   tauV1,   tauV12,   tauV2,   varphi, v, q, vhat, qhat = fes.TestFunction()
    
sigmaE  = CoefficientFunction(( sigmaE1,  sigmaE12,  sigmaE12,  sigmaE2), dims = (2,2) )
sigmaV  = CoefficientFunction(( sigmaV1,  sigmaV12,  sigmaV12,  sigmaV2), dims = (2,2) )
sigma = sigmaE + omega*sigmaV - alpha*psi*Id(mesh.dim)
    
tauE   = CoefficientFunction(( tauE1,   tauE12,   tauE12,   tauE2),   dims = (2,2) )
tauV   = CoefficientFunction(( tauV1,   tauV12,   tauV12,   tauV2),   dims = (2,2) )
tau    = tauE + omega*tauV - alpha*varphi*Id(mesh.dim)
    
AsigmaE  = a1E * sigmaE  - a2E * Trace(sigmaE) *  Id(mesh.dim)
VsigmaV  = a1D * sigmaV  - a2D * Trace(sigmaV)  * Id(mesh.dim)
    
    
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
a += (1/dt)*InnerProduct(AsigmaE, tauE)*dx +  (1/dt)*InnerProduct(omega*VsigmaV, omega*tauV)*dx("left_dom") 
a += (1/dt)*InnerProduct(s*psi, varphi)*dx
    
a +=   0.5*InnerProduct(beta*p, q)*dx + 0.5*InnerProduct(VsigmaV, omega*tauV)*dx("left_dom")
    
a +=   0.5*InnerProduct(sigma, grad(v))*dx    - 0.5*InnerProduct(psi, div(q))*dx
a += - 0.5*InnerProduct(sigma*n, jump_v)*dS   + 0.5*InnerProduct(jump_q, psi*n)*dS
    
a += - 0.5*InnerProduct(tau, grad(u))*dx      + 0.5*InnerProduct(varphi, div(p))*dx
a +=   0.5*InnerProduct(tau*n, jump_u)*dS     - 0.5*InnerProduct(jump_p, varphi*n)*dS
    
a +=   0.5*((k+1)**2/h)*jump_u*jump_v*dS + 0.5*((k+1)**2/h)*jump_p*jump_q*dS
    
a.Assemble()

inv_A = a.mat.Inverse(freedofs=fes.FreeDofs(coupling=True))
    
    
M = BilinearForm(fes)
M += (1/dt)*InnerProduct(rho*u, v)*dx +      (1/dt)*InnerProduct(chi*p, q)*dx
M += (1/dt)*InnerProduct(AsigmaE, tauE)*dx + (1/dt)*InnerProduct(omega*VsigmaV, omega*tauV)*dx("left_dom")
M += (1/dt)*InnerProduct(s*psi, varphi)*dx
    
M +=  - 0.5*InnerProduct(beta*p, q)*dx - 0.5*InnerProduct(VsigmaV, omega*tauV)*dx("left_dom")
    
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
Draw( u0.components[6], mesh, "pressure")
Draw( u0.components[7], mesh, "Svelocity")



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
    velo.Set( u0.components[7])
    
        
    if abs(t_intermediate - 0.2) < 1e-5:
        vtk = VTKOutput(ma=mesh, coefs=[u0.components[6], velo, sqrt( 
        (u0.components[0] + omega*u0.components[3] - alpha*u0.components[6])**2 +
        (u0.components[2] + omega*u0.components[5] - alpha*u0.components[6])**2 -\
        (u0.components[0] + omega*u0.components[3] - alpha*u0.components[3])*
        (u0.components[2] + omega*u0.components[5] - alpha*u0.components[3]) + \
        3*(u0.components[1] + omega*u0.components[4])**2)], names = ["pressure", "velocity", "stress"], filename="poroVE02", subdivision=3)
        vtk.Do()
            
    if abs(t_intermediate - 0.4) < 1e-5:
        vtk = VTKOutput(ma=mesh, coefs=[u0.components[6], velo, sqrt( 
        (u0.components[0] + omega*u0.components[3] - alpha*u0.components[6])**2 +
        (u0.components[2] + omega*u0.components[5] - alpha*u0.components[6])**2 -\
        (u0.components[0] + omega*u0.components[3] - alpha*u0.components[3])*
        (u0.components[2] + omega*u0.components[5] - alpha*u0.components[3]) + \
        3*(u0.components[1] + omega*u0.components[4])**2)], names = ["pressure", "velocity", "stress"], filename="poroVE04", subdivision=3)
        vtk.Do()
    
    if abs(t_intermediate - 0.6) < 1e-5:
        vtk = VTKOutput(ma=mesh, coefs=[u0.components[6], velo, sqrt( 
        (u0.components[0] + omega*u0.components[3] - alpha*u0.components[6])**2 +
        (u0.components[2] + omega*u0.components[5] - alpha*u0.components[6])**2 -\
        (u0.components[0] + omega*u0.components[3] - alpha*u0.components[3])*
        (u0.components[2] + omega*u0.components[5] - alpha*u0.components[3]) + \
        3*(u0.components[1] + omega*u0.components[4])**2)], names = ["pressure", "velocity", "stress"], filename="poroVE06", subdivision=3)
        vtk.Do()
    
    t_intermediate += dt
    
    print("\r",t_intermediate,end="")
    Redraw(blocking=True)
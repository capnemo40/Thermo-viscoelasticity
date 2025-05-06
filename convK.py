from netgen.geom2d import unit_square
from ngsolve import *
from math import pi
ngsglobals.msg_level = 2

'''
This code corresponds to Section 6.1 (Figure 3) of the [arXiv preprint](https://arxiv.org/abs/2504.19250)


Tested with NGSolve version 6.2.2404

k-Convergence test for fully transient poro-viscoelastic problem
Manufactured solutions on a squared domain
Pure Dirichlet BCs
HDG approximation in terms of stress, pressure & solid/fluid velocities
Crank-Nicolson scheme for time discretization
'''

def weaksym(k, h, dt, tend):
    
    # k: polynomial degree
    # h: mesh size
    # dt: time step
    # tend: final time

    # ********* Model coefficients and parameters ********* #
    
    t = Parameter(0.0)
    
    
    #Biot-Willis coefficient
    alpha = 1

    #solid density
    rho = 1
    
    #relaxation time
    omega = 1
    
    # permeability*(dynamic fluid viscosity)**{-1}
    beta = 1 
    
    #Constrained specific storage (Table 1)
    s = 1
    
    chi = 1
    
    #Lamé coef. corresponding to C
    mu  = 10
    lam = 30
    
    #Lamé coef. corresponding to D
    muV  = 20
    lamV = 40
    
    muD  = muV - mu
    lamD = lamV - lam
    
    #needed for A = C**{-1}
    a1E = 0.5 / mu
    a2E = lam / (4.0 * mu * (lam + mu))
    
    #needed for V = (D - C)**{-1}
    a1D = 0.5 / muD
    a2D = lamD / (4.0 * muD * (lamD + muD))
    

    # ******* Exact solutions for error analysis ****** #
    
    #Exact displacement d
    exactd0 = y*sin(pi*x)*cos(pi*y)*sin(2*pi*t)
    exactd1 = x*cos(pi*x)*sin(pi*y)*sin(2*pi*t)
    exactd = CoefficientFunction((exactd0, exactd1))
    
    #Exact solid velocity u
    exactu0 = 2*pi*y*cos(2*pi*t)*cos(pi*y)*sin(pi*x)
    exactu1 = 2*pi*x*cos(2*pi*t)*cos(pi*x)*sin(pi*y)
    exactu = CoefficientFunction((exactu0, exactu1))
    
    #Exact elastic Stress sigmaE
    sigmaE00 = pi*cos(pi*x)*cos(pi*y)*sin(2*pi*t)*(lam*x + lam*y + 2*mu*y)
    sigmaE01 = mu*sin(2*pi*t)*(cos(pi*x)*sin(pi*y) + cos(pi*y)*sin(pi*x) - pi*x*sin(pi*x)*sin(pi*y) - pi*y*sin(pi*x)*sin(pi*y))
    sigmaE11 = pi*cos(pi*x)*cos(pi*y)*sin(2*pi*t)*(lam*x + lam*y + 2*mu*x)
    exactSigmaE = CoefficientFunction((sigmaE00, sigmaE01, sigmaE01, sigmaE11), dims = (2,2))
    
    #Exact viscous Stress sigmaV: \dot{sigmaV} + sigmaV\omega =  (\cD - \cC) \beps(\bu)
    sigmaV00 = (pi**2*cos(pi*x + pi*y)*(2*pi*sin(2*pi*t) + cos(2*pi*t)/omega)*(lamD*x + lamD*y + 2*muD*y))/(omega*(1/omega**2 + 4*pi**2)) - exp(-t/omega)*((pi**2*cos(pi*x + pi*y)*(lamD*x + lamD*y + 2*muD*y))/(omega**2*(1/omega**2 + 4*pi**2)) + (pi**2*cos(pi*x - pi*y)*(lamD*x + lamD*y + 2*muD*y))/(omega**2*(1/omega**2 + 4*pi**2))) + (pi**2*cos(pi*x - pi*y)*(2*pi*sin(2*pi*t) + cos(2*pi*t)/omega)*(lamD*x + lamD*y + 2*muD*y))/(omega*(1/omega**2 + 4*pi**2))
    sigmaV01 = (muD*pi*exp(-t/omega)*(exp(t/omega)*cos(2*pi*t) + 2*omega*pi*exp(t/omega)*sin(2*pi*t) - 1)*(2*sin(pi*x + pi*y) + pi*x*cos(pi*x + pi*y) - pi*x*cos(pi*x - pi*y) + pi*y*cos(pi*x + pi*y) - pi*y*cos(pi*x - pi*y)))/(4*pi**2*omega**2 + 1)
    sigmaV11 = (pi**2*cos(pi*x + pi*y)*(2*pi*sin(2*pi*t) + cos(2*pi*t)/omega)*(lamD*x + lamD*y + 2*muD*x))/(omega*(1/omega**2 + 4*pi**2)) - exp(-t/omega)*((pi**2*cos(pi*x + pi*y)*(lamD*x + lamD*y + 2*muD*x))/(omega**2*(1/omega**2 + 4*pi**2)) + (pi**2*cos(pi*x - pi*y)*(lamD*x + lamD*y + 2*muD*x))/(omega**2*(1/omega**2 + 4*pi**2))) + (pi**2*cos(pi*x - pi*y)*(2*pi*sin(2*pi*t) + cos(2*pi*t)/omega)*(lamD*x + lamD*y + 2*muD*x))/(omega*(1/omega**2 + 4*pi**2))
    exactSigmaV = CoefficientFunction((sigmaV00, sigmaV01, sigmaV01, sigmaV11), dims = (2,2))
    
    #Fluid pressure/temperature
    exactpsi = sin(pi*x)*sin(pi*y)*cos(2*pi*t)
    
    #Filtartion velocity/heat flux
    p_0 = (exp(-(beta*t)/chi)*(beta*pi*sin(pi*x + pi*y) - beta*pi*sin(pi*x - pi*y)))/(8*pi**2*chi**2 + 2*beta**2) - (pi*sin(pi*x + pi*y)*(2*pi*sin(2*pi*t) + (beta*cos(2*pi*t))/chi))/(2*chi*(4*pi**2 + beta**2/chi**2)) + (pi*sin(pi*x - pi*y)*(2*pi*sin(2*pi*t) + (beta*cos(2*pi*t))/chi))/(2*chi*(4*pi**2 + beta**2/chi**2))
    
    p_1 = (exp(-(beta*t)/chi)*(beta*pi*sin(pi*x + pi*y) + beta*pi*sin(pi*x - pi*y)))/(8*pi**2*chi**2 + 2*beta**2) - (pi*sin(pi*x + pi*y)*(2*pi*sin(2*pi*t) + (beta*cos(2*pi*t))/chi))/(2*chi*(4*pi**2 + beta**2/chi**2)) - (pi*sin(pi*x - pi*y)*(2*pi*sin(2*pi*t) + (beta*cos(2*pi*t))/chi))/(2*chi*(4*pi**2 + beta**2/chi**2))
    
    exactp = CoefficientFunction((p_0, p_1))
    
    #Source term F
    F0 = mu*pi*sin(2*pi*t)*(2*sin(pi*x)*sin(pi*y) - cos(pi*x)*cos(pi*y) + pi*x*cos(pi*y)*sin(pi*x) + pi*y*cos(pi*y)*sin(pi*x)) + alpha*pi*cos(2*pi*t)*cos(pi*x)*sin(pi*y) - lam*pi*cos(pi*x)*cos(pi*y)*sin(2*pi*t) + pi**2*cos(pi*y)*sin(2*pi*t)*sin(pi*x)*(lam*x + lam*y + 2*mu*y) - 4*pi**2*rho*y*cos(pi*y)*sin(2*pi*t)*sin(pi*x) + (2*muD*omega*pi**2*exp(-t/omega)*(exp(t/omega)*cos(2*pi*t) + 2*omega*pi*exp(t/omega)*sin(2*pi*t) - 1)*(2*sin(pi*x)*sin(pi*y) - cos(pi*x)*cos(pi*y) + pi*x*cos(pi*y)*sin(pi*x) + pi*y*cos(pi*y)*sin(pi*x)))/(4*pi**2*omega**2 + 1) + (2*omega*pi**2*exp(-t/omega)*cos(pi*y)*(exp(t/omega)*cos(2*pi*t) + 2*omega*pi*exp(t/omega)*sin(2*pi*t) - 1)*(lamD*pi*x*sin(pi*x) - lamD*cos(pi*x) + lamD*pi*y*sin(pi*x) + 2*muD*pi*y*sin(pi*x)))/(4*pi**2*omega**2 + 1)
    
    F1 = mu*pi*sin(2*pi*t)*(2*sin(pi*x)*sin(pi*y) - cos(pi*x)*cos(pi*y) + pi*x*cos(pi*x)*sin(pi*y) + pi*y*cos(pi*x)*sin(pi*y)) + alpha*pi*cos(2*pi*t)*cos(pi*y)*sin(pi*x) - lam*pi*cos(pi*x)*cos(pi*y)*sin(2*pi*t) + pi**2*cos(pi*x)*sin(2*pi*t)*sin(pi*y)*(lam*x + lam*y + 2*mu*x) - 4*pi**2*rho*x*cos(pi*x)*sin(2*pi*t)*sin(pi*y) + (2*muD*omega*pi**2*exp(-t/omega)*(exp(t/omega)*cos(2*pi*t) + 2*omega*pi*exp(t/omega)*sin(2*pi*t) - 1)*(2*sin(pi*x)*sin(pi*y) - cos(pi*x)*cos(pi*y) + pi*x*cos(pi*x)*sin(pi*y) + pi*y*cos(pi*x)*sin(pi*y)))/(4*pi**2*omega**2 + 1) + (2*omega*pi**2*exp(-t/omega)*cos(pi*x)*(exp(t/omega)*cos(2*pi*t) + 2*omega*pi*exp(t/omega)*sin(2*pi*t) - 1)*(lamD*pi*x*sin(pi*y) - lamD*cos(pi*y) + lamD*pi*y*sin(pi*y) + 2*muD*pi*x*sin(pi*y)))/(4*pi**2*omega**2 + 1)
    
    F = CoefficientFunction( (F0, F1) )
    
    #Source term g
    g = (pi**2*cos(pi*(x - y))*(beta*cos(2*pi*t) + 2*chi*pi*sin(2*pi*t)))/(4*pi**2*chi**2 + beta**2) - (pi**2*cos(pi*(x + y))*(beta*cos(2*pi*t) + 2*chi*pi*sin(2*pi*t)))/(4*pi**2*chi**2 + beta**2) - 2*pi*s*sin(2*pi*t)*sin(pi*x)*sin(pi*y) + 2*alpha*pi**2*x*cos(2*pi*t)*cos(pi*x)*cos(pi*y) + 2*alpha*pi**2*y*cos(2*pi*t)*cos(pi*x)*cos(pi*y) - (4*beta*pi**2*exp(-(beta*t)/chi)*sin(pi*x)*sin(pi*y))/(8*pi**2*chi**2 + 2*beta**2)
        
    # ******* Mesh of the unit square ****** #

    mesh = Mesh(unit_square.GenerateMesh(maxh=h))

    # ********* Finite element spaces ********* #

    S = L2(mesh, order =k)
    W = VectorL2(mesh, order =k+1)
    hatU = VectorFacetFESpace(mesh, order=k+1, dirichlet="bottom|left|right|top")
    hatP = VectorFacetFESpace(mesh, order=k+1)
    fes = FESpace([S, S, S, S, S, S, S, W, W, hatU, hatP])
    
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
    a += (1/dt)*InnerProduct(AsigmaE, tauE)*dx +  (1/dt)*InnerProduct(omega*VsigmaV, omega*tauV)*dx 
    a += (1/dt)*InnerProduct(s*psi, varphi)*dx
    
    a +=   0.5*InnerProduct(beta*p, q)*dx + 0.5*InnerProduct(VsigmaV, omega*tauV)*dx
    
    a +=   0.5*InnerProduct(sigma, grad(v))*dx    - 0.5*InnerProduct(psi, div(q))*dx
    a += - 0.5*InnerProduct(sigma*n, jump_v)*dS   + 0.5*InnerProduct(jump_q, psi*n)*dS
    
    a += - 0.5*InnerProduct(tau, grad(u))*dx      + 0.5*InnerProduct(varphi, div(p))*dx
    a +=   0.5*InnerProduct(tau*n, jump_u)*dS     - 0.5*InnerProduct(jump_p, varphi*n)*dS
    
    a +=   0.5*((k+1)**2/h)*jump_u*jump_v*dS + 0.5*((k+1)**2/h)*jump_p*jump_q*dS
    
    a.Assemble()

    inv_A = a.mat.Inverse(freedofs=fes.FreeDofs(coupling=True))
    
    
    M = BilinearForm(fes)
    M += (1/dt)*InnerProduct(rho*u, v)*dx +      (1/dt)*InnerProduct(chi*p, q)*dx
    M += (1/dt)*InnerProduct(AsigmaE, tauE)*dx + (1/dt)*InnerProduct(omega*VsigmaV, omega*tauV)*dx
    M += (1/dt)*InnerProduct(s*psi, varphi)*dx
    
    M +=  - 0.5*InnerProduct(beta*p, q)*dx - 0.5*InnerProduct(VsigmaV, omega*tauV)*dx
    
    M +=  - 0.5*InnerProduct(sigma, grad(v))*dx    + 0.5*InnerProduct(psi, div(q))*dx
    M +=    0.5*InnerProduct(sigma*n, jump_v)*dS   - 0.5*InnerProduct(jump_q, psi*n)*dS
    
    M +=   0.5*InnerProduct(tau, grad(u))*dx      - 0.5*InnerProduct(varphi, div(p))*dx
    M += - 0.5*InnerProduct(tau*n, jump_u)*dS     + 0.5*InnerProduct(jump_p, varphi*n)*dS
    
    M +=  - 0.5*((k+1)**2/h)*jump_u*jump_v*dS - 0.5*((k+1)**2/h)*jump_p*jump_q*dS
    
    M.Assemble()
    
    # Right-hand side

    ft = LinearForm(fes)
    ft += F * v * dx
    ft += - exactpsi*(qhat.Trace()*n) *ds(definedon=mesh.Boundaries("bottom|left|right|top"))
    ft += g * varphi * dx

    # ********* instantiation of initial conditions ****** #
    
    u0 = GridFunction(fes)
    u0.components[0].Set(exactSigmaE[0,0])
    u0.components[1].Set(exactSigmaE[0,1])
    u0.components[2].Set(exactSigmaE[1,1])
    u0.components[3].Set(exactSigmaV[0,0])
    u0.components[4].Set(exactSigmaV[0,1])
    u0.components[5].Set(exactSigmaV[1,1])
    u0.components[6].Set(exactpsi)
    u0.components[7].Set(exactu)
    u0.components[8].Set(exactp)
    u0.components[9].Set(exactu, dual=True)
    u0.components[10].Set(exactp, dual=True)

    
    ft.Assemble()
    
    res = u0.vec.CreateVector()
    b0  = u0.vec.CreateVector()
    b1  = u0.vec.CreateVector()
    
    b0.data = ft.vec

    t_intermediate = dt # time counter within one block-run
    
    # ********* Time loop ************* # 

    while t_intermediate < tend:

        t.Set(t_intermediate)
        ft.Assemble()
        b1.data = ft.vec
     
        res.data = M.mat*u0.vec + 0.5*(b0.data + b1.data)

        u0.vec[:] = 0.0 
        u0.components[9].Set(exactu, BND)

        res.data = res - a.mat * u0.vec

        res.data += a.harmonic_extension_trans * res

        u0.vec.data += inv_A * res
        
        u0.vec.data += a.inner_solve * res
        u0.vec.data += a.harmonic_extension * u0.vec
        
        b0.data = b1.data
        t_intermediate += dt
        
        print('t=%g' % t_intermediate)
        

    # ********* L2-errors at time tend ****** #
    
    gfsigmaE1, gfsigmaE12, gfsigmaE2, gfsigmaV1, gfsigmaV12, gfsigmaV2, gfpsi, gfu, gfp = u0.components[0:9]

    gfsigmaE = CoefficientFunction(( gfsigmaE1, gfsigmaE12, gfsigmaE12, gfsigmaE2), dims = (2,2) )
    gfsigmaV = CoefficientFunction(( gfsigmaV1, gfsigmaV12, gfsigmaV12, gfsigmaV2), dims = (2,2) )

    # Solid/fluid velocities error
    norm_H1  = InnerProduct(rho*(gfu - exactu), gfu - exactu)
    norm_H1 += InnerProduct(chi*(gfp - exactp), gfp - exactp)
    norm_H1  = Integrate(norm_H1, mesh)
    norm_H1  = sqrt(norm_H1)
    
    
    
    # Stess-pressure/temperature error
    norm_H2  = InnerProduct(a1E*(exactSigmaE - gfsigmaE)  - a2E*Trace(exactSigmaE  - gfsigmaE)*  Id(mesh.dim), exactSigmaE - gfsigmaE)
    norm_H2 += InnerProduct(a1D*(exactSigmaV  - gfsigmaV) - a2D*Trace(exactSigmaV  - gfsigmaV)*  Id(mesh.dim), omega**2*(exactSigmaV - gfsigmaV))
    norm_H2 += InnerProduct(s*(exactpsi - gfpsi), exactpsi - gfpsi)
    norm_H2 = Integrate(norm_H2, mesh)
    norm_H2 = sqrt(norm_H2)

    return norm_H2, norm_H1

# ********* Error collector ************* # 

def collecterrors(maxk, h, dt, tend):
    l2e_s = []
    l2e_r = []
    for k in range(0, maxk):
        er_1, er_2 = weaksym(k, h, dt, tend)
        l2e_s.append(er_1)
        l2e_r.append(er_2)
    return l2e_s, l2e_r


# ********* Convergence table ************* # 

def hconvergenctauEble(e_1, e_2, maxk):
    print("============================================================")
    print(" k   Errors_s   Error_u   ")
    print("------------------------------------------------------------")
    
    for i in range(maxk):
        print(" %-4d %8.2e    %8.2e   " % (i, e_1[i], 
               e_2[i]))

    print("============================================================")


# ============= MAIN DRIVER ==============================

maxk = 7 #number of k refinements
dt = 10e-5 #time step
tend = 0.3 #final time
h = 1/5 #mesh size

er_s, er_u = collecterrors(maxk, h, dt, tend)
hconvergenctauEble(er_s, er_u, maxk)
import numpy as np
import time
import matplotlib.pyplot as plt


def SSGW(kd,kH2, N=2048, tol=1e-14):
    """
    SSGW: Steady Surface Gravity Waves.
    Computation of irrotational 2D periodic surface pure gravity waves 
    of arbitrary length in arbitrary depth. 

    MANDATORY INPUT PARAMETERS:
    kd  = k*d   : relative depth (wavenumber "k" times mean water depth "d").
    kH2 = k*H/2 : steepness (half the total wave height "H" times the wavenumber "k").

    OPTIONAL INPUT PARAMETERS:
    N   : number of positive Fourier modes (default, N=2048).
    tol : tolerance (default, tol=1e-14).

    OUTPUT PARAMETERS:
    zs  = complex abscissas at the free surface (at the computational nodes).
    ws  = complex velocity at the free surface (at the computational nodes).
    PP  = Physical Parameters: PP(1)=depth, PP(2)=wavenumber, PP(3)=wavelenght, 
    PP(4)=celerity c_e, PP(5)=celerity c_s, PP(6)=Bernoulli constant, 
    PP(7)=crest height, PP(8)=trough height, PP(9)=impulse, 
    PP(10)=potential energy, pp(11)=kinetic energy, PP(12)=radiation stress,
    PP(13)=momentum flux, PP(14)=energy flux, PP(16)=group velocity.

    NOTE: The output quantities are dimensionless with the following scaling.
    In deep water:   rho = g = k = 1.
    In finite depth: rho = g = d = 1.

    EXAMPLE 1. To compute a wave of steepness kH2=0.3 in infinite depth:
    [zs,ws,PP]=SSGW(np.inf,0.3)

    EXAMPLE 2. To compute a cnoidal wave with height-over-depth=0.5 and 
    length-over-depth=100:
    Hd=0.5; Ld=100; kd=2*np.pi/Ld; kH2=np.pi*Hd/Ld; [zs,ws,PP]=SSGW(kd,kH2)

    EXAMPLE 3. For steep and long waves, the default number of Fourier modes
    must be increased. For instance, in order to compute a cnoidal wave with 
    height-over-depth=0.7 and length-over-depth=10000:
    Hd=0.7; Ld=10000; kd=2*np.pi/Ld; kH2=np.pi*Hd/Ld; [zs,ws,PP]=SSGW(kd,kH2,2^19)

    The program works for all but the (almost) highest waves.
    Edit the m-file for more details.

    For details of the algorithm and the notations, read:
    Clamond, D. & Dutykh, D. 2017. Accurate fast computation of steady 
    two-dimensional surface gravity waves in arbitrary depth. Arxiv.

    This m-file was written with the purpose of clarity. The notations closely 
    match those of the paper above.

    Authors: D. Clamond & D. Dutykh.
    Version: 2017-02-08.

    Converted to Python by Andreas Amann 2018-02-23

    -------------------------------------------------------------------------
    """
    
    # Check input parameters.
    if kd<0 or kH2<0:
        print('Input scalar parameters kd and kH2 must be real and positive.')
        raise ValueError

    # Determine depth and choose parameters.
    if 1-np.tanh(kd) < tol:                                                        # Deep water case.
        d   = np.inf                                                               # Depth.
        k   = 1                                                                    # Wavenumber.
        g   = 1                                                                    # Acceleration due to gravity.
        lam = 1/k                                                                  # Characteristic wavelength lambda.
    else:                                                                          # Finite depth case.
        d   = 1                                                                    # Depth.
        k   = kd/d                                                                 # Wavenumber.
        g   = 1                                                                    # Acceleration due to gravity.
        lam = np.tanh(kd)/k                                                        # Characteristic wavelength lambda.

    c02 = g*lam                                                                    # Linear phase velocity squared.
    H   = 2*kH2/k                                                                  # Total wwave height.
    L   = np.pi/k                                                                  # Half-length of the computational domain (with c_r=c_e).
    dal = L/N                                                                      # Delta alpha.
    dk  = np.pi/L                                                                  # Delta k.

    # Vectors.
    va  = np.arange(2*N)*dal                                                       # Vector of abscissas in the conformal space.
    vk  = np.array(list(range(N))+list(range(-N,0)))*dk                            # Vector of wavenumbers.

    # Initial guess for the solution:
    Ups = (H/2)*(1+np.cos(k*va))                                                   # Airy solution for Upsilon.
    sig = 1                                                                        # Parameter sigma.

    # Commence Petviashvili's iterations.
    err  = np.inf                                                                  # Enforce loop entry.
    iter = 0                                                                       # Iterations counter.
    tic = time.time()                                                              # Start clocking.
    while (err > tol):
        # Compute sigma and delta.      
        mUps = Ups.mean()                                                          # << Upsilon >>.
        Ys   = Ups - mUps                                                          # Y_s.
        if d == np.inf:                                                            # Deep water.
            sig = 1                                                                # sigma.
            CYs = np.fft.ifft(abs(vk)*np.fft.fft(Ys)).real                         # C{ Y_s }.
            mys = -np.dot(Ys,CYs)/N/2                                              # << y_s >>.
        else:                                                                      # Finite depth.
            C_hat  =  vk*1/np.tanh((sig*d)*vk);      C_hat[0]  = 1/(sig*d)         # Operator C in Fourier space. # AA: Can this line be deleted?
            S2_hat = (vk*1/np.sinh((sig*d)*vk))**2;  S2_hat[0] = 1/(sig*d)**2      # Operator S^2 in Fourier space.
            Ys_hat  = np.fft.fft(Ys)                                               
            E  = (Ys*np.fft.ifft(C_hat*Ys_hat).real).mean() + (sig-1)*d            # Equation for sigma.
            dE = d - d*(Ys*np.fft.ifft(S2_hat*Ys_hat).real).mean()                 # Its derivative.
            sig = sig - E/dE                                                       # Newton new sigma.
            mys = (sig-1)*d                                                        # << y_s >>.
        del_ = mys - mUps                                                          # Parameter delta.
        C_hat  =  vk*1/np.tanh((sig*d)*vk);  C_hat[0] = 1/(sig*d)                  # Updated operator C in Fourier space. 
 
        # Compute Bernoulli constant B.
        Ups2  = Ups*Ups                                                            # Upsilon^2.
        mUps2 = Ups2.mean()                                                        # << Upsilon^2 >>.
        CUps  = np.fft.ifft(C_hat*np.fft.fft(Ups)).real                            # C{ Upsilon }.
        CUps2 = np.fft.ifft(C_hat*np.fft.fft(Ups2)).real                           # C{ Upsilon^2 }.
        DCU   = CUps[N] -  CUps[0]                                                 # C{ Upsilon }_trough - C{ Upsilon }_crest.
        DCU2  = CUps2[N] - CUps2[0]                                                # C{ Upsilon^2 }_trough - C{ Upsilon^2 }_crest.
        Bg    = 2*del_ - H/sig*(1+del_/d+sig*CUps[0])/DCU + DCU2/DCU/2             # B/g.
  
        # Define linear operators in Fourier space.
        Cinf_hat = abs(vk);  Cinf_hat[0] = 0                                       # Operator C_inf.  
        CIC_hat  = np.tanh((sig*d)*abs(vk))                                        # Operator C_inf o C^{-1}.      
        if d==np.inf:
            CIC_hat[0] = 1                                                         # Regularisation.
        L_hat    = (Bg-2*del_)*Cinf_hat - ((1+del_/d)/sig)*CIC_hat                 # Operator L.
        IL_hat   = 1./L_hat;  IL_hat[0] = 1                                        # Operator L^-1.
 
        # Petviashvili's iteration.
        Ups_hat = np.fft.fft(Ups)                                                  # Fourier transform of Upsilon.
        CUps_hat = C_hat*Ups_hat
        LUps = np.fft.ifft(L_hat*Ups_hat).real                                     # L{Upsilon}.
        Ups2_hat = np.fft.fft(Ups*Ups)                                             # Fourier transform of Upsilon^2.
        NUps_hat = CIC_hat*np.fft.fft(Ups*np.fft.ifft(CUps_hat).real)
        NUps_hat = NUps_hat + Cinf_hat*Ups2_hat/2                                  # Nonlinear term in Fourier space.
        NUps = np.fft.ifft(NUps_hat).real                                          # N{ Upsilon }.
        S = np.dot(Ups,LUps)/np.dot(Ups,NUps)                                      # Weight.
        U = S*S*np.fft.ifft(NUps_hat*IL_hat).real                                  # New Upsilon.
        U = H * ( U - U[N] ) / ( U[0] - U[N] )                                     # Enforce mean value.
  
        # Update values.
        err = max(abs(U-Ups))                                                      # Error measured by the L_inf norm.
        Ups = U                                                                    # New Upsilon.
        iter = iter+1
  
    toc = time.time() - tic

    # Post processing.
    IH_hat = -1j*1/np.tanh(sig*d*vk);  IH_hat[0] = 0                               # Inverse Hilbert transform.
    Ys  = Ups - Ups.mean()
    Ys_hat  = np.fft.fft(Ys)
    CYs = np.fft.ifft(C_hat*Ys_hat).real
    Xs  = np.fft.ifft(IH_hat*Ys_hat).real
    mys = -np.dot(Ys,CYs)/N/2
    Zs  = Xs + 1j*Ys
    dZs = np.fft.ifft(1j*vk*np.fft.fft(Zs))
    zs  = va + 1j*mys + Zs
    dzs = 1 + dZs
    B   = g*Bg
    ce  = sum( (1+CYs)/abs(dzs)**2 )/2/N
    ce  = np.sqrt(B/ce)
    cs  = sig*ce
    ws  = -ce/dzs
    a   = max(zs.imag)
    b   = -min(zs.imag)

    xs = np.concatenate([ zs[N:].real-2*np.pi/k, zs[:N].real ])
    ys = np.concatenate([ zs[N:].imag, zs[:N].imag ])

    if d==np.inf:
        Bce2d = 0
        IC = 1/abs(vk); IC[0] = 0
    else:
        Bce2d = (B-ce**2)*d
        IC = np.tanh(vk*sig*d)/vk; IC[0] = sig*d                                   # Inverse C-operator.

    ydx  = dzs.real*zs.imag
    intI = -ce*mys                                                                 # Impulse.
    intV = (ydx*zs.imag).mean()*g/2                                                # Potential energy.
    intK = intI*ce/2                                                               # Kinetic energy.
    intSxx = 2*ce*intI - 2*intV + Bce2d                                            # Radiation stress.
    intS = intSxx - intV + g*d**2/2                                                # Momentum flux.
    intF = Bce2d*ce/2 + (B+ce**2)*intI/2 + (intK-2*intV)*ce                        # Energy flux.
    cg   = intF/(intK+intV)                                                        # Group velocity.
    K1   = (np.dot(zs.imag,(zs.imag)/2)-Bg)/2/N
    ICydx  = np.fft.ifft(IC*np.fft.fft(ydx)).real 
    errfun = max(abs( zs.imag - Bg + np.sqrt(Bg**2+2*K1-2*ICydx)))                 # Residual.

    # Output data.

    PP = [ d, k, H, ce, cs, B, a, b, intI, intV, intK, intSxx, intS, intF, cg]

    # Display results.
    plt.subplot(211)
    plt.plot(xs,ys,label='Surface Elevation')
    plt.plot(xs,0*ys,'k--',label='Still Water Level')
    if d==np.inf:
        plt.xlim([-np.pi, np.pi])
        plt.ylim(np.array([-b, a])*1.0333)
        plt.xlabel('$k\ x$')
        plt.ylabel('$k\ \eta$')
    else:
        plt.xlim([xs[0], xs[-1]])
        plt.ylim(np.array([-b, a])*1.0333)
        plt.xlabel('$x / d$')
        plt.ylabel('$\eta / d$')
 
    plt.title('Free Surface')
    plt.legend(loc="upper right")

    plt.subplot(212)
    plt.semilogy(np.arange(2*N),abs(np.fft.fft(zs.imag)))
    plt.xlim([0, N-1])
    plt.ylim([1e-17, max(abs(np.fft.fft(zs.imag)))*1.0333])
    plt.xlabel('Fourier modes')
    plt.ylabel(r'$\log_{10} | \mathcal{F}\{\tilde{y}\} |$')
    plt.title('Spectrum')
    plt.show()

    # Print physical parameters.
    print('NUMERICAL PARAMETERS')
    print('Number of positive Fourier modes:     N = %9i' % N)
    print('Tolerance:                          tol = %15.14e'% tol)
    print('Number of iterations:              iter = %9i'% iter)
    print('Residual:                           res = %9i'% errfun)
    print('Iterations time (s)                time = %15.14f'% toc)
    print()
    print('PHYSICAL PARAMETERS')
    print('Mean depth:                           d = %15.14e'% d)
    print('Acceleration due to gravity:          g = %15.14e'% g)
    wl =  2*np.pi/k
    print('Wavelength:                      2*pi/k = %15.14e'% wl)
    print()
    print('WAVE CHARACTERISTICS')
    print('Wave height:                          H = %15.14f'% H)
    print('Crest height (amplitude):             a = %15.14f'% a)
    print('Trough height:                        b = %15.14f'% b)
    print('Stokes first phase celerity:        c_e = %15.14f'% ce)
    print('Stokes second phase celerity:       c_s = %15.14f'% cs)
    sqglam =  np.sqrt(g*lam)
    print('Linear phase celerity:              c_0 = %15.14f'% sqglam)
    print('Bernoulli constant:                   B = %15.14f'% B)
    print()
    print('INTERGRAL QUANTITIES (in the frame of reference with zero circulation)')
    print('Impulse:                              I = %15.14f'% intI)
    print('Potential energy:                     V = %15.14f'% intV)
    print('Kinetic energy:                       K = %15.14f'% intK)
    print('Radiation stress:                   Sxx = %15.14f'% intSxx)
    if d!=np.inf:
        print('Momentum flux:                        S = %15.14f'% intS)
    print('Energy flux:                          F = %15.14f'% intF)
    print('Group celerity:                     c_g = %15.14f'% cg)
    return [zs,ws,PP]

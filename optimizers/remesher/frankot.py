# frankot.py

import numpy as np
import numpy.fft as fft
import math

# Perform Frankot-Chellapa intergration.
# Input: numpy 2D matrices representing x and y partial differentials
# Output: outputs closest integrable heighfield.
def project_surface(p, q):
    sz = p.shape;
    N = sz[0];
    M = sz[1];
    # Frequency indices.
    halfM = (M-1)/2;
    halfN = (N-1)/2;
    (u, v) = np.meshgrid(np.linspace(-math.ceil(halfN),math.floor(halfN),sz[0]), np.linspace(-math.ceil(halfM),math.floor(halfM), sz[1]));
    u = fft.ifftshift(u);
    v = fft.ifftshift(v);

    # Compute the Fourier transform of 'p' and 'q'.
    Fp = fft.fft2(p);
    Fq = fft.fft2(q);
    #print(Fp.shape);
    #print(p.shape);
    #print(u.shape);
    
    denominator = ((u/N)**2 + (v/M)**2);
    #print denominator[:10,:10];
    # Compute the Fourier transform of 'Z'.
    Fz = -1j/(2*np.pi) * np.divide(np.multiply(u,Fp/N) + np.multiply(v,Fq/M), denominator);
    #print Fz[:10, :10];
    #print Fp[:10, :10];
    #print u[:10, :10];

    # Set DC component.
    Fz[1] = 0;

    Z = np.real(fft.ifft2(Fz));

    return Z;

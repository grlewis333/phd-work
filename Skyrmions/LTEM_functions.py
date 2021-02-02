import numpy as np                              # For maths
from scipy import constants                     # For scientific constants
from skimage.restoration import unwrap_phase    # unwrap phase

def B_phase_calc(MXr,MYr,MZr,ds = 0.1,kx = 0.1,ky = 0.1,kv = 300,Cs = 8000,Msat = 480767.832897):

    #ds # Defocus step in mm
    #kx # Tikhonov filter radius in x in pixels
    #ky # Tikhonov filter radius in y in pixels
    #kv # Acceleratig voltage of electrons in kV
    #Cs # Spherical aberration coefficient in mm
    with HiddenPrints():
        x_res, y_res, z_res, x_size, y_size, z_size, x_begin, y_begin, z_begin, x_end, y_end, z_end = unify_input(X,Y,Z,MXr,MYr,MZr)
    
    
    MXpad = np.pad(MXr,[(100,100),(100,100),(100,100)], mode='constant', constant_values=0)
    MYpad = np.pad(MYr,[(100,100),(100,100),(100,100)], mode='constant', constant_values=0)
    
    ave_m_x = np.mean(MXpad,axis = 2)* Msat
    ave_m_y = np.mean(MYpad,axis = 2)* Msat
    

    
    sx = 1/(x_size*x_res) # sampling in reciprocal space 
    sy = 1/(y_size*y_res) # identical in both directions
    const = 1j/(2*constants.codata.value('mag. flux quantum')/((constants.nano)**2))
    λ = λ_func(kv) # Wavelength of electrons in nm

    ft_mx = np.fft.fft2(ave_m_x*z_size)# , axes=(-2, -1))
    ft_my = np.fft.fft2(ave_m_y*z_size)#, axes=(-2, -1))

    FreqCompRows = np.fft.fftfreq(ft_mx.shape[0],d=x_res)
    FreqCompCols = np.fft.fftfreq(ft_mx.shape[1],d=y_res)
    Xft, Yft = np.meshgrid(FreqCompCols,FreqCompRows, indexing='ij') # Create a grid of coordinates

    nume =  ((Xft**2)+(Yft**2))
    dnom =  ((Xft**2)+(Yft**2)+(sx**2)*(kx**2)+(sy**2)*(ky**2))**2
    cross = -ft_my*Xft+ft_mx*Yft
    B0 = 4*np.pi*1e-7 # * size_n
    ft_phase = np.array(const*cross*nume/dnom) * B0
    phase_B = np.fft.ifft2(ft_phase).real
    
    return phase_B[100:-100,100:-100]

def λ_func(V):
    V *= constants.kilo
    λ = constants.h/(constants.nano*np.sqrt(2*V*constants.m_e*constants.e))
    λ *= 1/(np.sqrt(1+(constants.e*V)/(2*constants.m_e*constants.c**2)))
    return λ

def calculate_B(phase_B):
    d_phase = np.gradient(phase_B)
    b_const = (constants.codata.value('mag. flux quantum')/(constants.nano**2))/(np.pi*z_size)
    b_field_x = -b_const*d_phase[0]/y_res
    b_field_y = b_const*d_phase[1]/x_res

    mag_B = np.hypot(b_field_x,b_field_y)
    
    return mag_B,b_field_x,b_field_y
import numpy as np                              # For maths
from scipy import constants                     # For scientific constants
from skimage.restoration import unwrap_phase    # unwrap phase
from matplotlib import pyplot as plt
from scipy import ndimage                       # For image rotations
from mpl_toolkits.axes_grid1 import make_axes_locatable

def B_phase_calc(MXr,MYr,MZr,ds = 0.1,kx = 0.1,ky = 0.1,kv = 300,Cs = 8000,Msat = 1,x_res =1,y_res=1,z_res=1):

    #ds # Defocus step in mm
    #kx # Tikhonov filter radius in x in pixels
    #ky # Tikhonov filter radius in y in pixels
    #kv # Acceleratig voltage of electrons in kV
    #Cs # Spherical aberration coefficient in mm
#     with HiddenPrints():
#         x_res, y_res, z_res, x_size, y_size, z_size, x_begin, y_begin, z_begin, x_end, y_end, z_end = unify_input(X,Y,Z,MXr,MYr,MZr)
    x_size,y_size,z_size = np.shape(MXr)
    
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

def calculate_B(phase_B,z_size=1,y_res=1,x_res=1):
    d_phase = np.gradient(phase_B)
    b_const = (constants.codata.value('mag. flux quantum')/(constants.nano**2))/(np.pi*z_size)
    b_field_x = -b_const*d_phase[0]/y_res
    b_field_y = b_const*d_phase[1]/x_res

    mag_B = np.hypot(b_field_x,b_field_y)
    
    return mag_B,b_field_x,b_field_y

def plot_2d(X,Y,Z,MX,MY,MZ,s=5,size=0.1, width = 0.005, title=''):
    """
    Plot magnetisation data in 2D
    
    Input:
    x,y = 'Projected' 2D coordinates (nxn)
    u,v = 'Projected' 2D magnetisation (nxn)
    s = Quiver plot skip density
    size = Arrow length scaling
    width = Arrow thickness 
    
    Output:
    2D Plot of magnetisation:
    - Arrows show direction of M
    - Background color shows magnitude of M
    """
    # Project along z by averaging
    x_proj = np.mean(X,axis=2)
    y_proj = np.mean(Y,axis=2)
    z_proj = np.mean(Z,axis=2)
    u_proj = np.mean(MX,axis=2)
    v_proj = np.mean(MY,axis=2)
    w_proj = np.mean(MZ,axis=2)
    
    # Create figure
    fig = plt.figure(figsize=(6, 8))
    grid = plt.GridSpec(4, 3)
    ax1 = fig.add_subplot(grid[3, 0])
    ax2 = fig.add_subplot(grid[3, 1])
    ax3 = fig.add_subplot(grid[3, 2])
    ax4 = fig.add_subplot(grid[:3, :])
    
    # plot Mx
    pos =  ax1.imshow(np.flipud(u_proj.T), vmin=-1,vmax=1,cmap='RdBu'); 
    ax1.set_title('Mx',fontsize=13); ax1.set_xlabel('x',fontsize=14);ax1.set_ylabel('y',fontsize=14);

    # Plot My
    pos = ax2.imshow(np.flipud(v_proj.T), vmin=-1,vmax=1,cmap='RdBu');
    ax2.set_title('My',fontsize=13); ax2.set_xlabel('x',fontsize=14);ax2.set_ylabel('y',fontsize=14); 

    #Plot Mz
    pos = ax3.imshow(np.flipud(w_proj.T), vmin=-1,vmax=1,cmap='RdBu',extent=[x_begin, x_end, y_begin, y_end]);
    ax3.set_title('Mz',fontsize=13); ax3.set_xlabel('x',fontsize=14);ax3.set_ylabel('y',fontsize=14); 
    fig.colorbar(pos,ax=ax3,fraction=0.046, pad=0.04)

    # Main arrow plot
    ax4.quiver(x_proj[::s,::s],y_proj[::s,::s],u_proj[::s,::s],v_proj[::s,::s],scale=1/size,pivot='mid',width=width)
    
    # Calculate vector magnitude
    magnitude = (u_proj**2 + v_proj**2)**0.5
    
    # Plot vector magnitude
    im1 = ax4.imshow(np.flipud(magnitude.T),vmin=0,vmax=1,cmap='Blues',
                     extent=(np.min(x_proj),np.max(x_proj),np.min(y_proj),np.max(y_proj)))
    
    # Add colorbar and labels
    clb = fig.colorbar(im1,ax=ax4,fraction=0.046, pad=0.04)
    ax4.set_xlabel('x / nm',fontsize=14)
    ax4.set_ylabel('y / nm',fontsize=14)
    ax4.set_title(title, fontsize= 16)
    for ax in [ax1,ax2,ax3]:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    
    plt.tight_layout()
    
def rotate_bulk(U,V,W,ax,ay,az):
    """ 
    Rotate magnetisation locations from rotation angles ax,ay,az 
    about the x,y,z axes (given in degrees) 
    
    NOTE: This implementation of scipy rotations is EXTRINSIC
    Therefore, to make it compatible with our intrinsic vector
    rotation, we swap the order of rotations (i.e. x then y then z)
    """
    # Due to indexing, ay needs reversing for desired behaviour
    ay = -ay
    
    U = ndimage.rotate(U,ax,reshape=False,axes=(1,2),order=1)
    U = ndimage.rotate(U,ay,reshape=False,axes=(2,0),order=1)
    U = ndimage.rotate(U,az,reshape=False,axes=(0,1),order=1)
    
    V = ndimage.rotate(V,ax,reshape=False,axes=(1,2),order=1)
    V = ndimage.rotate(V,ay,reshape=False,axes=(2,0),order=1)
    V = ndimage.rotate(V,az,reshape=False,axes=(0,1),order=1)
    
    W = ndimage.rotate(W,ax,reshape=False,axes=(1,2),order=1)
    W = ndimage.rotate(W,ay,reshape=False,axes=(2,0),order=1)
    W = ndimage.rotate(W,az,reshape=False,axes=(0,1),order=1)

    return U,V,W

def grid_to_coor(U,V,W):
    """ Convert gridded 3D data (3,n,n,n) into coordinates (n^3, 3) """
    coor_flat = []
    nx,ny,nz = np.shape(U)
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x = U[ix,iy,iz]
                y = V[ix,iy,iz]
                z = W[ix,iy,iz]
                coor_flat.append([x,y,z])
                
    return coor_flat

def coor_to_grid(coor_flat,shape):
    """ Convert coordinates (n^3, 3) into gridded 3D data (3,n,n,n) """
    n = int(np.round(np.shape(coor_flat)[0]**(1/3)))
    nx,ny,nz = shape
    x = np.take(coor_flat,0,axis=1)
    y = np.take(coor_flat,1,axis=1)
    z = np.take(coor_flat,2,axis=1)
    U = x.reshape((nx,ny,nz))
    V = y.reshape((nx,ny,nz))
    W = z.reshape((nx,ny,nz))

    return U, V, W

def rotation_matrix(ax,ay,az):
    """ 
    Generate 3D rotation matrix from rotation angles ax,ay,az 
    about the x,y,z axes (given in degrees) 
    (Uses convention of rotating about z, then y, then x)
    """

    ax = ax * np.pi/180
    Cx = np.cos(ax)
    Sx = np.sin(ax)
    mrotx = np.array([[1,0,0],[0,Cx,-Sx],[0,Sx,Cx]])
    
    ay = ay * np.pi/180
    Cy = np.cos(ay)
    Sy = np.sin(ay)
    mroty = np.array([[Cy,0,Sy],[0,1,0],[-Sy,0,Cy]])
    
    az = az * np.pi/180
    Cz = np.cos(az)
    Sz = np.sin(az)
    mrotz = np.array([[Cz,-Sz,0],[Sz,Cz,0],[0,0,1]])
    
    mrot = mrotz.dot(mroty).dot(mrotx)
    
    return mrot

def rotate_vector(coor_flat,ax,ay,az):
    """ Rotates vectors by specified angles ax,ay,az 
    about the x,y,z axes (given in degrees) """
    
    # Get rotation matrix
    mrot = rotation_matrix(ax,ay,az)    

    coor_flat_r = np.zeros_like(coor_flat)
    
    # Apply rotation matrix to each M vector
    for i,M in enumerate(coor_flat):
        coor_flat_r[i] = mrot.dot(M)
    
    return coor_flat_r

def rotate_magnetisation(U,V,W,ax=0,ay=0,az=0):
    """ 
    Takes 3D gridded magnetisation values as input
    and returns them after an intrinsic rotation ax,ay,az 
    about the x,y,z axes (given in degrees) 
    (Uses convention of rotating about z, then y, then x)
    """
    # Rotate the gridded locations of M values
    Ub, Vb, Wb = rotate_bulk(U,V,W,ax,ay,az)
    
    # Convert gridded values to vectors
    coor_flat = grid_to_coor(Ub,Vb,Wb)
    
    # Rotate vectors
    coor_flat_r = rotate_vector(coor_flat,ax,ay,az)
    
    # Convert vectors back to gridded values
    shape = np.shape(U)
    Ur,Vr,Wr = coor_to_grid(coor_flat_r,shape)
    
    # Set small values to 0
    # (In theory the magnitude of M in each cell should be Ms,
    #  so we can set magnitude lower than this to zero -
    #  typically python rounding errors lead to very small values,
    #  which it is useful to exclude here)
#     mag_max = (np.max(U)**2+np.max(V)**2+np.max(W)**2)**0.5
#     mag = (Ur**2+Vr**2+Wr**2)**.5
#     for M in [Ur,Vr,Wr]:
#         M[abs(M)<1e-5*mag_max] = 0
#         M[mag<.6*mag_max] = 0
    
    return Ur,Vr,Wr

def plot_phase(Mx,My,Mz,ax=None,vmax=0.01429,vmin=0):
    if ax == None:
        plt.figure(figsize=(9,9))
        ax = plt.gca()
        
    # Get phase image
    phase_B = B_phase_calc(Mx,My,Mz,x_res=3,y_res=3,z_res=2)

    # Unwrap phase image
    pu = unwrap_phase(phase_B)

    # plot
    a = pu[:,::-1].T
    im = ax.imshow(a/np.pi+abs(np.min(a/np.pi)),cmap='binary_r',vmax=0.007,vmin=vmin)
    ax.axis('off')
    ax.set_title('Phase image',fontsize=25)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im,cax=cax)
    
    return pu
    
def omf_to_mag(data):
    """ Extract magnetization in grid array from ubermag 'system' object """
    #ms = system.m.array
    ms = data
    shape = np.shape(ms)

    xs,ys,zs,mx,my,mz = [],[],[],[],[],[]
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                xs.append(i)
                ys.append(j)
                zs.append(k)
                mx.append(ms[i][j][k][0])
                my.append(ms[i][j][k][1])
                mz.append(ms[i][j][k][2])
    Mx,My,Mz = np.reshape(mx,(shape[0],shape[1],shape[2])),\
                            np.reshape(my,(shape[0],shape[1],shape[2])), \
                            np.reshape(mz,(shape[0],shape[1],shape[2]))
    return Mx,My,Mz

def plot_mag(Mx,My,Mz,direction = 'z',s=1,ax=None,title=None):
    if ax == None:
        plt.figure(figsize=(9,9))
        ax = plt.gca()
    
    if direction == 'z':
        M_proj = np.mean(Mz,axis=2)[:,::-1].T
        a = np.mean(Mx,axis=2)[:,::-1].T
        b = np.mean(My,axis=2)[:,::-1].T
        
    if direction == 'y':
        M_proj = np.mean(My,axis=1)[:,::-1].T
        a = np.mean(Mx,axis=1)[:,::-1].T
        b = np.mean(Mz,axis=1)[:,::-1].T
        
    ax.quiver(a[::s,::s],b[::s,::s],pivot='middle',width=0.006,scale=8e6,edgecolors='w',linewidth=.5)
    mmax = np.max((Mx**2+My**2+Mz**2)**0.5)
    im = ax.imshow(M_proj,cmap='seismic',extent=(0-.5,np.shape(a[::s,::s])[1]-.5,np.shape(a[::s,::s])[0]-.5,0-.5),vmin=-mmax,vmax=mmax)
    ax.axis('off')
    if title == None:
        title = 'Projected Magnetisation'
    ax.set_title(title,fontsize=25)
    
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(im,cax=cax)
    cbar.ax.set_ylabel('$M_{\parallel}$',rotation=0,fontsize=20)
    
def cart2pol(x, y):
    """ Convert cartesian to polar coordinates
    rho = magnitude, phi = angle """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def plot_rafal(pu,mag_B,b_field_x,b_field_y,ax1=None):
    s = 5
    size_arrow = 0.25
    angle = 0
    
    import matplotlib.colors
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as patches

    cvals  = [-2., 2]
    colors = ["black","black"]
    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))
    # Set alpha
    my_cmap[int(0.15*cmap.N):int(0.95*cmap.N),-1] = np.linspace(1, 0, int(0.8*cmap.N+1))
    my_cmap[int(0.95*cmap.N):,-1] = np.ones_like(my_cmap[int(0.95*cmap.N):,-1])*0
    # Create new colormap
    my_cmap = ListedColormap(my_cmap)
    
    if ax1 == None:
        fig, ax1 = plt.subplots(ncols=1, figsize=(8, 8))
    
    a = b_field_x
    x_begin,x_end,y_begin,y_end=(0-.5,np.shape(a[::s,::s])[1]-.5,np.shape(a[::s,::s])[0]-.5,0-.5)
    
    # plot B field direction as a colour
    ax1.imshow(np.arctan2(b_field_y,b_field_x).T,origin='lower', \
               extent=(0-.5,np.shape(a[::s,::s])[1]-.5,np.shape(a[::s,::s])[0]-.5,0-.5), cmap='hsv')

    # Plot magnitude of B as in black/transparent scale
    ax1.imshow(mag_B.T,origin='lower', extent=(0-.5,np.shape(a[::s,::s])[1]-.5,np.shape(a[::s,::s])[0]-.5,0-.5),\
            interpolation='spline16', cmap=my_cmap)
    #cs = ax1.contour(pu.T,origin='lower',levels=np.pi, extent=(0-.5,np.shape(a[::s,::s])[1]-.5,np.shape(a[::s,::s])[0]-.5,0-.5),interpolation='spline16', alpha = 1,colors='k',ls=10,antialiased=True)
    #plt.contour(pu, cmap='gray', linewidth=.1)


    #c.linewidth=10
    pa=1000
    cos_phase = np.cos(pu*pa)
    # Plot cosine of phase as black/transparent
    ax1.imshow(cos_phase.T,origin='lower', extent=(0-.5,np.shape(a[::s,::s])[1]-.5,np.shape(a[::s,::s])[0]-.5,0-.5),\
               interpolation='spline16', cmap=my_cmap)

    #ax1.set_title(r'$\bf{B}$$_\perp$', fontsize=25)
    ax1.set_xlabel('x', fontsize = 16)
    ax1.set_ylabel('y', fontsize = 16)



    #### color wheel
    




    # Create coordinate space
    pos = (0.8,-0.8) # x,y position fractionally from -1 to 1
    wheel_rad = 0.3 # as a fraction of the image width 0 to 1

    x = np.linspace(-1,1,200)
    y = x
    X,Y = np.meshgrid(x,y)



    # Map theta values onto coordinate space 
    thetas = np.ones_like(X)*0
    for ix, xx in enumerate(x):
        for iy, yy in enumerate(y):
            thetas[ix,iy] = cart2pol(xx+pos[1],yy-pos[0])[1]

    # Plot hsv colormap of angles
    im1 = ax1.imshow(thetas,cmap='hsv_r', extent=(0-.5,np.shape(a[::s,::s])[1]-.5,np.shape(a[::s,::s])[0]-.5,0-.5))

    # Map circle radii onto xy coordinate space
    circ = np.ones_like(X)*0
    for ix, xx in enumerate(x):
        for iy, yy in enumerate(y):
            if (xx+pos[1])**2 + (yy-pos[0])**2 < wheel_rad**2:
                circ[ix,iy] = cart2pol(xx+pos[1],yy-pos[0])[0]

    # Plot circle
    im2 = ax1.imshow(circ, cmap=my_cmap, extent=(0-.5,np.shape(a[::s,::s])[1]-.5,np.shape(a[::s,::s])[0]-.5,0-.5))

    # Clip to make it circular
    x_frac = (pos[0] + 1) / 2
    y_frac = (pos[1] + 1) / 2
    centre_x = x_begin + x_frac*(x_end-x_begin)
    centre_y = y_begin + y_frac*(y_end-y_begin)
    patch = patches.Circle((centre_x, centre_y), transform=ax1.transData, radius=.8)#wheel_rad*(x_end-x_begin)/2-1)
    im2.set_clip_path(patch)
    im1.set_clip_path(patch)

    ax1.set_title(r'Colour: $\tan^{-1}(B_\perp^y/ B_\perp^x)$,  Intensity: $\cos(1000 \phi_B)$',fontsize=18)

    ax1.axis('off')
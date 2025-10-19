import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image, display

# Output directory
OUTDIR = '/mnt/data/phonon_transport_outputs'
os.makedirs(OUTDIR, exist_ok=True)

# --- Functions ---

def generate_lattice(n_sites, box_size=1.0, seed=None):
    rng = np.random.default_rng(seed)
    positions = rng.random((n_sites, 2)) * box_size
    return positions


def compute_phonon_coupling(positions, cutoff=0.1, g0=1.0, decay_len=0.02):
    n = len(positions)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, pos=positions[i])
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(positions[i]-positions[j])
            if dist <= cutoff:
                g = g0 * np.exp(-dist/decay_len)
                G.add_edge(i,j, conductance=g)
    return G


def map_local_conductance(G, positions, box_size=1.0, grid_res=64):
    grid = np.zeros((grid_res, grid_res))
    xedges = np.linspace(0, box_size, grid_res+1)
    yedges = np.linspace(0, box_size, grid_res+1)
    for u,v,d in G.edges(data=True):
        mid = 0.5*(G.nodes[u]['pos']+G.nodes[v]['pos'])
        ix = np.searchsorted(xedges, mid[0])-1
        iy = np.searchsorted(yedges, mid[1])-1
        ix = np.clip(ix,0,grid_res-1)
        iy = np.clip(iy,0,grid_res-1)
        grid[iy, ix] += d['conductance']
    return grid


def fourier_lowpass(grid, lowpass_frac=0.08):
    f = np.fft.fftshift(np.fft.fft2(grid))
    ny,nx = grid.shape
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    KX,KY = np.meshgrid(kx,ky)
    K = np.sqrt(KX**2 + KY**2)
    Kmax = 0.5
    mask = (K <= lowpass_frac*Kmax).astype(float)
    f_filtered = f*mask
    return np.real(np.fft.ifft2(np.fft.ifftshift(f_filtered)))


def animate_lattice(positions, box_size=1.0, thermal_scale=0.005, frames=100, outpath=None):
    fig, ax = plt.subplots(figsize=(5,5))
    sc = ax.scatter(positions[:,0], positions[:,1], s=8, c='royalblue')
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_title('Thermal lattice motion')

    def update(frame):
        steps = np.random.normal(0, thermal_scale, size=positions.shape)
        positions[:] = (positions + steps) % box_size
        sc.set_offsets(positions)
        return sc,

    anim = FuncAnimation(fig, update, frames=frames, interval=80, blit=True)
    if outpath:
        anim.save(outpath, writer=PillowWriter(fps=15))
    plt.close(fig)
    return outpath


# --- Example Run ---
n_sites = 400
box_size = 1.0
positions = generate_lattice(n_sites, box_size, seed=42)
G = compute_phonon_coupling(positions, cutoff=0.12, g0=1.0, decay_len=0.02)

# Map conductance and apply Fourier filtering
grid = map_local_conductance(G, positions, box_size, grid_res=128)
grid_filt = fourier_lowpass(grid, lowpass_frac=0.06)

# Save images
plt.figure(figsize=(5,4))
plt.imshow(grid, origin='lower')
plt.title('Local Phonon Conductance Map')
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR,'conductance_map.png'), dpi=200)
plt.close()

plt.figure(figsize=(5,4))
plt.imshow(grid_filt, origin='lower')
plt.title('Low-pass Filtered Conductance')
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR,'conductance_lowpass.png'), dpi=200)
plt.close()

# Animate lattice thermal motion
gif_path = os.path.join(OUTDIR,'lattice_thermal_motion.gif')
gif_path = animate_lattice(positions, box_size, thermal_scale=0.007, frames=120, outpath=gif_path)

print('Simulation complete. Outputs saved to', OUTDIR)
display(Image(filename=gif_path))
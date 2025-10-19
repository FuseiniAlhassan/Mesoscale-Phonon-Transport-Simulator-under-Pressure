# Mesoscale Phonon Transport Simulator
# Author: Shanti Deemyad (Conceptual Goal)
#
# This script simulates the behavior of a 2D crystal lattice under tunable
# pressure and temperature. It models lattice sites as nodes in a network
# with spring-like edges. The simulation visualizes how pressure distorts
# the lattice and animates the resulting changes in phonon (vibrational) modes.

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# --- Configuration Parameters ---
# Lattice Settings
GRID_WIDTH = 20
GRID_HEIGHT = 15
LATTICE_SPACING = 1.0  # Initial equilibrium distance between sites

# Physics Simulation Settings
TIMESTEPS = 300       # Total steps for the simulation
DT = 0.1             # Time step duration for integration
STIFFNESS = 1.0        # Spring constant for lattice bonds
MASS = 1.0           # Mass of each lattice site
TEMPERATURE = 0.05   # Controls the magnitude of random thermal kicks
DAMPING = 0.02         # Damps oscillations to help reach equilibrium

# Pressure Settings
INITIAL_PRESSURE_FACTOR = 1.0  # Starts with no pressure (1.0 = normal spacing)
FINAL_PRESSURE_FACTOR = 0.85 # Compresses lattice to 85% of original spacing
PRESSURE_RAMP_STEPS = 150    # How many steps to reach final pressure

# --- 1. Lattice Initialization ---

def create_lattice(width, height, spacing):
    """
    Creates a 2D grid graph representing the crystal lattice.
    Each edge stores a 'rest_length' attribute for pressure simulation.
    """
    G = nx.grid_2d_graph(width, height)
    # Initialize node positions and physics properties
    for node in G.nodes():
        G.nodes[node]['pos'] = np.array(node, dtype=float) * spacing
        G.nodes[node]['vel'] = np.zeros(2, dtype=float)
        G.nodes[node]['force'] = np.zeros(2, dtype=float)
    
    # Initialize edge properties (rest length for spring forces)
    for edge in G.edges():
        G.edges[edge]['rest_length'] = spacing
        
    print(f"Created a {width}x{height} lattice with {G.number_of_nodes()} sites.")
    return G

# --- 2. Physics Simulation Core ---

def apply_forces(G):
    """
    Calculates the spring forces on each node based on bond tension/compression.
    F = -k * (|L| - L_rest) * (L / |L|)
    """
    for node in G.nodes():
        G.nodes[node]['force'][:] = 0.0 # Reset forces each step
        
    for u, v in G.edges():
        pos_u = G.nodes[u]['pos']
        pos_v = G.nodes[v]['pos']
        
        vec = pos_v - pos_u
        dist = np.linalg.norm(vec)
        if dist == 0: continue
            
        rest_length = G.edges[u, v]['rest_length']
        displacement = dist - rest_length
        
        direction = vec / dist
        force_magnitude = STIFFNESS * displacement
        
        force_vec = force_magnitude * direction
        
        # Apply forces to both nodes (Newton's 3rd law)
        G.nodes[u]['force'] += force_vec
        G.nodes[v]['force'] -= force_vec

def update_positions(G):
    """
    Updates node positions and velocities using Verlet integration.
    Includes thermal kicks and damping.
    """
    for node in G.nodes():
        # Apply thermal perturbation (stochastic kick)
        thermal_force = np.random.normal(0, 1, 2) * np.sqrt(2 * DAMPING * TEMPERATURE / DT)
        
        # Update velocity
        accel = (G.nodes[node]['force'] + thermal_force) / MASS
        G.nodes[node]['vel'] += accel * DT
        G.nodes[node]['vel'] *= (1 - DAMPING) # Apply damping
        
        # Update position
        G.nodes[node]['pos'] += G.nodes[node]['vel'] * DT

def update_pressure(G, current_step):
    """Gradually ramp up the pressure by reducing the rest length of bonds."""
    if current_step < PRESSURE_RAMP_STEPS:
        # Linearly interpolate the pressure factor from initial to final
        factor = np.interp(current_step, [0, PRESSURE_RAMP_STEPS], [INITIAL_PRESSURE_FACTOR, FINAL_PRESSURE_FACTOR])
        new_rest_length = LATTICE_SPACING * factor
        for edge in G.edges():
            G.edges[edge]['rest_length'] = new_rest_length

# --- 3. Analysis and Visualization ---

def compute_local_conductance_metric(G):
    """
    Computes a simplified metric for local phonon conductance.
    This proxy is based on the normalized correlation of velocities
    between neighboring nodes. High correlation suggests efficient energy transfer.
    """
    conductance = {}
    velocities = np.array([G.nodes[n]['vel'] for n in G.nodes()])
    
    for u, v in G.edges():
        node_u_idx = list(G.nodes()).index(u)
        node_v_idx = list(G.nodes()).index(v)
        
        vel_u = velocities[node_u_idx]
        vel_v = velocities[node_v_idx]
        
        # Dot product of velocities captures their alignment
        corr = np.dot(vel_u, vel_v)
        
        # Normalize by magnitudes to get a cosine similarity like metric
        norm_u = np.linalg.norm(vel_u)
        norm_v = np.linalg.norm(vel_v)
        
        if norm_u * norm_v > 0:
            conductance[(u, v)] = corr / (norm_u * norm_v)
        else:
            conductance[(u, v)] = 0
            
    return conductance

def visualize_final_state(G):
    """
    Plots the final state of the lattice, coloring nodes by displacement
    and edges by the conductance metric.
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    
    initial_pos = {n: np.array(n) * LATTICE_SPACING for n in G.nodes()}
    final_pos = {n: G.nodes[n]['pos'] for n in G.nodes()}
    
    # Node color based on magnitude of displacement from original position
    displacements = [np.linalg.norm(final_pos[n] - initial_pos[n]) for n in G.nodes()]
    
    # Edge color based on conductance metric
    conductance = compute_local_conductance_metric(G)
    edge_colors = [conductance.get(e, 0) for e in G.edges()]
    
    nodes = nx.draw_networkx_nodes(G, pos=final_pos, node_color=displacements, cmap=plt.cm.viridis, node_size=100, ax=ax)
    edges = nx.draw_networkx_edges(G, pos=final_pos, edge_color=edge_colors, edge_cmap=plt.cm.inferno, width=1.5, ax=ax)
    
    # Add colorbars
    cbar_nodes = fig.colorbar(nodes, ax=ax, orientation='vertical')
    cbar_nodes.set_label('Total Displacement', fontsize=12)
    cbar_edges = fig.colorbar(edges, ax=ax, orientation='vertical', pad=0.05)
    cbar_edges.set_label('Phonon Conductance Proxy (Velocity Correlation)', fontsize=12)
    
    ax.set_title("Final Lattice State Under Pressure", fontsize=16)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig("final_lattice_state.png")
    print("\nSaved final lattice state visualization to 'final_lattice_state.png'")
    plt.close()

def animate_simulation(history):
    """
    Creates an animation of the lattice deforming and vibrating under pressure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def update(frame):
        ax.clear()
        G = history[frame]
        pos = {n: G.nodes[n]['pos'] for n in G.nodes()}
        displacements = [np.linalg.norm(pos[n] - (np.array(n)*LATTICE_SPACING)) for n in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_size=50, node_color=displacements, cmap=plt.cm.plasma, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7, edge_color='gray', ax=ax)

        ax.set_title(f"Lattice Dynamics - Step {frame}/{TIMESTEPS}", fontsize=14)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    print("Generating animation (this may take a while)...")
    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=30)
    ani.save('lattice_pressure_animation.gif', writer='pillow')
    print("Saved animation to 'lattice_pressure_animation.gif'")
    plt.close()


# --- Main Execution Block ---

if __name__ == "__main__":
    # 1. Initialize the lattice
    lattice = create_lattice(GRID_WIDTH, GRID_HEIGHT, LATTICE_SPACING)
    
    # 2. Run the simulation
    simulation_history = []
    print("Running simulation...")
    for t in tqdm(range(TIMESTEPS)):
        update_pressure(lattice, t)
        apply_forces(lattice)
        update_positions(lattice)
        
        # Store a copy of the graph state for animation
        if t % 2 == 0: # Store every other frame to speed up animation generation
            simulation_history.append(lattice.copy())
            
    print("Simulation complete.")
    
    # 3. Visualize the final results
    visualize_final_state(lattice)
    
    # 4. Create and save the animation
    animate_simulation(simulation_history)
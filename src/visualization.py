import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot2d(vertices, tx=None, rx=None, name="2d_scene"):
    """
    Plot a 2D scene with obstacles defined by vertices, optional transmitter (tx) and receiver (rx).
    """
    plt.figure(figsize=(6,6))
    for idx, obj in enumerate(vertices):
        x, y = zip(*obj)
        if len(obj) > 2:
            x += (x[0],)
            y += (y[0],)
        label = 'Walls' if idx == 0 else None
        plt.plot(x, y, 'b-', linewidth=2, label=label)

    if tx is not None:
        tx = jnp.atleast_2d(tx)
        plt.plot(tx[:, 0], tx[:, 1], 'rx', markersize=7)
        plt.plot(tx[0, 0], tx[0, 1], 'rx', label='tx')
    if rx is not None:
        rx = jnp.atleast_2d(rx)
        plt.plot(rx[:, 0], rx[:, 1], marker='o', color='deeppink', markersize=4, linestyle='None')
        plt.plot(rx[0, 0], rx[0, 1], marker='o', color='deeppink', markersize=4, label='rx', linestyle='None')
        
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D scene")
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig(f"{name}.pdf")
    plt.show()

def plot2d_heatmap(data, x, y, vertices, title="2D Heatmap", min_val=None, max_val=None, tx=None, rx=None):
    """
    Plot a 2D scene with obstacles defined by vertices, optional transmitter (tx) and receiver (rx), and heatmap of data accross the scene.
    """
    plt.figure(figsize=(8,6))
    plt.pcolormesh(x, y, data, shading='auto', cmap='viridis', rasterized=True, fontsize=16)
    if min_val is not None and max_val is not None:
        plt.clim(min_val, max_val)
    plt.colorbar()
    for obj in vertices:
        x, y = zip(*obj)
        x += (x[0],)
        y += (y[0],)
        plt.plot(x, y, 'b-', linewidth=2)

    if tx is not None:
        tx = jnp.atleast_2d(tx)
        plt.plot(tx[:, 0], tx[:, 1], 'rx', markersize=5, fontsize=16)
    if rx is not None:
        rx = jnp.atleast_2d(rx)
        plt.plot(rx[:, 0], rx[:, 1], 'gx', markersize=5, fontsize=16)

    plt.xlabel("X-axis", fontsize=16)
    plt.ylabel("Y-axis", fontsize=16)
    plt.savefig(f"{title}.pdf")
    plt.show()

def plot2d_initialization(vertices, tx=None, rx=None, candidates=None):
    """
    Plot a 2D scene with obstacles defined by vertices, optional transmitter (tx) and receiver (rx), and candidate positions in order.
    """
    plt.figure(figsize=(6,6))
    for obj in vertices:
        x, y = zip(*obj)
        x += (x[0],)
        y += (y[0],)
        plt.plot(x, y, 'b-', linewidth=2)
    
    if tx is not None:
        tx = jnp.atleast_2d(tx)
        plt.plot(tx[:, 0], tx[:, 1], 'rx', markersize=7)
        plt.plot(tx[0, 0], tx[0, 1], 'rx', label='tx')

    if rx is not None:
        rx = jnp.atleast_2d(rx)
        plt.plot(rx[:, 0], rx[:, 1], marker='o', color='deeppink', markersize=4, linestyle='None')
        plt.plot(rx[0, 0], rx[0, 1], marker='o', color='deeppink', markersize=4, label='rx', linestyle='None')

    candidates = jnp.atleast_2d(candidates)
    for i, candidate in enumerate(candidates):
        plt.plot(candidate[0], candidate[1], 'bx', markersize=3)
        plt.text(candidate[0], candidate[1], str(i) + " ", fontsize=8, ha='right', label='tx candidate')

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D scene initialization")
    plt.legend()
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.savefig("initialization.pdf")
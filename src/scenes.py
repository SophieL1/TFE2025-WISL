import jax
from jax import numpy as jnp
from differt.geometry import TriangleMesh

def square_scene(l=10.0, w=10.0, h=10.0): 
    """
    Create a square scene with a mesh and 2D representation.
    """
    mesh = (TriangleMesh.box(length=l, width=w, height=h, with_top=False, with_floor=False)
        .set_assume_quads()
        .set_face_colors(key=jax.random.key(1234))
    )
    scene2d = [[[-l/2, -l/2], [l/2, -l/2], [l/2, l/2], [-l/2, l/2]]]

    return mesh, scene2d

def basic_scene(l=10.0, w=10.0, h=10.0, l1=5.0, l2=4.0): 
    """
    Create a basic scene (square with 2 walls) with a mesh and 2D representation.
    """
    mesh1 = (TriangleMesh.box(length=l, width=w, height=h, with_top=False, with_floor=False)
        .set_assume_quads()
        .set_face_colors(key=jax.random.key(1))
    )
    mesh2 = (TriangleMesh.plane(vertex_a=jnp.array([0.0, 2.5, 0.0]), vertex_b=jnp.array([0.0, 2.5, -1.0]), vertex_c=jnp.array([2.5, 2.5, 0.0]), side_length=l1)
        .set_assume_quads()
        .set_face_colors(key=jax.random.key(2))
    )
    mesh = mesh1.append(mesh2)

    mesh3 = (TriangleMesh.plane(vertex_a=jnp.array([1.0, -2.5, 0.0]), vertex_b=jnp.array([1.0, -2.5, -1.0]), vertex_c=jnp.array([2.5, -2.5, 0.0]), side_length=l2)
        .set_assume_quads()
        .set_face_colors(key=jax.random.key(3))
    )
    mesh = mesh.append(mesh3)
    
    scene2d = [[[-l/2, -l/2], [l/2, -l/2], [l/2, l/2], [-l/2, l/2]], 
               [[-l1/2, 2.5], [l1/2, 2.5]], 
               [[-l2/2+1.0, -2.5], [l2/2+1.0, -2.5]]]

    return mesh, scene2d


def build_random_rooms(key: int=1, l=10.0, w=10.0, h=10.0, l1=4.0, l2=4.0):
    """
    Build a square scene with partially random walls creating rooms in the left part until x=0.45 
    and in the right part until x=0.55 - corridor in the middle. 
    Random positions for the transmitter and the nb_receivers.
    """
    key1, key2, key3, key4 = jax.random.split(jax.random.PRNGKey(key), 4)
    mesh1 = (TriangleMesh.box(length=l, width=w, height=h, with_top=False, with_floor=False)
        .set_assume_quads()
        .set_face_colors(key=jax.random.key(1))
    )
    # Add a wall for room 1
    y_wall1 = jax.random.uniform(key1, minval=-4.0, maxval=-1.0)
    mesh2 = (TriangleMesh.plane(vertex_a=jnp.array([(-l+l1)/2, y_wall1, 0.0]), vertex_b=jnp.array([(-l+l1)/2, y_wall1, -1.0]), vertex_c=jnp.array([-l/2, y_wall1, 0.0]), side_length=l1)
        .set_assume_quads()
        .set_face_colors(key=jax.random.key(2))
    )
    mesh = mesh1.append(mesh2)

    # Add a wall for room 2
    y_wall2 = jax.random.uniform(key2, minval=1.0, maxval=4.0)
    mesh3 = (TriangleMesh.plane(vertex_a=jnp.array([(-l+l1)/2, y_wall2, 0.0]), vertex_b=jnp.array([(-l+l1)/2, y_wall2, -1.0]), vertex_c=jnp.array([-l/2, y_wall2, 0.0]), side_length=l1)
        .set_assume_quads()
        .set_face_colors(key=jax.random.key(3))
    )
    mesh = mesh.append(mesh3)

    # Add a wall for room 3
    y_wall3 = jax.random.uniform(key3, minval=-4.0, maxval=-1.0)
    mesh4 = (TriangleMesh.plane(vertex_a=jnp.array([(l-l2)/2, y_wall3, 0.0]), vertex_b=jnp.array([l-l2/2, y_wall3, -1.0]), vertex_c=jnp.array([l/2, y_wall3, 0.0]), side_length=l2)
        .set_assume_quads()
        .set_face_colors(key=jax.random.key(4))
    )
    mesh = mesh.append(mesh4)

    # Add a wall for room 4
    y_wall4 = jax.random.uniform(key4, minval=1.0, maxval=4.0)
    mesh5 = (TriangleMesh.plane(vertex_a=jnp.array([(l-l2)/2, y_wall4, 0.0]), vertex_b=jnp.array([(l-l2)/2, y_wall4, -1.0]), vertex_c=jnp.array([l/2, y_wall4, 0.0]), side_length=l2)
        .set_assume_quads()
        .set_face_colors(key=jax.random.key(5))
    )
    mesh = mesh.append(mesh5)

    # Add vertical wall 1
    mesh6 = (TriangleMesh.plane(vertex_a=jnp.array([-l//2+l1, y_wall1, 0.0]), vertex_b=jnp.array([-l//2+l1, y_wall1, -1.0]), vertex_c=jnp.array([-l//2+l1, y_wall1+0.5, 0.0]), side_length=2.0)
        .set_assume_quads()
        .set_face_colors(key=jax.random.key(6))
    )
    mesh = mesh.append(mesh6)

    # Add vertical wall 2
    mesh7 = (TriangleMesh.plane(vertex_a=jnp.array([-l//2+l1, y_wall2, 0.0]), vertex_b=jnp.array([-l//2+l1, y_wall2, -1.0]), vertex_c=jnp.array([-l//2+l1, y_wall2-0.5, 0.0]), side_length=2.0)
        .set_assume_quads()
        .set_face_colors(key=jax.random.key(7))
    )
    mesh = mesh.append(mesh7)

    # Add vertical wall 3
    mesh8 = (TriangleMesh.plane(vertex_a=jnp.array([l//2-l2, y_wall3, 0.0]), vertex_b=jnp.array([l//2-l2, y_wall3, -1.0]), vertex_c=jnp.array([l//2-l2, y_wall3+0.5, 0.0]), side_length=2.0)
        .set_assume_quads()
        .set_face_colors(key=jax.random.key(8))
    )
    mesh = mesh.append(mesh8)

    # Add vertical wall 4
    mesh9 = (TriangleMesh.plane(vertex_a=jnp.array([l//2-l2, y_wall4, 0.0]), vertex_b=jnp.array([l//2-l2, y_wall4, -1.0]), vertex_c=jnp.array([l//2-l2, y_wall4-0.5, 0.0]), side_length=2.0)
        .set_assume_quads()
        .set_face_colors(key=jax.random.key(9))
    )
    mesh = mesh.append(mesh9)
    
    scene2d = [[[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]], 
               [[-l/2, y_wall1], [-l/2+l1, y_wall1]],
                [[-l/2, y_wall2], [-l/2+l1, y_wall2]],
                [[l/2, y_wall3], [l/2-l2, y_wall3]],
                [[l/2, y_wall4], [l/2-l2, y_wall4]], 
                [[-l/2+l1, y_wall1-0.5], [-l/2+l1, y_wall1+0.5]],
                [[-l/2+l1, y_wall2-0.5], [-l/2+l1, y_wall2+0.5]],
                [[l/2-l2, y_wall3-0.5], [l/2-l2, y_wall3+0.5]],
                [[l/2-l2, y_wall4-0.5], [l/2-l2, y_wall4+0.5]]]

    return mesh, scene2d

def choose_receivers(key: int, nb_receivers: int, l: float = 10.0, w: float = 10.0, factor: int = 5, dist: float = 0.5, eps: float = 1e-2):
    """
    Choose `nb_receivers` random positions from a grid,
    with no two receivers closer than `dist`.

    Args:
        key: Random key for generating positions.
        nb_receivers: Number of receivers to place.
        l: Length of the area.
        w: Width of the area.
        factor: Factor to increase the number of candidate positions.
        dist: Minimum distance between receivers.
        eps: Small value to avoid placing receivers on the edges.

    Returns:
        A 2D array of shape (nb_receivers, 3) with the positions of the receivers.
    """
    # Create grid of candidate positions
    x = jnp.linspace(-l/2+eps, l/2-eps, nb_receivers * factor)
    y = jnp.linspace(-w/2+eps, w/2-eps, nb_receivers * factor)
    X, Y = jnp.meshgrid(x, y)
    Z = jnp.zeros_like(X)
    candidates = jnp.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

    key = jax.random.PRNGKey(key)
    receivers = []

    while len(receivers) < nb_receivers and len(candidates) > 0:
        # Sample one candidate
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (), 0, candidates.shape[0])
        selected = candidates[idx]
        receivers.append(selected)

        # Compute distances to all candidates and keep only those far enough
        dists = jnp.linalg.norm(candidates - selected, axis=1)
        mask = dists > dist
        candidates = candidates[mask]

    if len(receivers) < nb_receivers:
        raise ValueError(f"Could only place {len(receivers)} receivers with min dist {dist}; try increasing factor or reducing dist.")

    return jnp.stack(receivers)
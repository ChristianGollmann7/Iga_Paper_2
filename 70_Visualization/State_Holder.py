from dataclasses import dataclass

@dataclass
class StateHolder:
    pinn = None
    G = None
    boundary_conditions = None

    base = None
    dirichlet_boundary = None
    neumann_boundary = None
    dirichlet_indices = None
    control_points = None

import pytest
import numpy as np
from FrameSolver3D import FrameNode, FrameElement, FrameStructure

def setup_example_problem():
    """
    Set up the example problem described in the image.
    """
    # Material properties
    E = 1000  # Young's modulus
    nu = 0.3  # Poisson's ratio

    # Cross-sectional properties
    b = 0.5  # Width
    h = 1.0  # Height
    A = b * h  # Cross-sectional area
    Iy = (h * b**3) / 12  # Moment of inertia about y-axis
    Iz = (b * h**3) / 12  # Moment of inertia about z-axis
    J = Iy + Iz  # Polar moment of inertia

    # Create nodes
    node0 = FrameNode(0, 0, 0, 10.0)  # Node 0
    node1 = FrameNode(1, 15.0, 0, 10.0)  # Node 1
    node2 = FrameNode(2, 15.0, 0, 0)  # Node 2

    # Apply boundary conditions (fully fix nodes 0 and 2)
    node0.set_boundary_condition([True, True, True, True, True, True])  # Fix all DOFs
    node2.set_boundary_condition([True, True, True, True, True, True])  # Fix all DOFs

    # Apply a force at node 1 in the negative z-direction
    node1.apply_force([0, 0, -100, 0, 0, 0])  # [Fx, Fy, Fz, Mx, My, Mz]

    # Create elements
    element0 = FrameElement(0, node0, node1, E, nu, A, Iy, Iz, J)  # Element 0
    element1 = FrameElement(1, node1, node2, E, nu, A, Iy, Iz, J)  # Element 1

    # Create the structure and add nodes/elements
    structure = FrameStructure()
    structure.add_node(node0)
    structure.add_node(node1)
    structure.add_node(node2)
    structure.add_element(element0)
    structure.add_element(element1)

    return structure

def test_displacements():
    """
    Test the displacements at each node.
    """
    structure = setup_example_problem()
    displacements, _ = structure.solve()

    # Expected displacements (approximate values based on the problem)
    expected_displacements = {
        0: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Node 0 is fixed
        1: [0.0, 0.0, -0.1, 0.0, 0.0, 0.0],  # Node 1 has a displacement in z-direction
        2: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Node 2 is fixed
    }

    # Compare computed displacements with expected values
    for node_id, expected_disp in expected_displacements.items():
        for dof in range(6):
            computed_disp = displacements[structure.dof_map[(node_id, dof)]]
            assert np.isclose(computed_disp, expected_disp[dof], rtol=1e-2), (
                f"Displacement mismatch at Node {node_id}, DOF {dof}: "
                f"Expected {expected_disp[dof]}, got {computed_disp}"
            )

def test_reaction_forces():
    """
    Test the reaction forces at each node.
    """
    structure = setup_example_problem()
    _, reactions = structure.solve()

    # Expected reaction forces (approximate values based on the problem)
    expected_reactions = {
        0: [0.0, 0.0, 50.0, 0.0, 0.0, 0.0],  # Node 0 reaction in z-direction
        1: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Node 1 has no reaction (free to move)
        2: [0.0, 0.0, 50.0, 0.0, 0.0, 0.0],  # Node 2 reaction in z-direction
    }

    # Compare computed reactions with expected values
    for node_id, expected_reaction in expected_reactions.items():
        for dof in range(6):
            computed_reaction = reactions[structure.dof_map[(node_id, dof)]]
            assert np.isclose(computed_reaction, expected_reaction[dof], rtol=1e-2), (
                f"Reaction mismatch at Node {node_id}, DOF {dof}: "
                f"Expected {expected_reaction[dof]}, got {computed_reaction}"
            )

def test_global_stiffness_matrix():
    """
    Test the global stiffness matrix assembly.
    """
    structure = setup_example_problem()
    structure.assign_dofs()
    structure.assemble_global_stiffness()

    # Check that the global stiffness matrix is symmetric
    assert np.allclose(structure.global_stiffness, structure.global_stiffness.T), (
        "Global stiffness matrix is not symmetric."
    )

    # Check that the global stiffness matrix is positive semi-definite
    eigenvalues = np.linalg.eigvals(structure.global_stiffness)
    assert np.all(eigenvalues >= -1e-10), (
        "Global stiffness matrix is not positive semi-definite."
    )

def test_boundary_conditions():
    """
    Test that boundary conditions are applied correctly.
    """
    structure = setup_example_problem()
    structure.assign_dofs()
    structure.assemble_global_stiffness()
    structure.apply_boundary_conditions()

    # Check that fixed DOFs have zero displacement
    for node in structure.nodes.values():
        for local_dof, is_fixed in enumerate(node.boundary_conditions):
            if is_fixed:
                global_dof = structure.dof_map[(node.id, local_dof)]
                assert structure.global_stiffness[global_dof, global_dof] == 1, (
                    f"Boundary condition not applied correctly at Node {node.id}, DOF {local_dof}."
                )

def test_singular_matrix():
    """
    Test that the solver raises an error for a singular stiffness matrix.
    """
    structure = FrameStructure()

    # Create two nodes without any elements
    node0 = FrameNode(0, 0, 0, 0)
    node1 = FrameNode(1, 1, 0, 0)
    structure.add_node(node0)
    structure.add_node(node1)

    # Attempt to solve (should raise an error due to singular matrix)
    with pytest.raises(ValueError, match="Global stiffness matrix is singular."):
        structure.solve()
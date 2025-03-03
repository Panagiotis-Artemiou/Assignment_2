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

    # Apply boundary conditions ()
    node0.set_boundary_condition([True, True, True, True, True, True])  # Fix all DOFs
    node2.set_boundary_condition([True, True, True, False, False, False])  # Fix disp DOFs

    # Apply a force at node 1 in the negative z-direction
    node1.apply_force([0.1, 0.05, -0.07, 0.05, -0.1, 0.25])  # [Fx, Fy, Fz, Mx, My, Mz]

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

def test_example_problem():
    """
    Test the example problem and print the displacements and reactions.
    """
    # Set up the structure
    structure = setup_example_problem()

    # Solve for displacements and reactions
    displacements, reactions = structure.solve()

    # Print results
    print("Example Problem:")
    print("Displacements at each node:")
    for node_id, node in structure.nodes.items():
        disp_values = []
        for dof in range(6):  # Each node has up to 6 DOFs
            if (node_id, dof) in structure.dof_map:  # Check if DOF is still in the mapping
                disp_values.append(f"{displacements[structure.dof_map[(node_id, dof)]]:6.4e}")
            else:
                disp_values.append("Fixed")  # Indicate that this DOF was constrained

        print(f"Node {node_id}: " + ", ".join(disp_values))

    print("\nReaction forces at each node:")
    for node_id, node in structure.nodes.items():
        reaction_values = []
        for dof in range(6):  # Each node has up to 6 DOFs
            if (node_id, dof) in structure.dof_map:  # Check if DOF is still in the mapping
                reaction_values.append(f"{reactions[structure.dof_map[(node_id, dof)]]:6.4e}")
            else:
                reaction_values.append("Fixed")  # Indicate that this DOF was constrained

        print(f"Node {node_id}: " + ", ".join(reaction_values))

if __name__ == "__main__":
    test_example_problem()


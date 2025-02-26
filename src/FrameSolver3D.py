import numpy as np
from math_utils import (
    local_elastic_stiffness_matrix_3D_beam,
    rotation_matrix_3D,
    transformation_matrix_3D
)

class FrameNode:
    """
    Represents a node in the 3D frame structure.
    Attributes:
        id (int): Unique node identifier.
        coordinates (tuple): (x, y, z) coordinates of the node.
        forces (list): Applied forces and moments [Fx, Fy, Fz, Mx, My, Mz].
        boundary_conditions (list): Boolean list indicating fixed DOFs [Ux, Uy, Uz, Rx, Ry, Rz].
    """
    def __init__(self, node_id, x, y, z):
        self.id = node_id
        self.coordinates = (x, y, z)
        self.forces = [0.0] * 6  # Initialize forces/moments to zero
        self.boundary_conditions = [False] * 6  # Initialize all DOFs as free

    def apply_force(self, force_vector):
        """
        Apply a force/moment vector to the node.
        Args:
            force_vector (list): List of 6 elements [Fx, Fy, Fz, Mx, My, Mz].
        """
        if len(force_vector) != 6:
            raise ValueError("Force vector must have exactly 6 elements.")
        self.forces = force_vector

    def set_boundary_condition(self, bc_vector):
        """
        Set boundary conditions for the node.
        Args:
            bc_vector (list): List of 6 booleans [Ux, Uy, Uz, Rx, Ry, Rz].
        """
        if len(bc_vector) != 6:
            raise ValueError("Boundary condition vector must have exactly 6 elements.")
        self.boundary_conditions = bc_vector


class FrameElement:
    """
    Represents a 3D beam element connecting two nodes.
    Attributes:
        id (int): Unique element identifier.
        node1 (FrameNode): First end node.
        node2 (FrameNode): Second end node.
        material_props (dict): Material properties (E, nu).
        section_props (dict): Section properties (A, Iy, Iz, J).
    """
    def __init__(self, element_id, node1, node2, E, nu, A, Iy, Iz, J):
        self.id = element_id
        self.node1 = node1
        self.node2 = node2
        self.material_props = {"E": E, "nu": nu}
        self.section_props = {"A": A, "Iy": Iy, "Iz": Iz, "J": J}
        self.length = self._compute_length()

    def _compute_length(self):
        """
        Compute the length of the element.
        Returns:
            float: Length of the element.
        """
        x1, y1, z1 = self.node1.coordinates
        x2, y2, z2 = self.node2.coordinates
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    def local_stiffness_matrix(self):
        """
        Compute the local stiffness matrix for the element.
        Returns:
            np.ndarray: 12x12 local stiffness matrix.
        """
        E = self.material_props["E"]
        nu = self.material_props["nu"]
        A = self.section_props["A"]
        Iy = self.section_props["Iy"]
        Iz = self.section_props["Iz"]
        J = self.section_props["J"]
        L = self.length

        return local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)

    def global_stiffness_matrix(self):
        """
        Compute the global stiffness matrix for the element.
        Returns:
            np.ndarray: 12x12 global stiffness matrix.
        """
        x1, y1, z1 = self.node1.coordinates
        x2, y2, z2 = self.node2.coordinates
        gamma = rotation_matrix_3D(x1, y1, z1, x2, y2, z2)
        T = transformation_matrix_3D(gamma)
        k_local = self.local_stiffness_matrix()
        return T.T @ k_local @ T


class FrameStructure:
    """
    Represents the entire 3D frame structure.
    Attributes:
        nodes (dict): Dictionary of nodes {node_id: FrameNode}.
        elements (dict): Dictionary of elements {element_id: FrameElement}.
        dof_map (dict): Maps (node_id, local_dof) to global DOF indices.
        global_stiffness (np.ndarray): Global stiffness matrix.
        global_force (np.ndarray): Global force vector.
    """
    def __init__(self):
        self.nodes = {}
        self.elements = {}
        self.dof_map = {}
        self.global_stiffness = None
        self.global_force = None

    def add_node(self, node):
        """
        Add a node to the structure.
        Args:
            node (FrameNode): Node to add.
        """
        self.nodes[node.id] = node

    def add_element(self, element):
        """
        Add an element to the structure.
        Args:
            element (FrameElement): Element to add.
        """
        self.elements[element.id] = element

    def assign_dofs(self):
        """
        Assign global DOF numbers to each node's DOFs.
        """
        dof_counter = 0
        for node_id, node in sorted(self.nodes.items()):
            for local_dof in range(6):  # 6 DOFs per node
                self.dof_map[(node_id, local_dof)] = dof_counter
                dof_counter += 1

    def assemble_global_stiffness(self):
        """
        Assemble the global stiffness matrix.
        """
        num_dofs = len(self.nodes) * 6
        self.global_stiffness = np.zeros((num_dofs, num_dofs))

        for element in self.elements.values():
            k_global = element.global_stiffness_matrix()
            node1_id = element.node1.id
            node2_id = element.node2.id

            # Get global DOF indices for the element
            dofs = []
            for node_id in [node1_id, node2_id]:
                for local_dof in range(6):
                    dofs.append(self.dof_map[(node_id, local_dof)])

            # Place element stiffness into the global matrix
            for i in range(12):
                for j in range(12):
                    self.global_stiffness[dofs[i], dofs[j]] += k_global[i, j]

    def apply_boundary_conditions(self):
        """
        Apply boundary conditions to the global stiffness matrix and force vector.
        """
        for node in self.nodes.values():
            for local_dof, is_fixed in enumerate(node.boundary_conditions):
                if is_fixed:
                    global_dof = self.dof_map[(node.id, local_dof)]
                    # Zero out the row and column for the fixed DOF
                    self.global_stiffness[global_dof, :] = 0
                    self.global_stiffness[:, global_dof] = 0
                    # Set the diagonal to 1 to enforce zero displacement
                    self.global_stiffness[global_dof, global_dof] = 1
                    # Zero out the corresponding entry in the force vector
                    self.global_force[global_dof] = 0

    def solve(self):
        """
        Solve for displacements and reactions.
        Returns:
            displacements (np.ndarray): Nodal displacements.
            reactions (np.ndarray): Reaction forces.
        """
        self.assign_dofs()
        self.assemble_global_stiffness()

        # Assemble global force vector
        self.global_force = np.zeros(len(self.nodes) * 6)
        for node in self.nodes.values():
            for local_dof in range(6):
                global_dof = self.dof_map[(node.id, local_dof)]
                self.global_force[global_dof] = node.forces[local_dof]

        # Apply boundary conditions
        self.apply_boundary_conditions()

        # Solve for displacements
        try:
            displacements = np.linalg.solve(self.global_stiffness, self.global_force)
        except np.linalg.LinAlgError:
            raise ValueError("Global stiffness matrix is singular. Check boundary conditions.")

        # Compute reactions
        reactions = self.global_stiffness @ displacements - self.global_force

        return displacements, reactions
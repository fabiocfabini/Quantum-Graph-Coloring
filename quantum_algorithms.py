from typing import List, Tuple, Dict
from itertools import combinations

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, Aer

import numpy as np


class NKColoring:
    """
    Solves the Graph k-Coloring problem.
    """

    def __init__(self, num_nodes: int, num_colors: int, edges: List[Tuple[int, int]]):
        self.num_nodes = num_nodes
        self.num_colors = num_colors
        self.edges = edges

        self._build_circuit()

    @staticmethod
    def _simulate_circuit(circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        backend = Aer.get_backend('qasm_simulator')
        job = execute(circuit, backend=backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        return counts

    def solve(self, shots=1024):
        #TODO: Filter results that have less than num_colors colors
        return self._simulate_circuit(self.quantum_circuit, shots)

    def _define_registers(self):
        # Create a Quantum Register for each node with num_colors qubits and an ancilla qubit
        self.node_ancilla_registers: List[Tuple[QuantumRegister, QuantumRegister]] = []
        for node in range(self.num_nodes):
            tuple_register = (QuantumRegister(self.num_colors, f"node_{node}"), QuantumRegister(1, f"ancilla_{node}"))
            self.node_ancilla_registers.append(tuple_register)

        self.node_pairs = list(combinations(range(self.num_nodes), 2))

        # Create a Quantum Register for each pair of nodes with 1 qubit
        self.pairs_register: List = []
        for node1, node2 in self.node_pairs:
            self.pairs_register.append(QuantumRegister(1, f"pair_{node1}_{node2}"))

        # Create a Quantum Register for each edge of the graph with 1 qubit
        self.edge_register: List = []
        for node1, node2 in self.node_pairs:
            self.edge_register.append(QuantumRegister(1, f"edge_{node1}_{node2}"))

        # Create a Classical Register with num_nodes bits.
        self.classical_register = ClassicalRegister(self.num_nodes * self.num_colors, "classical")

    def _init_edges(self):
        for edge in self.edges:
            if edge in self.node_pairs:
                self.quantum_circuit.x(self.edge_register[self.node_pairs.index(edge)])
            elif (edge[1], edge[0]) in self.node_pairs:
                self.quantum_circuit.x(self.edge_register[self.node_pairs.index((edge[1], edge[0]))])
            else:
                raise Exception(f"Edge '{edge}' is not a valid edge.")

    def _generate_color_matrix(self):
        # Start with all possible color arrangements
        for node, ancilla in self.node_ancilla_registers:
            self.quantum_circuit.h(node)

        for node_colors, ancilla in self.node_ancilla_registers:

            # Remove all arrangements where one node has more than one color
            for n1, n2 in combinations(range(self.num_colors), 2):
                color1 = node_colors[n1]
                color2 = node_colors[n2]

                self.quantum_circuit.ccx(color1, color2, ancilla)
                self.quantum_circuit.cx(ancilla, color1)
                self.quantum_circuit.reset(ancilla)

            # Remove arrangement where node has no color
            self.quantum_circuit.x(node_colors)
            self.quantum_circuit.mct(node_colors, ancilla)
            self.quantum_circuit.x(node_colors)
            self.quantum_circuit.cx(ancilla, node_colors[-1])
            self.quantum_circuit.reset(ancilla)
        self.quantum_circuit.barrier()

    def _detect_color_conflicts(self):
        for n, (n1, n2) in enumerate(self.node_pairs):
            for i in range(self.num_colors):
                node1_color = self.node_ancilla_registers[n1][0][i]
                node2_color = self.node_ancilla_registers[n2][0][i]

                self.quantum_circuit.ccx(node1_color, node2_color, self.pairs_register[n])
        self.quantum_circuit.barrier()

    def _resolve_color_conflicts(self):
        for n, (edge, pair) in enumerate(zip(self.edge_register, self.pairs_register)):
            for node, ancilla in self.node_ancilla_registers:
                for i in range(self.num_colors):
                    node_color = node[i]

                    self.quantum_circuit.mct([edge, pair, node_color], ancilla)
                    self.quantum_circuit.cx(ancilla, node_color)
                    self.quantum_circuit.reset(ancilla)
        self.quantum_circuit.barrier()

    def _measure(self):
        for i, (node, _) in enumerate(self.node_ancilla_registers):
            self.quantum_circuit.measure(node, self.classical_register[i * self.num_colors:(i + 1) * self.num_colors])

    def _build_circuit(self):
        self._define_registers()

        self.quantum_circuit = QuantumCircuit(
            *[reg for reg_tuple in self.node_ancilla_registers for reg in reg_tuple], 
            *self.pairs_register, 
            *self.edge_register,
            self.classical_register
        )

        self._init_edges()

        self._generate_color_matrix()

        self._detect_color_conflicts()

        self._resolve_color_conflicts()

        self._measure()


class Grover2GC:
    """
    Solve the Graph 2-Coloring problem using Grover's algorithm.
    """
#TODO: Fix spelling error on "adjancency_matrix"
    def __init__(self, num_nodes: int, adjancency_matrix: List[List[int]], num_correct_answers: int):
        self.num_nodes = num_nodes
        self.edges = self._get_edges(adjancency_matrix)
        self.num_qubits = num_nodes * self.color_bits
        self.num_edges = int(num_nodes * (num_nodes - 1) / 2)

        self.edge_combinations = list(combinations(range(self.num_nodes), 2))
        self.num_iterations = round(((np.pi/2)/np.arccos(np.sqrt((num_nodes**2-num_correct_answers)/num_nodes**2)) - 1)/2)

        self._build_circuit()

    @staticmethod
    def _simulate_circuit(circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        backend = Aer.get_backend('qasm_simulator')
        job = execute(circuit, backend=backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        return counts

    def solve(self, shots: int = 1024) -> Dict[str, int]:
        return self._simulate_circuit(self.quantum_circuit, shots)

    def _get_edges(self, adjancency_matrix: List[List[int]]) -> List[Tuple[int, int]]:
        edges = []
        for i in range(len(adjancency_matrix)):
            for j in range(i + 1, len(adjancency_matrix[0])):
                if adjancency_matrix[i][j] == 1:
                    edges.append((i, j))
        return edges

    def _disconnected_nodes(self):
        for n, (n1, n2) in enumerate(self.edge_combinations):
            if (n1, n2) in self.edges:
                continue
            self.quantum_circuit.x(self.edges_register[n])

    def _r(self):
        self.quantum_circuit.h(self.nodes_register)
        self.quantum_circuit.barrier()

    def _r_dg(self):
        self.quantum_circuit.h(self.nodes_register)
        self.quantum_circuit.barrier()

    def _oracle(self):
        for n, (n1, n2) in enumerate(self.edge_combinations):
            if (n1, n2) in self.edges:
                self.quantum_circuit.x(self.nodes_register[n1])
                self.quantum_circuit.ccx(self.nodes_register[n1], self.nodes_register[n2], self.edges_register[n])
                self.quantum_circuit.x(self.nodes_register[n1])

                self.quantum_circuit.x(self.nodes_register[n2])
                self.quantum_circuit.ccx(self.nodes_register[n1], self.nodes_register[n2], self.edges_register[n])
                self.quantum_circuit.x(self.nodes_register[n2])

        self.quantum_circuit.mct(self.edges_register, self.phase_register[0])

        for n, (n1, n2) in enumerate(reversed(self.edge_combinations)):
            if (n1, n2) in self.edges:
                self.quantum_circuit.x(self.nodes_register[n2])
                self.quantum_circuit.ccx(self.nodes_register[n1], self.nodes_register[n2], self.edges_register[self.num_edges-1-n])
                self.quantum_circuit.x(self.nodes_register[n2])

                self.quantum_circuit.x(self.nodes_register[n1])
                self.quantum_circuit.ccx(self.nodes_register[n1], self.nodes_register[n2], self.edges_register[self.num_edges-1-n])
                self.quantum_circuit.x(self.nodes_register[n1])

    def _diffuser(self):
        self._r_dg()

        self.quantum_circuit.x(self.nodes_register)
        self.quantum_circuit.mct(self.nodes_register, self.phase_register)
        self.quantum_circuit.x(self.nodes_register)
        self.quantum_circuit.barrier()

        self._r()

    def _measure(self):
        self.quantum_circuit.measure(self.nodes_register, self.classical_register)

    def _build_circuit(self):
        self.nodes_register = QuantumRegister(self.num_qubits, 'nodes')
        self.edges_register = QuantumRegister(self.num_edges, 'edges')
        self.phase_register = QuantumRegister(1, 'phase')
        self.classical_register = ClassicalRegister(self.num_qubits, 'classical')
        self.quantum_circuit = QuantumCircuit(self.nodes_register, self.edges_register, self.phase_register, self.classical_register)

        self._disconnected_nodes()

        self.quantum_circuit.h(self.phase_register)
        self.quantum_circuit.z(self.phase_register)

        self._r()

        for i in range(max(1, self.num_iterations)):
            self._oracle()
            self._diffuser()

        self._measure()


class Grover3GC:
    """
    Solves the Graph 3-Coloring problem using Grover's algorithm.
    """
    color_bits = 2

    def __init__(self, num_nodes: int, adjancency_matrix: List[List[int]], num_correct_answers: int):
        self.num_nodes = num_nodes
        self.edges = self._get_edges(adjancency_matrix)
        self.num_qubits = num_nodes * self.color_bits
        self.num_edges = int(num_nodes * (num_nodes - 1) / 2)

        self.edge_combinations = list(combinations(range(self.num_nodes), 2))
        self.num_iterations = round(((np.pi/2)/np.arccos(np.sqrt((num_nodes**3-num_correct_answers)/num_nodes**3)) - 1)/2)

        self._build_circuit()

    @staticmethod
    def _simulate_circuit(circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        backend = Aer.get_backend('qasm_simulator')
        job = execute(circuit, backend=backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        return counts

    def solve(self, shots: int = 1024) -> Dict[str, int]:
        #TODO: Filter results that have only 2 colors
        return self._simulate_circuit(self.quantum_circuit, shots)

    def _get_edges(self, adjancency_matrix: List[List[int]]) -> List[Tuple[int, int]]:
        edges = []
        for i in range(len(adjancency_matrix)):
            for j in range(i + 1, len(adjancency_matrix)):
                if adjancency_matrix[i][j] == 1:
                    edges.append((i, j))
        return edges

    def _disconnected_nodes(self):
        for n, (n1, n2) in enumerate(self.edge_combinations):
            if (n1, n2) in self.edges:
                continue
            self.quantum_circuit.x(self.edges_register[n])

    def _r00(self):
        for node in range(self.num_nodes):
            self.quantum_circuit.ry(2*np.arcsin(np.sqrt(2/3)), self.nodes_register[node*self.color_bits])
            self.quantum_circuit.ch(self.nodes_register[node*self.color_bits], self.nodes_register[node*self.color_bits+1])
            self.quantum_circuit.x(self.nodes_register[node*self.color_bits+1])
        self.quantum_circuit.barrier()

    def _r00_dg(self):
        for node in range(self.num_nodes):
            self.quantum_circuit.x(self.nodes_register[node*self.color_bits+1])
            self.quantum_circuit.ch(self.nodes_register[node*self.color_bits], self.nodes_register[node*self.color_bits+1])
            self.quantum_circuit.ry(-2*np.arcsin(np.sqrt(2/3)), self.nodes_register[node*self.color_bits])
        self.quantum_circuit.barrier()

    def _oracle(self):
        for n, (n1, n2) in enumerate(self.edge_combinations):
            if (n1, n2) in self.edges:
                self.quantum_circuit.ccx(self.nodes_register[n1*self.color_bits], self.nodes_register[n2*self.color_bits+1], self.edges_register[n])
                self.quantum_circuit.ccx(self.nodes_register[n1*self.color_bits+1], self.nodes_register[n2*self.color_bits], self.edges_register[n])

        self.quantum_circuit.mct(self.edges_register, self.phase_register[0])

        for n, (n1, n2) in enumerate(reversed(self.edge_combinations)):
            if (n1, n2) in self.edges:
                self.quantum_circuit.ccx(self.nodes_register[n1*self.color_bits+1], self.nodes_register[n2*self.color_bits], self.edges_register[self.num_edges-1-n])
                self.quantum_circuit.ccx(self.nodes_register[n1*self.color_bits], self.nodes_register[n2*self.color_bits+1], self.edges_register[self.num_edges-1-n])
        self.quantum_circuit.barrier()

    def _diffuser(self):
        self._r00_dg()

        self.quantum_circuit.x(self.nodes_register)
        self.quantum_circuit.mct(self.nodes_register, self.phase_register)
        self.quantum_circuit.x(self.nodes_register)
        self.quantum_circuit.barrier()

        self._r00()

    def _measure(self):
        self.quantum_circuit.measure(self.nodes_register, self.classical_register)

    def _build_circuit(self):
        self.nodes_register = QuantumRegister(self.num_qubits, 'nodes')
        self.edges_register = QuantumRegister(self.num_edges, 'edges')
        self.phase_register = QuantumRegister(1, 'phase')
        self.classical_register = ClassicalRegister(self.num_qubits, 'classical')
        self.quantum_circuit = QuantumCircuit(self.nodes_register, self.edges_register, self.phase_register, self.classical_register)

        self._disconnected_nodes()

        self.quantum_circuit.h(self.phase_register)
        self.quantum_circuit.z(self.phase_register)

        self._r00()

        for i in range(self.num_iterations):
            self._oracle()
            self._diffuser()

        self._measure()

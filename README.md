# Quantum Graph Coloring

This is a project for the course "Quantum Computing" at the University of Minho. The goal is to implement a quantum algorithm for the graph coloring problem.

Graph coloring is a problem in which we have to color the vertices of a graph with a limited number of colors, such that no two adjacent vertices have the same color. The problem is NP-complete, which means that there is no polynomial time algorithm for it. However, there are some heuristics that can be used to solve it in a reasonable amount of time. In order to test advance my knowledge in quantum computing, I decided to implement a quantum algorithm for this problem. The algorithm is based on the following papers:

- [Quantum Optimization for the Graph Coloring Problem with Space-Efficient Embedding](https://arxiv.org/pdf/2009.07314.pdf)
- [Hybrid quantum-classical algorithms for approximate graph coloring](https://quantum-journal.org/papers/q-2022-03-30-678/pdf/)
- [Graph Coloring With Quantum Annealing](https://arxiv.org/pdf/2012.04470.pdf)

### Implementation

I implemented 4 quantum algorithms that solved the graph coloring problem. These include:

- A graph specific quantum circuit that analyzes the graph structure. This is the least interesting algorithm, since it is not a quantum algorithm, but rather a classical one. It is designed to work with a single graph, and it is not scalable. It is included in the project for comparison purposes.

- A General Quantum Circuit that can be used to solve any graph coloring problem. It can be used with any graph, but it is not scalable since the number of qubits required to solve the problem grows exponentially with the number of vertices in the graph rendering it impractical for large graphs.

- Grover's algorithm, which is a quantum algorithm that can be used to obtain quadratic speedup over classical algorithms. There are 2 implementations of this algorithm. The first one is a 2-coloring algorithm, and the second one is a 3-coloring algorithm. Both of these algorithms are scalable as they take advantage of the Quantum Interference Effect and Quantum Superposition to obtain a quadratic speedup while using a quadratic number of qubits. Even though the 2-coloring algorithm is scalable, it is not the most efficient algorithm for the problem, since it is possible to solve the problem classically in linear time.

## Installation

The dependencies are listed in the `requirements.txt` file. To install them, run:

```bash
pip install -r requirements.txt
```

## Usage

Simply walk through the `main.ipynb` notebook. It contains all the code and explanations for the project.
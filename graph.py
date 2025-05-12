import dataclasses
from typing import Any, Generator, List, Tuple

type Node = int
type Path = List[Node]
type PathCost = Tuple[Path, float]
type DirectedGraph = dict[Node : list[tuple[Node, float]]]


@dataclasses.dataclass
class Arc:
    source: int
    target: int
    cost: float

    def __repr__(self):
        return f"Arc(source={self.source}, target={self.target}, cost={self.cost})"


class Graph:
    def __init__(self, node_count: int, aliases: list[str], shortest_path: float, bidirectional: bool = True):
        self.paths: DirectedGraph = {i: [] for i in range(node_count)}
        self.aliases: list[str] = aliases
        self.node_count = node_count
        self.bidirectional = bidirectional
        self.shortest_path = shortest_path
        self._from_alias = {alias: i for i, alias in enumerate(self.aliases)}

    def alias(self, node: Node | str):
        if isinstance(node, int):
            return self.aliases[node]
        return node

    def unalias(self, node: Node | str):
        if isinstance(node, str):
            return self._from_alias[node]
        return node

    def add_arc(self, source: Node | str, target: Node | str, length: float):
        self.paths[self.unalias(source)].append((self.unalias(target), length))
        if self.bidirectional:
            self.paths[self.unalias(target)].append((self.unalias(source), length))

    def remove_arc(self, source: Node | str, target: Node | str):
        for path in self.paths[self.unalias(source)].copy():
            if path[0] == target:
                self.paths[self.unalias(source)].remove(path)

    def adjacent(self, node: Node | str) -> Generator[Arc, Any, None]:
        for dest, cost in self.paths[self.unalias(node)]:
            yield Arc(node, dest, cost)

    def arc(self, source: Node | str, target: Node | str) -> float:
        return [
            arc.cost
            for arc in self.adjacent(source)
            if arc.target == self.unalias(target)
        ][0]

    def all_arcs(self):
        all_arcs: list[Arc] = []
        for node in range(self.node_count):
            all_arcs += self.adjacent(node)
        return all_arcs


# Simple Graph
simple_graph: Graph = Graph(5, [chr(65 + i) for i in range(5)], shortest_path=23.6)

simple_graph.add_arc("A", "B", 4.5)
simple_graph.add_arc("B", "C", 11.6)
simple_graph.add_arc("B", "D", 24.3)
simple_graph.add_arc("B", "E", 27.2)
simple_graph.add_arc("C", "D", 13.5)
simple_graph.add_arc("C", "E", 7.5)

# Complex Graph
complex_graph: Graph = Graph(20, [chr(65 + i) for i in range(20)], shortest_path=23.6)

complex_graph.add_arc("A", "C", 4.5)
complex_graph.add_arc("B", "C", 0.8)
complex_graph.add_arc("B", "E", 11.7)
complex_graph.add_arc("B", "F", 2.7)
complex_graph.add_arc("C", "D", 1)
complex_graph.add_arc("D", "G", 2.7)
complex_graph.add_arc("D", "H", 5.9)
complex_graph.add_arc("E", "F", 5)
complex_graph.add_arc("E", "I", 3.2)
complex_graph.add_arc("F", "J", 4.2)
complex_graph.add_arc("G", "H", 4.2)
complex_graph.add_arc("G", "K", 1.9)
complex_graph.add_arc("H", "M", 1.8)
complex_graph.add_arc("I", "J", 0.9)
complex_graph.add_arc("I", "P", 6.4)
complex_graph.add_arc("J", "Q", 1.4)
complex_graph.add_arc("K", "L", 2.6)
complex_graph.add_arc("K", "O", 1.3)
complex_graph.add_arc("L", "M", 1.6)
complex_graph.add_arc("L", "R", 2.6)
complex_graph.add_arc("M", "N", 2.8)
complex_graph.add_arc("P", "S", 11)
complex_graph.add_arc("Q", "R", 8)
complex_graph.add_arc("R", "S", 6.8)
complex_graph.add_arc("S", "T", 1.5)


# Dynamic Graph
dynamic_graph: Graph = Graph(20, [chr(65 + i) for i in range(20)], shortest_path=29.9)

dynamic_graph.add_arc("A", "C", 4.5)
dynamic_graph.add_arc("B", "C", 0.8)
dynamic_graph.add_arc("B", "E", 11.7)
dynamic_graph.add_arc("B", "F", 2.7)
dynamic_graph.add_arc("C", "D", 1)
dynamic_graph.add_arc("D", "G", 2.7)
dynamic_graph.add_arc("D", "H", 5.9)
dynamic_graph.add_arc("E", "F", 5)
dynamic_graph.add_arc("E", "I", 3.2)
dynamic_graph.add_arc("F", "J", 4.2)
dynamic_graph.add_arc("G", "H", 4.2)
dynamic_graph.add_arc("I", "J", 0.9)
dynamic_graph.add_arc("I", "P", 6.4)
dynamic_graph.add_arc("J", "Q", 1.4)
dynamic_graph.add_arc("L", "R", 2.6)
dynamic_graph.add_arc("P", "S", 11)
dynamic_graph.add_arc("Q", "R", 8)
dynamic_graph.add_arc("R", "S", 6.8)
dynamic_graph.add_arc("S", "T", 1.5)

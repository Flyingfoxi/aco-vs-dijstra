import time
from typing import Any, Generator, List, Set

import numpy as np
from numpy.random import choice

from graph import complex_graph, Graph, Node, Path, PathCost, simple_graph


class AntColony:
    def __init__(
        self,
        graph: Graph,
        n_ants: int,
        n_best: int,
        decay: float,
        alpha: float = 1,
        beta: float = 2,
        gamma: float = float("1e-9"),
        n_iter: int | None = None,
    ):
        self.graph = graph
        self.pheromone = np.zeros(shape=(graph.node_count, graph.node_count))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iter = n_iter or 10
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.nodes = list(range(graph.node_count))

        for arc in self.graph.all_arcs():
            self.pheromone[arc.source, arc.target] = 1 / self.graph.node_count

    def run(
        self, start: Node | str, end: Node | str, n_iter: int | None = None
    ) -> Generator[tuple[list[Any], float, int], Any, Any]:
        if n_iter is None:
            n_iter = self.n_iter
        best: PathCost = ([], np.inf)
        for it in range(n_iter):
            paths = [
                self._build_path(self.graph.unalias(start), self.graph.unalias(end))
                for _ in range(self.n_ants)
            ]
            paths.sort(key=lambda x: x[1])
            self._spread_pheromone(paths[: self.n_best])
            if paths[0][1] < best[1]:
                best = paths[0]
                yield [self.graph.alias(int(i)) for i in best[0]], best[1], it
            self.pheromone *= self.decay

    def _build_path(self, start: Node, end: Node) -> PathCost:
        path: Path = [start]
        visited: Set[Node] = {start}
        total_cost = 0
        current = start
        while current != end:
            move = self._select_move(current, visited)
            if move is None:
                return [], float("inf")
            path.append(move)
            total_cost += self.graph.arc(current, move)
            visited.add(move)
            current = move
        return path, total_cost

    def _select_move(self, current: Node, visited: Set[Node]) -> Node | None:
        arcs: list[int] = list(map(lambda x: x.target, self.graph.adjacent(current)))
        lengths: list[float] = list(map(lambda x: x.cost, self.graph.adjacent(current)))

        pher = np.copy(self.pheromone[current])
        pher[list(visited)] = 0
        heur = np.array(
            [
                (lengths[arcs.index(i)] if i in arcs else 0)
                for i in range(self.graph.node_count)
            ]
        )
        scores = pher**self.alpha * heur**self.beta
        prob = scores / scores.sum()
        if scores.sum() == 0:
            return None
        return choice(self.nodes, p=prob)

    def _spread_pheromone(self, paths: List[PathCost]) -> None:
        for path, cost in paths:
            if cost == np.inf:
                continue
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += (5.0 / self.decay) / (
                    cost * self.n_ants
                )

    def update_graph(self, new_graph: Graph):
        self.graph = new_graph


if __name__ == "__main__":
    perf = time.perf_counter()
    aco = AntColony(complex_graph, n_ants=25, n_best=2, decay=0.9)
    path, cost = aco.run("A", "T", n_iter=50)
    print("Path:", path)
    print("Cost:", cost)
    print("Took:", time.perf_counter() - perf, "s")

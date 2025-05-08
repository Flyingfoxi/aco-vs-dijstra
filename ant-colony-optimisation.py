import numpy as np
from numpy.random import choice
from typing import List, Tuple, Set, TypeAlias

Node: TypeAlias = int
Path: TypeAlias = List[Node]
PathCost: TypeAlias = Tuple[Path, float]


class AntColony:
    def __init__(
        self,
        graph: np.ndarray,
        n_ants: int,
        n_best: int,
        decay: float,
        alpha: float = 1,
        beta: float = 2,
        gamma: float = float("1e-9"),
        n_iter: int | None = None,
    ):
        self.graph = graph
        self.pheromone = np.ones(graph.shape) / len(self.graph)
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iter = n_iter or 10
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.nodes = list(range(len(graph)))

    def run(self, start: Node, end: Node, n_iter: int | None = None) -> PathCost:
        if n_iter is None:
            n_iter = self.n_iter
        best: PathCost = ([], np.inf)
        for _ in range(n_iter):
            paths = [self._build_path(start, end) for _ in range(self.n_ants)]
            paths.sort(key=lambda x: x[1])
            self._spread_pheromone(paths[: self.n_best])
            if paths[0][1] < best[1]:
                best = paths[0]
            self.pheromone *= self.decay
        return [int(i) for i in best[0]], best[1]

    def _build_path(self, start: Node, end: Node) -> PathCost:
        path: Path = [start]
        visited: Set[Node] = {start}
        total_cost = 0
        current = start
        while current != end:
            move = self._select_move(current, visited)
            path.append(move)
            total_cost += self.graph[current, move]
            visited.add(move)
            current = move
        return path, total_cost

    def _select_move(self, current: Node, visited: Set[Node]) -> Node | None:
        pher = np.copy(self.pheromone[current])
        pher[list(visited)] = self.gamma
        heur = np.where(np.isnan(self.graph[current]), 0, 1.0 / self.graph[current])
        scores = pher**self.alpha * heur**self.beta
        prob = scores / scores.sum()
        return choice(self.nodes, p=prob)

    def _spread_pheromone(self, paths: List[PathCost]) -> None:
        for path, cost in paths:
            if cost == np.inf:
                continue
            for i in range(len(path) - 1):
                self.pheromone[path[i], path[i + 1]] += (5.0 / self.decay) / (cost * self.n_ants)


if __name__ == "__main__":
    g = np.array([
        [np.nan, 4.5, np.nan, np.nan, np.nan],
        [4.5, np.nan, 11.6, 24.3, 27.2],
        [np.nan, 11.6, np.nan, 13.5, 7.5],
        [np.nan, 24.3, 13.5, np.nan, 17.8],
        [np.nan, 27.2, 7.5, 17.8, np.nan]
    ])
    aco = AntColony(g, n_ants=2, n_best=10, decay=0.9)
    path, cost = aco.run(0, 4, n_iter=5)
    print("Path:", path)
    print("Cost:", cost)

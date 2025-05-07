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
        n_iter: int,
        decay: float,
        alpha: float = 1,
        beta: float = 2,
    ):
        self.graph = graph
        self.pheromone = np.ones(graph.shape)
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iter = n_iter
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.nodes = list(range(len(graph)))

    def run(self, start: Node, end: Node) -> PathCost:
        best: PathCost = ([], np.inf)
        for _ in range(self.n_iter):
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
            if move is None:
                return [], np.inf
            path.append(move)
            total_cost += self.graph[current, move]
            visited.add(move)
            current = move
        return path, total_cost

    def _select_move(self, current: Node, visited: Set[Node]) -> Node | None:
        pher = np.copy(self.pheromone[current])
        pher[list(visited)] = 0
        heur = np.where(self.graph[current] > 0, 1.0 / self.graph[current], 0)
        scores = pher**self.alpha * heur**self.beta
        if scores.sum() == 0:
            return None
        prob = scores / scores.sum()
        return choice(self.nodes, p=prob)

    def _spread_pheromone(self, paths: List[PathCost]) -> None:
        for path, cost in paths:
            if cost == np.inf:
                continue
            for i in range(len(path) - 1):
                self.pheromone[path[i], path[i + 1]] += 1.0 / cost


if __name__ == "__main__":
    g = np.array(
        [
            [np.inf, 1, 2, np.inf],
            [1, np.inf, np.inf, 4],
            [2, np.inf, np.inf, 1],
            [0, 4, 1, np.inf],
        ]
    )
    aco = AntColony(g, n_ants=10, n_best=3, n_iter=10, decay=0.9)
    path, cost = aco.run(0, 3)
    print("Path:", path)
    print("Cost:", cost)

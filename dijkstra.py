import heapq
from dataclasses import dataclass


@dataclass
class Edge:
    to: int
    dist: float


def dijkstra(graph: list[list[Edge]], start: int, end: int) -> float:
    distances: list[float] = [float("inf") for _ in range(len(graph))]
    priority_queue: list[tuple[float, int]] = []
    distances[start] = 0

    for e in graph[start]:
        heapq.heappush(priority_queue, (e.dist, e.to))

    while len(priority_queue):
        dist, to = heapq.heappop(priority_queue)

        if to == end:
            return dist

        if distances[to] != float("inf"):
            continue

        distances[to] = dist
        for e in graph[to]:
            if distances[e.to] == float("inf"):
                heapq.heappush(priority_queue, (dist + e.dist, e.to))

    return float("inf")


def main():
    graph: list[list[Edge]] = [
        [Edge(1, 3.7), Edge(2, 1.0)],
        [Edge(0, 3.7), Edge(2, 0.5)],
        [Edge(0, 1.0), Edge(1, 0.5)],
    ]
    print("Shortest Path:", dijkstra(graph, 0, 2))


if __name__ == "__main__":
    main()

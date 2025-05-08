import heapq
from dataclasses import dataclass

import numpy as np


def get_path(trace: list[int], point: int, target: int) -> list[int]:
    result = [point]
    while point != target:
        point = trace[point]
        result.append(point)
    result.reverse()
    return result


def dijkstra(graph: np.ndarray, start: int, end: int) -> tuple[list[int], float]:
    distances: list[float] = [np.inf for _ in range(len(graph))]
    prev_node: list[int] = [-1 for _ in range(len(graph))]
    priority_queue: list[tuple[float, int, int]] = []

    distances[start] = 0
    prev_node[start] = start

    for i, e in enumerate(graph[start]):
        if not np.isnan(e):
            heapq.heappush(priority_queue, (e, i, start))

    while len(priority_queue):
        dist, to, prev = heapq.heappop(priority_queue)

        if to == end:
            prev_node[to] = prev
            return get_path(prev_node, end, start), dist

        if not np.isinf(distances[to]):
            continue

        distances[to] = dist
        prev_node[to] = prev

        for i, e in [(i, e) for i, e in enumerate(graph[to]) if not np.isnan(i)]:
            if np.isinf(distances[i]):
                heapq.heappush(priority_queue, (dist + e, i, to))

    return [], np.inf


def main():
    g = np.array([
        [np.nan, 4.5, np.nan, np.nan, np.nan],
        [4.5, np.nan, 11.6, 24.3, 27.2],
        [np.nan, 11.6, np.nan, 13.5, 7.5],
        [np.nan, 24.3, 13.5, np.nan, 17.8],
        [np.nan, 27.2, 7.5, 17.8, np.nan]
    ])
    path, cost = dijkstra(g, 0, 4)
    print("Path:", path)
    print("Cost:", cost)


if __name__ == "__main__":
    main()

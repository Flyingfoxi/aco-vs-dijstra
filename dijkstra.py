import heapq
import time

import numpy as np

from graph import Graph, Node, PathCost, simple_graph


def get_path(trace: list[int], point: int, target: int) -> list[int]:
    result = [point]
    while point != target:
        point = trace[point]
        result.append(point)
    result.reverse()
    return result


def dijkstra(graph: Graph, start: Node | str, end: Node | str) -> PathCost:
    start = graph.unalias(start)
    end = graph.unalias(end)

    distances: list[float] = [-1 for _ in range(graph.node_count)]
    prev_node: list[int] = [-1 for _ in range(graph.node_count)]
    priority_queue: list[tuple[float, int, int]] = []

    distances[start] = 0
    prev_node[start] = start

    for arc in graph.adjacent(start):
        heapq.heappush(priority_queue, (arc.cost, arc.source, arc.target))

    while len(priority_queue):
        dist, source, target = heapq.heappop(priority_queue)

        if target == end:
            prev_node[end] = source
            return [graph.alias(i) for i in get_path(prev_node, target, start)], dist

        if distances[target] != -1:
            continue

        distances[target] = dist
        prev_node[target] = source

        for edge in graph.adjacent(target):
            if distances[edge.target] == -1:
                heapq.heappush(
                    priority_queue, (dist + edge.cost, edge.source, edge.target)
                )

    return [], np.inf


if __name__ == "__main__":
    perf = time.perf_counter()
    path, cost = dijkstra(simple_graph, "A", "E")
    print("Path:", path)
    print("Cost:", cost)
    print("Took:", time.perf_counter() - perf, "s")

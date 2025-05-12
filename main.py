import time

from ant_colony import AntColony
from dijkstra import dijkstra
from graph import complex_graph, simple_graph, dynamic_graph

MAX_ACO_DIFFERENCE = 2


def main():
    print("\n### Dijkstra's Algorithm ###")

    perf = time.perf_counter()
    path, cost = dijkstra(simple_graph, "A", "E")
    print("-- Simple Graph --")
    print(f"[path]: {', '.join(list(map(str, path)))}")
    print(f"[length]: {cost}")
    print(f"[time]: {time.perf_counter() - perf:0.6f}")

    perf = time.perf_counter()
    path, cost = dijkstra(complex_graph, "A", "T")
    print("-- Complex Graph --")
    print(f"[path]: {', '.join(list(map(str, path)))}")
    print(f"[length]: {cost:0.1f}")
    print(f"[time]: {time.perf_counter() - perf:0.6f}")

    perf = time.perf_counter()
    path, cost = dijkstra(dynamic_graph, "A", "T")
    print("-- Dynamic Graph --")
    print(f"[path]: {', '.join(list(map(str, path)))}")
    print(f"[length]: {cost:0.1f}")
    print(f"[time]: {time.perf_counter() - perf:0.6f}")

    print("\n### Ant Colony Optimisation ###")

    perf = time.perf_counter()
    aco = AntColony(simple_graph, n_ants=10, n_best=2, decay=0.9)
    iteration = None
    for path, cost, iteration in aco.run("A", "E", n_iter=25):
        if abs(cost - complex_graph.shortest_path) < MAX_ACO_DIFFERENCE:
            break
    print("-- Simple Graph --")
    print(f"[path]: {', '.join(list(map(str, path)))}")
    print(f"[length]: {cost}")
    print(f"[iter]: {iteration}")
    print(f"[time]: {time.perf_counter() - perf:0.6f}")

    perf = time.perf_counter()
    aco = AntColony(complex_graph, n_ants=50, n_best=10, decay=1)
    iteration = None
    for path, cost, iteration in aco.run("A", "T", n_iter=250):
        if abs(cost - complex_graph.shortest_path) < MAX_ACO_DIFFERENCE:
            break
    print("-- Complex Graph --")
    print(f"[path]: {', '.join(list(map(str, path)))}")
    print(f"[length]: {cost:0.1f}")
    print(f"[iter]: {iteration}")
    print(f"[time]: {time.perf_counter() - perf:0.6f}")

    perf = time.perf_counter()
    aco.update_graph(dynamic_graph)
    aco.gamma = 0.1e-9
    aco.n_ants = 10
    aco.n_best = 1
    iteration = None
    for path, cost, iteration in aco.run("A", "T", n_iter=100):
        if abs(cost - complex_graph.shortest_path) < MAX_ACO_DIFFERENCE:
            break

    print("-- Dynamic Graph --")
    print(f"[path]: {', '.join(list(map(str, path)))}")
    print(f"[length]: {cost:0.1f}")
    print(f"[iter]: {iteration}")
    print(f"[time]: {time.perf_counter() - perf:0.6f}")


if __name__ == "__main__":
    main()

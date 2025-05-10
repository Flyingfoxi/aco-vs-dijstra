import time

from ant_colony import AntColony
from dijkstra import dijkstra
from graph import complex_graph, simple_graph


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
    print("-- Complex Graph -- ")
    print(f"[path]: {', '.join(list(map(str, path)))}")
    print(f"[length]: {cost:0.1f}")
    print(f"[time]: {time.perf_counter() - perf:0.6f}")

    print("\n### Ant Colony Optimisation ###")

    perf = time.perf_counter()
    aco = AntColony(simple_graph, n_ants=4, n_best=2, decay=0.7)
    path, cost = aco.run("A", "E", n_iter=5)
    print("-- Simple Graph --")
    print(f"[path]: {', '.join(list(map(str, path)))}")
    print(f"[length]: {cost}")
    print(f"[time]: {time.perf_counter() - perf:0.6f}")

    perf = time.perf_counter()
    aco = AntColony(complex_graph, n_ants=12, n_best=6, decay=0.9)
    path, cost = aco.run("A", "T", n_iter=50)
    print("-- Complex Graph -- ")
    print(f"[path]: {', '.join(list(map(str, path)))}")
    print(f"[length]: {cost:0.1f}")
    print(f"[time]: {time.perf_counter() - perf:0.6f}")


if __name__ == "__main__":
    main()

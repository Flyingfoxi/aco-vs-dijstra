import time

import matplotlib.pyplot as plt

import dijkstra
from ant_colony import AntColony
from graph import complex_graph, dynamic_graph, simple_graph
from main import MAX_ACO_DIFFERENCE


def get_time_data(iterations: int) -> list[list[float]]:
    result_data: list[list[float]] = [
        [-1] * iterations,
        [-1] * iterations,
        [-1] * iterations,
        [-1] * iterations,
        [-1] * iterations,
        [-1] * iterations,
    ]

    # ACO
    print("processing ACO - simple graph")
    for i in range(0, iterations):
        perf = time.perf_counter()
        aco = AntColony(simple_graph, n_ants=10, n_best=2, decay=0.9)
        for path, cost, iteration in aco.run("A", "E", n_iter=25):
            if abs(cost - complex_graph.shortest_path) < MAX_ACO_DIFFERENCE:
                break
        result_data[0][i] = float(time.perf_counter() - perf)

    print("processing ACO - complex + dynamic")
    for i in range(0, iterations):
        perf = time.perf_counter()
        aco = AntColony(complex_graph, n_ants=50, n_best=10, decay=1)
        for path, cost, iteration in aco.run("A", "T", n_iter=250):
            if abs(cost - complex_graph.shortest_path) < MAX_ACO_DIFFERENCE:
                break
        result_data[1][i] = float(time.perf_counter() - perf)

        perf = time.perf_counter()
        aco.update_graph(dynamic_graph)
        aco.gamma = 0.1e-9
        aco.n_ants = 10
        aco.n_best = 1
        for path, cost, iteration in aco.run("A", "T", n_iter=150):
            if abs(cost - complex_graph.shortest_path) < MAX_ACO_DIFFERENCE:
                break
        result_data[2][i] = float(time.perf_counter() - perf)

    # Dijkstra
    print("processing Dijkstra")
    for di, start, end, graph in [
        (3, "A", "E", simple_graph),
        (4, "A", "T", complex_graph),
        (5, "A", "T", dynamic_graph),
    ]:
        for i in range(0, iterations):
            perf = time.perf_counter()
            dijkstra.dijkstra(graph, start, end)
            result_data[di][i] = float(time.perf_counter() - perf)

    return result_data


def create_graph() -> None:
    test_data = get_time_data(100)
    fig, ax = plt.subplots(figsize=(12, 9))

    plt.title("ACO & Dijkstra speed comparison")
    plt.ylabel("Duration [s]")

    ax.plot(test_data[0], label="ACO simple graph")
    ax.plot(test_data[1], label="ACO complex graph")
    ax.plot(test_data[2], label="ACO dynamic graph")
    ax.plot(test_data[3], label="Dijkstra simple graph")
    ax.plot(test_data[4], label="Dijkstra complex graph")
    ax.plot(test_data[5], label="Dijkstra dynamic graph")

    ax.legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    create_graph()

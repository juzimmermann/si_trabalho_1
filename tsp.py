import random as rd
from math import exp
import copy
from plot_results import plot_results

# constantes para ag_search
mut_ratio = 0.15  # de 0 a 1 - taxa de ocorrência de mutação
min_cost = 1  # peso mínimo de arestas
max_cost = 100  # peso máximo de arestas
##


def g_create(n_cities):  # Gera a matriz de adjacências
    new_g = []
    for i in range(n_cities):
        row = []
        for j in range(n_cities):
            row.append(0)
        new_g.append(row)
    for i in range(n_cities):
        new_g[i][i] = 0
        for j in range(i + 1, n_cities):
            new_g[i][j] = rd.randint(min_cost, max_cost)
            new_g[j][i] = new_g[i][j]
    return new_g


def g_print(tsp_g, n_cities):  # Exibe a matriz
    for i in range(n_cities):
        print()
        for j in range(n_cities):
            print("{:<4}".format(tsp_g[i][j]), end=" ")


def swap(route):
    """Swaps two randon vertices in the route"""
    pos1 = rd.randint(0, len(route) - 1)
    pos2 = rd.choice([j for j in range(0, len(route)) if j != pos1])

    route[pos1], route[pos2] = route[pos2], route[pos1]

    altered_route = route.copy()
    return altered_route


def evaluate_cost(tsp_graph, route):
    """Calculates the total cost for the provided route"""

    cost = 0
    for idx, elem in enumerate(route):
        if idx + 1 < len(route):
            cost += tsp_graph[elem][route[idx + 1]]

    cost += tsp_graph[route[len(route) - 1]][route[0]]
    return cost


def schedule(control, temp):
    """Implementation for the function that decrements 'temperature',
    representing the probability to accept worst results"""

    new_temp = control * temp
    return new_temp


def find_best_result(results):
    """Gets the list containing the results evaluated and returns
    the element with the smallest cost"""
    smallest = results[0]["cost"]
    generation = 0

    for idx, elem in enumerate(results):
        if elem["cost"] < smallest:
            smallest = elem["cost"]
            generation = idx

    return results[generation]


def simulated_annealing(tsp_graph, graph_size, max_iterations):
    results = []

    # inicializa o cost como o cost total de uma rota aleatória
    cities = [i for i in range(0, graph_size - 1)]
    route = cities.copy()
    rd.shuffle(route)
    cost = evaluate_cost(tsp_graph, route)
    results.append({"generation": 0, "route": route, "cost": cost})

    n_iter = 1
    temp = 1000
    control = 0.89
    while n_iter <= max_iterations:

        # trocar as posições de dois vértices do grafo aleatoriamente e calcular o cost
        new_route = swap(route)
        new_cost = evaluate_cost(tsp_graph, new_route)

        delta_cost = cost - new_cost

        # se a nova rota se mostrou uma solução best, aceitar o resultado
        if delta_cost > 0:
            route = new_route
            cost = new_cost

        # se não é best nem pior, continua na posição que está
        elif delta_cost == 0:
            pass

        # ver a probabilidade de aceitar o resultado mesmo sendo pior
        elif rd.uniform(0, 1) <= exp(float(delta_cost) / float(temp)):
            route = new_route
            cost = new_cost

        # registrar os resultados encontrados nesta iteração
        results.append({"generation": n_iter, "route": route, "cost": cost})
        n_iter += 1
        temp = schedule(control, temp)

    return results


def exec_simulated_annealing(tsp, graph_size, max_iterations):

    results = simulated_annealing(tsp, graph_size, max_iterations)
    best = find_best_result(results)

    print(f"\nBest solution found: {best}")

    plot_results(results, graph_size, "Simulated Annealing")


def fitness(generation, n_cities, repeat_w, pop_size):
    print()
    value_sum = 0
    for i in range(pop_size):
        n_repeat = generation[i][n_cities + 1]
        value = 1 / (generation[i][n_cities] * n_repeat * repeat_w)
        generation[i].append(value)
        value_sum += value
    for i in range(pop_size):
        generation[i][n_cities + 2] /= value_sum

    generation = sorted(generation, key=lambda x: x[n_cities + 2], reverse=True)
    return generation


def calc_cost(ag_graph, route, n_cities):
    total = ag_graph[route[0]][route[n_cities - 1]]
    for i in range(n_cities - 1):
        total += ag_graph[route[i]][route[i + 1]]
    return total


def init_gen(ag_graph, n_cities, pop_size):
    gen_one = []

    for i in range(pop_size):
        route = []
        for j in range(n_cities):
            route.append(rd.randint(0, n_cities - 1))
        route.append(calc_cost(ag_graph, route, n_cities))
        route.append(calc_repeat(route, n_cities))
        gen_one.append(route)
    return gen_one


def pick(parents, n_cities, pop_size):
    total = 0
    sorteado = rd.random()
    for i in range(pop_size):
        total += parents[i][n_cities + 2]  # posição do coeficiente
        if sorteado <= total:
            return i
    return 19


def reprod(ag_graph, parents, n_cities, pop_size):
    filhos = []
    corte = int(n_cities / 2)
    # Etapa de seleção
    for i in range(pop_size):
        children = []
        a = pick(parents, n_cities, pop_size)
        b = pick(parents, n_cities, pop_size)
        while b == a:  # não reproduz o mesmo indivíduo
            b = pick(parents, n_cities, pop_size)
        par_a = parents[a]
        par_b = parents[b]
        # Etapa de crossover
        for j in range(corte):
            children.append(par_a[j])
        for j in range(corte, n_cities):
            children.append(par_b[j])
        # Etapa de mutação
        if rd.random() < mut_ratio:
            local = rd.randint(1, n_cities - 1)
            mut_value = rd.randint(1, n_cities - 1)
            children[local] = mut_value
        children.append(calc_cost(ag_graph, children, n_cities))
        children.append(calc_repeat(children, n_cities))
        filhos.append(children)
    return filhos


def calc_repeat(route, n_cities):
    histogram = [0 for _ in range(n_cities)]
    count = 1
    for i in range(n_cities):
        vert = route[i]
        histogram[vert] += 1
        if histogram[vert] > 1:
            count += 1
    return count


def ag_search(ag_graph, n_cities, max_iterations, repeat_w, pop_size):
    generation = init_gen(ag_graph, n_cities, pop_size)
    generation = fitness(generation, n_cities, repeat_w, pop_size)
    best = [i for i in range(n_cities)]
    best_cost = calc_cost(ag_graph, best, n_cities)
    best.append(best_cost)
    repeat_index = n_cities + 1
    cost_index = n_cities
    results = []
    results_gen = []

    for i in range(max_iterations):
        generation = reprod(ag_graph, generation, n_cities, pop_size)
        generation = fitness(generation, n_cities, repeat_w, pop_size)
        # Se for ciclo simples e melhor que o conhecido
        if generation[0][repeat_index] == 1 and generation[0][cost_index] < best_cost:
            best = generation[0]
            best_cost = generation[0][cost_index]
        results.append({"generation": i, "route": best[:n_cities], "cost": best_cost})
        results_gen.append(
            {
                "generation": i,
                "route": generation[0][:n_cities],
                "cost": generation[0][cost_index],
            }
        )
    return results, results_gen


def exec_local_ag_search(ag_graph, n_cities, max_iterations, pop_size):
    repeat_w = n_cities  # multiplicador para a quantidade de repetições
    report = []
    report_gen = []
    report, report_gen = ag_search(
        ag_graph, n_cities, max_iterations, repeat_w, pop_size
    )
    best = find_best_result(report)
    print(f"\nBest solution found: {best}")
    plot_results(report, n_cities, "Local AG Search (lower known cost)")
    plot_results(
        report_gen, n_cities, "Local AG Search (lower cost of current generatio)"
    )#melhor avaliado de cada geração, não distingue inválidos
    print("Último gráfico inclui caminhos inválidos")


def main():
    n_cities = 10
    max_iterations = 100
    pop_size = 100 #40 para o relatório

    ag_graph = g_create(n_cities)
    print("Graph for simulated annealing is:")
    g_print(ag_graph, n_cities)

    sa_graph = copy.deepcopy(ag_graph)
    for i in range(n_cities):
        sa_graph[i][i] = 1 + max_cost

    print("\nGraph for local ag search is:")
    g_print(sa_graph, n_cities)

    exec_simulated_annealing(sa_graph, n_cities, max_iterations)
    exec_local_ag_search(ag_graph, n_cities, max_iterations, pop_size)


if __name__ == "__main__":
    main()

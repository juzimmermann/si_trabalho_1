import random as rd
from math import exp
import copy

from plot_results import plot_results

mut_ratio = 0.1 # de 0 a 1 - taxa de ocorrência de mutação 
limit_gen = 100 # quantidade limite de gerações produzidas. Condição de parada
amostras = 1 # quantidade de chamadas da busca AG
n_vert = 10 # quantidade de vertices do grafo
peso_repetição = n_vert #multiplicador para a quantidade de repetições
min_custo = 1
max_custo = 100 # min_custo sempre < que max_custo. É o intervalo de custos de arestas
pop_size = 40 # tamanho da população

def g_create(qtd_v):
    new_g = []
    for i in range(qtd_v):
        row = []
        for j in range(qtd_v):
            row.append(0)
        new_g.append(row)
    #inicializando
    for i in range(qtd_v):
        new_g[i][i]=0
        for j in range (i+1, qtd_v):
            new_g[i][j]=rd.randint(1, 100)
            new_g[j][i]=new_g[i][j]
    return new_g

def g_print(tsp_g, qtd_v):
    for i in range(qtd_v):
        print()
        for j in range(qtd_v):
            print("{:<4}".format(tsp_g[i][j]), end=" ")

def create_graphs(n_cities):
    
    tsp_graph = g_create(n_cities)
    print('Graph for simulated annealing is:')
    g_print(tsp_graph, n_cities)

    tsp_copy = copy.deepcopy(tsp_graph)
    for i in range(n_cities):
        tsp_copy[i][i] = 101

    print('Graph for local ag search is:')
    g_print(tsp_copy, n_cities)

    return tsp_graph, tsp_copy


def swap(route):
    """Swaps two randon vertices in the route"""
    pos1 = rd.randint(0, len(route)-1)
    pos2 = rd.choice([j for j in range(0, len(route)) if j != pos1])

    route[pos1], route[pos2] = route[pos2], route[pos1]

    altered_route = route.copy()
    return altered_route

def evaluate_cost(tsp_graph, route):
    """Calculates the total cost for the provided route"""

    cost = 0
    for idx, elem in enumerate(route):
        if idx+1 < len(route):
            cost += tsp_graph[elem][route[idx+1]]

    cost += tsp_graph[route[len(route)-1]][route[0]]
    return cost

def schedule(control, temp):
    """Implementation for the function that decrements 'temperature',
    representing the probability to accept worst results"""

    new_temp = control * temp
    return new_temp

def find_best_result(results):
    """Gets the list containing the results evaluated and returns 
    the element with the smallest cost"""
    smallest = results[0]['cost']
    generation = 0

    for idx, elem in enumerate(results):
        if elem['cost'] < smallest:
            smallest = elem['cost']
            generation = idx

    return results[generation]


def simulated_annealing(tsp_graph, graph_size, max_iterations):
    results = []

    #inicializa o custo como o custo total de uma rota aleatória
    cities = [i for i in range(0, graph_size-1)]
    route = cities.copy()
    rd.shuffle(route)
    cost = evaluate_cost(tsp_graph, route)
    results.append({'generation': 0, 'route': route, 'cost': cost})

    n_iter = 1
    temp = 1000
    control = 0.89
    while n_iter <= max_iterations:

        #trocar as posições de dois vértices do grafo aleatoriamente e calcular o custo
        new_route = swap(route)
        new_cost = evaluate_cost(tsp_graph, new_route)

        delta_cost = cost - new_cost

        #se a nova rota se mostrou uma solução melhor, aceitar o resultado
        if delta_cost > 0:
            route = new_route
            cost = new_cost
        
        #se não é melhor nem pior, continua na posição que está
        elif delta_cost == 0:
            pass

        #ver a probabilidade de aceitar o resultado mesmo sendo pior 
        elif rd.uniform(0, 1) <= exp(float(delta_cost)/float(temp)):
            route = new_route
            cost = new_cost

        #registrar os resultados encontrados nesta iteração
        results.append({'generation': n_iter, 'route': route, 'cost': cost})
        n_iter += 1
        temp = schedule(control, temp)

    return results

def exec_simulated_annealing(tsp, graph_size, max_iterations):

    results = simulated_annealing(tsp, graph_size, max_iterations)
    best = find_best_result(results)

    print(f'\nBest solution found: {best}')

    plot_results(results, graph_size, 'Simulated Annealing')


def fitness(generation):
    value_sum = 0
    for i in range(pop_size):
        value= 1 /  (generation[i][n_vert] * generation[i][n_vert + 1] * peso_repetição)
        generation[i].append(value)
        value_sum += value
        
    for i in range(pop_size):
        generation[i][n_vert + 2] /= value_sum
    generation = sorted(generation, key=lambda x: x[n_vert + 2], reverse=True)
    return generation    

def calc_cost(tsp_g, v_path):
    total=tsp_g[v_path[0]][v_path[n_vert - 1]]
    for i in range(n_vert - 1):
        total+=tsp_g[v_path[i]][v_path[i + 1]]
    return total   

def print_gen(gen_one):
    for i in range(pop_size):
        print()
        print(gen_one[i])

def init_gen(ag_graph):
    gen_one = [] 
    
    for i in range(pop_size):    
        dude = []
        for j in range(n_vert):
            dude.append(rd.randint(0, n_vert - 1))
        dude.append(calc_cost(ag_graph, dude))
        dude.append(calc_repet(dude))
        gen_one.append(dude)
    return gen_one    

def pick(pais):
    total=0
    sorteado = rd.random()
    for i in range(pop_size):
        total += pais[i][n_vert + 2]
        if sorteado <= total:
            return i
    return 19    

def crossover(ag_graph, pais):
    
    filhos = []
    corte = int(n_vert/2)
    for i in range(pop_size):
        baby = []
        a = pick(pais)
        b = pick(pais)
        while (b == a): #não reproduz o mesmo indivíduo
            b = pick(pais)
        par_a = pais[a]
        par_b = pais[b]
        for j in range(corte):
            baby.append(par_a[j])
        for j in range(corte, n_vert):
            baby.append(par_b[j])
        if rd.random() < mut_ratio: #gera mutação
            local = rd.randint(1, n_vert-1)
            gene = rd.randint(1, n_vert-1)
            baby[local]=gene
        baby.append(calc_cost(ag_graph, baby))
        baby.append(calc_repet(baby))
        filhos.append(baby)
    return filhos

def calc_repet(caminho):
    histograma = [0 for _ in range(n_vert)]
    cont = 1
    for i in range(n_vert):
        vert = caminho[i]
        histograma[vert] += 1
        if histograma[vert] > 1:
            cont += 1
    return cont


def ag_search(ag_graph):
    generation = init_gen(ag_graph)
    generation = fitness(generation)
    melhor = generation[0]
    melhor_custo = n_vert * (max_custo + 1)
    melhor_gen = 0
    repet_cont = n_vert + 1 
    cost_count = n_vert
    aprimoramentos = 0
    results = []
    

    for i in range(limit_gen):
        generation = crossover(ag_graph,generation) #mutação dentro desta função
        generation = fitness(generation)
        if generation[0][repet_cont] == 1 and generation[0][cost_count] < melhor_custo:#se é caminho válido e melhor
            melhor = generation[0]
            melhor_gen = 1 + i
            aprimoramentos += 1
            melhor_custo = generation[0][cost_count]
        results.append({'generation': i, 'route': melhor[:n_vert], 'cost': melhor_custo})    

    melhor.append(aprimoramentos)
    melhor.append(melhor_gen)
    return results

def ag_search_gen(ag_graph):
    generation = init_gen(ag_graph)
    generation = fitness(generation)
    melhor = generation[0]
    melhor_custo = n_vert * (max_custo + 1)
    melhor_gen = 0
    repet_cont = n_vert + 1 
    cost_count = n_vert
    aprimoramentos = 0
    results = []
    

    for i in range(limit_gen):
        generation = crossover(ag_graph,generation) #mutação dentro desta função
        generation = fitness(generation)
        if generation[0][repet_cont] == 1 and generation[0][cost_count] < melhor_custo:#se é caminho válido e melhor
            melhor = generation[0]
            melhor_gen = 1 + i
            aprimoramentos += 1
            melhor_custo = generation[0][cost_count]
        results.append({'generation': i, 'route': generation[0][:n_vert], 'cost': generation[0][cost_count]})    

    melhor.append(aprimoramentos)
    melhor.append(melhor_gen)
    return results

def exec_local_ag_search(ag_graph, n_vert):

    report = [] #relatório/report é array das respostas de cada chamada pra ag_search()
    report_gen = []
    menor = n_vert * (max_custo + 1) #custo do melhor caminho valido encontrado entre as chamadas. inicia com valor alto impossível  
    maior = 0 #custo do maior caminho valido encontrado
    validos = 0 #conta quantos caminhos não tem repetição - caminhos válidos
    repet_cont = n_vert + 1 #indice onde está armazenado a quantidade de repetições
    cost_count = n_vert #índice onde está armazenado o custo do caminho
    #aprimoramentos = n_vert + 3 #quantas vezes um indivíduo melhor avaliado apareceu
    #gen_cont = n_vert + 4 #índice da posição que contém a última geração com aprimoramento
    acumulado = 0
    report = ag_search(ag_graph)
    report_gen = ag_search_gen(ag_graph)
    
    #print(report[0])
    #print("\nVálidos:", validos, "Menor custo:", menor, "Maior custo:", maior, "Custo médio:", (acumulado/validos) )
    plot_results(report, n_vert, 'Local AG Search (lower known cost)')
    plot_results(report_gen, n_vert, 'Local AG Search (lower cost of current generatio)')


def main():
    n_cities = 10
    max_iterations = 100

    sa_graph, ag_graph = create_graphs(n_cities)

    exec_simulated_annealing(sa_graph, n_cities, max_iterations)

    exec_local_ag_search(ag_graph, n_cities)

if __name__ == '__main__':
    main()
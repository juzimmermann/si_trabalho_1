import matplotlib.pyplot as plt

def find_best_result(results):
    """Gets the list containing the results evaluated and returns 
    the element with the smallest cost"""
    smallest = results[0]['cost']
    generation = 0

    print("\nAll results found:")

    for idx, elem in enumerate(results):
        print(elem)
        if elem['cost'] < smallest:
            smallest = elem['cost']
            generation = idx

    return results[generation]

def plot_results(all_results, graphsize, algorithm):
    smallest = find_best_result(all_results)

    generations = [g['generation'] for g in all_results]
    costs = [c['cost'] for c in all_results]

    plt.plot(generations, costs)
    plt.scatter(smallest['generation'], smallest['cost'], color='red')

    plt.title(f'{algorithm} - {graphsize} cities')
    plt.xlabel('Number of generations')
    plt.ylabel('Cost of route')
    plt.legend(["Cost history", "Smallest cost"], loc="upper right")

    plt.show()

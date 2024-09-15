import random
from deap import base, creator, tools

def calculate_metrics(individual):
    # Example: Split orders into picking stations and calculate balance & pod visits
    
    num_picking_stations = 5  # Assume we have 5 picking stations
    pod_visit_penalty = 10    # Set an arbitrary penalty for each pod visit
    order_distribution = [[] for _ in range(num_picking_stations)]
    
    # Distribute orders across stations (this is just a simple example)
    for i, order in enumerate(individual):
        station_index = i % num_picking_stations
        order_distribution[station_index].append(order)
    
    # Calculate imbalance (difference in number of orders across stations)
    order_counts = [len(orders) for orders in order_distribution]
    max_orders = max(order_counts)
    min_orders = min(order_counts)
    balance = max_orders - min_orders  # We want this to be minimized

    # Calculate pod visits (more orders typically mean more pod visits)
    pod_visits = sum(order_counts) * pod_visit_penalty
    
    # Return the total metric score
    return balance, pod_visits


# Define the problem
def evaluate(individual):
    # Fitness function: minimize imbalance and pod visits
    balance, pod_visits = calculate_metrics(individual)
    return balance + pod_visits,  # Note the comma (DEAP expects a tuple)

# Create the individual (order sequence)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, 9)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Main genetic algorithm
def genetic_algorithm():
    population = toolbox.population(n=100)
    for gen in range(50):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

    return tools.selBest(population, 1)[0]

best_solution = genetic_algorithm()
print("Best solution:", best_solution)

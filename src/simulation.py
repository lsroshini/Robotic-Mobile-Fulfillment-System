import random
from deap import base, creator, tools
import numpy as np

# Genetic Algorithm setup for Order Allocation and Sequencing
def calculate_metrics(individual):
    num_picking_stations = 5
    pod_visit_penalty = 10
    order_distribution = [[] for _ in range(num_picking_stations)]

    # Distribute orders across picking stations
    for i, order in enumerate(individual):
        station_index = i % num_picking_stations
        order_distribution[station_index].append(order)
    
    # Calculate balance
    order_counts = [len(orders) for orders in order_distribution]
    balance = max(order_counts) - min(order_counts)

    # Calculate pod visits
    pod_visits = sum(order_counts) * pod_visit_penalty
    return balance + pod_visits,  # DEAP expects a tuple

# Define the problem using DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, 9)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", calculate_metrics)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Genetic Algorithm for order allocation and sequencing
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

# Ant Colony Optimization for shelf selection and robot scheduling
class AntColony:
    def __init__(self, graph, num_ants, num_iterations, decay, alpha=1, beta=2):
        self.graph = graph
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.pheromone = np.ones(self.graph.shape) / len(graph)

    def run(self):
        shortest_path = None
        all_time_best = ("placeholder", float('inf'))
        for _ in range(self.num_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_best[1]:
                all_time_best = shortest_path
            self.pheromone * self.decay  # Decay pheromone
        return all_time_best

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        total_dist = 0
        for _ in range(len(self.graph) - 1):
            move = self.pick_move(self.pheromone[prev], self.graph[prev], visited)
            path.append((prev, move))
            total_dist += self.graph[prev][move]
            prev = move
            visited.add(move)
        return path, total_dist

    def gen_all_paths(self):
        all_paths = []
        for ant in range(self.num_ants):
            path = self.gen_path(random.randint(0, len(self.graph) - 1))
            all_paths.append(path)
        return all_paths

    def spread_pheromone(self, all_paths):
        for path, dist in all_paths:
            for move in path:
                self.pheromone[move] += 1.0 / self.graph[move]

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)
        norm_row = row / row.sum()
        move = np.random.choice(len(pheromone), 1, p=norm_row)[0]
        return move

def create_distance_matrix(num_shelves):
    return np.random.randint(1, 20, size=(num_shelves, num_shelves))

# Combine Genetic Algorithm and Ant Colony Optimization
def run_simulation():
    print("Running Genetic Algorithm for order allocation...")
    best_allocation = genetic_algorithm()
    print("Best order allocation:", best_allocation)

    print("\nRunning Ant Colony Optimization for shelf selection and robot scheduling...")
    num_shelves = 10
    graph = create_distance_matrix(num_shelves)
    ant_colony = AntColony(graph, num_ants=10, num_iterations=100, decay=0.95)
    shortest_path = ant_colony.run()

    print("Best path found by ACO (shelf selection and robot scheduling):", shortest_path)

# Run the entire simulation
run_simulation()

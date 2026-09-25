import networkx as nx
import matplotlib.pyplot as plt
from numpy import random
import random as rnd
import heapq
import matplotlib.animation as animation
import numpy as np

random.seed(113)
rnd.seed(113)

#These should be changed later
sigma = 1 #Rate to get Infected from being exposed
gamma = 1 #Rate to get recovered from being infectious
tau = 1 #Rate to get exposed from being suspectible
beta = 1 #Rate to get suspectible from being recovered
delta = 0.1 #Rate to become dead from being infected


def create_network(type, node_nr, mean_nr_of_connections, seed):
    match type:
        case 'E':
            graph_erdos_renyi = nx.erdos_renyi_graph(node_nr, mean_nr_of_connections/(node_nr - 1), seed, directed = False)
            return graph_erdos_renyi
        case 'B':
            #Every new node connects to some number (I denote it as x) of existing nodes
            #edge_nr = node_nr * x
            #<k> is the mean degree of nodes
            #<k>=2*edge_nr/node_nr
            #<k>=2*node_nr * x/node_nr
            #<k>=2*x
            #x=<k>/2
            graph_barabasi_albert = nx.barabasi_albert_graph(node_nr, mean_nr_of_connections // 2, seed)
            return graph_barabasi_albert
        case 'C':
            random_draws = random.poisson(mean_nr_of_connections, node_nr)
            #Now I have random draws for the nr of outgoing edges for each node
            #But the nr of outgoing edges might be uneven
            #There might be edges from a node itself
            #There might be several edges between multiple nodes
            if sum(random_draws)%2 != 0:
                random_nr = rnd.randrange(0, len(random_draws))
                random_draws[random_nr] = random_draws[random_nr]+1
            initial_graph = nx.configuration_model(random_draws, create_using=None, seed=seed)
            graph_configuration = nx.Graph(initial_graph)
            graph_configuration.remove_edges_from(nx.selfloop_edges(graph_configuration))
            return graph_configuration


def initial_states(network, initially_infected_nr):
    nx.set_node_attributes(network, 'S', "State")
    randomly_infected = rnd.sample(list(network.nodes), initially_infected_nr)
    for infected_person in randomly_infected:
        network.nodes[infected_person]["State"] = 'I'
    return network





def plotting(graph):
    colors = []
    for node in graph.nodes():
        if graph.nodes[node]["State"] == "I":
            #Red
            colors.append((1,0,0))
        elif graph.nodes[node]["State"] == "S":
            #Green
            colors.append((0,1,0))
        elif graph.nodes[node]["State"] == "E":
            #White
            colors.append((1,1,1))
        elif graph.nodes[node]["State"] == "R":
            #Blue
            colors.append((0,0,1))
        elif graph.nodes[node]["State"] == "D":
            #Black
            colors.append((0,0,0))

    plt.figure(figsize=(10, 10))
    nx.draw(graph, node_color = colors)
    plt.show()


def simulate_epidemic():
    for type in ['E', 'B', 'C']:
        network = initial_states(create_network(type, 100, 7, 113), 2)
        heap = []
        heapq.heapify(heap)
        #First, put on the heap the first events that come from the first infected nodes
        current_time = 0
        for n in network.nodes():
            if network.nodes[n]["State"] == "I":
                time_infectious_recovered = current_time + random.exponential(1 / gamma)
                time_infectious_dead = current_time + random.exponential(1 / delta)
                if time_infectious_recovered < time_infectious_dead:
                    heapq.heappush(heap, (time_infectious_recovered, n, 'R'))
                    for neighbor in network.neighbors(n):
                        if network.nodes[neighbor]["State"] == "S":
                            time_suspectible_exposed = current_time + random.exponential(1 / tau)
                            if time_suspectible_exposed < time_infectious_recovered:
                                heapq.heappush(heap, (time_suspectible_exposed, neighbor, 'E'))
                else:
                    network.nodes[n]["State"] = "D"
                    for neighbor in network.neighbors(n):
                        if network.nodes[neighbor]["State"] == "S":
                            time_suspectible_exposed = current_time + random.exponential(1 / tau)
                            if time_suspectible_exposed < time_infectious_dead:
                                heapq.heappush(heap, (time_suspectible_exposed, neighbor, 'E'))
        while len(heap) > 0:
            (time, node, event) = heapq.heappop(heap)
            current_time = time
            match event:
                case 'I':
                    if network.nodes[node]["State"] == "E":
                        network.nodes[node]["State"] = "I"
                        time_infectious_recovered = current_time + random.exponential(1 / gamma)
                        time_infectious_dead = current_time + random.exponential(1 / delta)
                        if time_infectious_recovered < time_infectious_dead:
                            heapq.heappush(heap, (time_infectious_recovered, node, 'R'))
                            for neighbor in network.neighbors(node):
                                if network.nodes[neighbor]["State"] == "S":
                                    time_suspectible_exposed = current_time + random.exponential(1 / tau)
                                    if time_suspectible_exposed < time_infectious_recovered:
                                        heapq.heappush(heap, (time_suspectible_exposed, neighbor, 'E'))
                        else:
                            network.nodes[node]["State"] = "D"
                            for neighbor in network.neighbors(node):
                                if network.nodes[neighbor]["State"] == "S":
                                    time_suspectible_exposed = current_time + random.exponential(1 / tau)
                                    if time_suspectible_exposed < time_infectious_dead:
                                        heapq.heappush(heap, (time_suspectible_exposed, neighbor, 'E'))
                case 'S':
                    if network.nodes[node]["State"] == "R":
                        network.nodes[node]["State"] = "S"
                case 'R':
                    if network.nodes[node]["State"] == "I":
                        network.nodes[node]["State"] = "R"
                        time_recovered_suspectible = current_time + random.exponential(1 / beta)
                        heapq.heappush(heap, (time_recovered_suspectible, node, 'S'))
                case 'E':
                    if network.nodes[node]["State"] == "S":
                        network.nodes[node]["State"] = "E"
                        time_exposed_infectious = current_time + random.exponential(1 / sigma)
                        heapq.heappush(heap, (time_exposed_infectious, node, 'I'))
        plotting(network)


if __name__ == "__main__":
    simulate_epidemic()





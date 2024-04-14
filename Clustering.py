import numpy as np
import random

class UAV:
    def __init__(self, id):
        self.id = id
        self.neighbours = []
        self.weight = 0

    def add_neighbor(self, neighbor):
        self.neighbours.append(neighbor)

def generateNeighbourMatrix(n,rows,cols):
    adjacencyIdMatrix = np.zeros((rows, cols), dtype=int)
    num = 1
    for i in range(rows):
        for j in range(cols):
            if num <= numberOfUavs:
                adjacencyIdMatrix[i, j] = num
                num += 1
    flattened_matrix = adjacencyIdMatrix.flatten()
    np.random.shuffle(flattened_matrix)
    adjacencyIdMatrixShuffled = flattened_matrix.reshape(adjacencyIdMatrix.shape)

    return adjacencyIdMatrixShuffled


def khop_clustering(nodes, k):
    # Step 1: Assign random values from [0,k] to each node
    for node in nodes:
        node.weight = random.randint(1, k)
        print("Node:", node.id, "Weight:", node.weight, "Neighbours:", [neighbour.id for neighbour in node.neighbours])
    # Steps 2-6: Iterate until convergence
    converged = False
    counter = 1
    while not converged:
        print("counter", counter)
        converged = True
        for node in nodes:
            min_neighbor_weight = min(node.neighbours, key=lambda x: x.weight).weight
            print("minimum weight:", node.id, min_neighbor_weight)

            # Step 3: Set weight to min weight of neighbours
            node.weight = min_neighbor_weight

            # Step 4: Set weight to 0 if all neighbours have weight k
            if node.weight != 0 and all(neighbor.weight == k for neighbor in node.neighbours):
                node.weight = 0
                converged = False
                continue

            # Step 5: Increment weight if necessary
            elif node.weight != 0 and node.weight < min_neighbor_weight:
                node.weight += 1
                converged = False
                continue

            # Step 6: Increment weight if necessary
            elif node.weight == 0 and any(neighbor.weight == 0 for neighbor in node.neighbours):
                node.weight += 1
                converged = False
                continue

        counter += 1
    return nodes

# Example usage
if __name__ == "__main__":
    numberOfUavs = 100
    rows = int(np.ceil(np.sqrt(numberOfUavs)))
    cols = int(np.ceil(numberOfUavs / rows))

    # Create nodes  
    nodes = [UAV(i+1) for i in range(numberOfUavs)]
    # print(nodes)

    # Connect nodes randomly (for demonstration purposes)
    # for node in nodes:
    #     num_neighbours = random.randint(1, 5)
    #     for _ in range(num_neighbours):
    #         neighbor = random.choice(nodes)
    #         if neighbor != node and neighbor not in node.neighbours:
    #             node.add_neighbor(neighbor)
    #             neighbor.add_neighbor(node)
    
    # Generate Neighbour's adjacency Matrix
    adjacencyMatrix = generateNeighbourMatrix(numberOfUavs,rows,cols)
    print("Final adjacency matrix: ")
    print(adjacencyMatrix)
    
    for node in nodes:
        # print(" node", node)
        for i in range(rows):
            for j in range(cols):
                if node.id == adjacencyMatrix[i][j]:
                    if i > 0:  # connect with the UAV above
                        node.add_neighbor(nodes[adjacencyMatrix[i-1][j]-1])
                    if i < rows - 1:  # connect with the UAV below
                        node.add_neighbor(nodes[adjacencyMatrix[i + 1][j] -1])
                    if j > 0:  # connect with the UAV on the left
                        node.add_neighbor(nodes[adjacencyMatrix[i][j - 1]-1])
                    if j < cols - 1:  # connect with the UAV on the right
                        node.add_neighbor(nodes[adjacencyMatrix[i][j + 1]-1])
    # Set k
    k = 3

    # Run Khop clustering algorithm
    clustered_nodes = khop_clustering(nodes, k)

    # Print results
    print("After clustering")
    for node in clustered_nodes:
        print("Node:", node.id, "Weight:", node.weight, "Neighbours:", [neighbour.id for neighbour in node.neighbours])

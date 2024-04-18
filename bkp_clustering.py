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
    # np.random.shuffle(flattened_matrix)
    adjacencyIdMatrixShuffled = flattened_matrix.reshape(adjacencyIdMatrix.shape)

    return adjacencyIdMatrixShuffled

def countClusterheads(nodes):
    count = 0
    for node in nodes:
        if node.weight == 0:
            count += 1
    
    return count


def khop_clustering(nodes, k, numberOfUavs):
    # Step 1: Assign random values from [0,k] to each node
    for node in nodes:
        node.weight = random.randint(0, k)
        print("Node:", node.id, "Weight:", node.weight, "Neighbours:", [neighbour.id for neighbour in node.neighbours])
    # Steps 2-6: Iterate until convergence
    converged = False
    counter = 1
    while not converged:
        print("counter", counter)
        for node in nodes:
            print("Node:", node.id, "Weight:", node.weight, "Neighbours:", [neighbour.id for neighbour in node.neighbours])

        converged = True
        for node in nodes:
            print("beginning of for loop")
            print("Node:", node.id, "Weight:", node.weight, "Neighbours:", [neighbour.id for neighbour in node.neighbours])
            min_neighbor_weight = min(node.neighbours, key=lambda x: x.weight).weight
            # print("minimum weight:", node.id, min_neighbor_weight)
            
                
            # Step 4: Set weight to 0 if all neighbours have weight k
            if node.weight != 0 and all(neighbor.weight == k for neighbor in node.neighbours):
                node.weight = 0
                converged = False
                print("end of for loop 1")
                print("Node:", node.id, "Weight:", node.weight, "Neighbours:", [neighbour.id for neighbour in node.neighbours])
                continue
                

            # Step 5: Increment weight if necessary
            elif node.weight != 0 and node.weight < min_neighbor_weight:
                node.weight += 1
                converged = False
                print("end of for loop 2")
                print("Node:", node.id, "Weight:", node.weight, "Neighbours:", [neighbour.id for neighbour in node.neighbours])
                continue
                

            # Step 6: Increment weight if necessary
            elif node.weight == 0 and any(neighbor.weight == 0 for neighbor in node.neighbours):
                node.weight += 1
                converged = False
                print("end of for loop 3")
                print("Node:", node.id, "Weight:", node.weight, "Neighbours:", [neighbour.id for neighbour in node.neighbours])
                continue
            
            # Step 3: Set weight to min weight of neighbours
            elif (min_neighbor_weight < node.weight):
                node.weight = min_neighbor_weight
                converged = False
                print("end of for loop 4")
                print("Node:", node.id, "Weight:", node.weight, "Neighbours:", [neighbour.id for neighbour in node.neighbours])
                continue

            
                
        
        clusterheadCount = countClusterheads(nodes)
        if clusterheadCount >= np.ceil(np.sqrt(numberOfUavs)):
            for node in reversed(nodes):
                if node.weight == 0 and any(neighbor.weight == 0 for neighbor in node.neighbours):
                    node.weight += 1
                    continue
            clusterheadCount = countClusterheads(nodes)
            if clusterheadCount < np.ceil(np.sqrt(numberOfUavs)):
                converged = False
            else:
                converged = True
                break
            

        counter += 1
    return nodes

# Example usage
if __name__ == "__main__":
    # Set k
    k = 3
    numberOfUavs = 9
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
                    if (i > 0) and (adjacencyMatrix[i-1][j] != 0):  # connect with the UAV above
                        node.add_neighbor(nodes[adjacencyMatrix[i-1][j]-1])
                    if (i < rows - 1) and (adjacencyMatrix[i + 1][j] != 0 ):  # connect with the UAV below
                        node.add_neighbor(nodes[adjacencyMatrix[i + 1][j] -1])
                    if (j > 0) and (adjacencyMatrix[i][j - 1] != 0):  # connect with the UAV on the left
                        node.add_neighbor(nodes[adjacencyMatrix[i][j - 1]-1])
                    if (j < cols - 1) and (adjacencyMatrix[i][j + 1] != 0):  # connect with the UAV on the right
                        node.add_neighbor(nodes[adjacencyMatrix[i][j + 1]-1])
    

    # Run Khop clustering algorithm
    clustered_nodes = khop_clustering(nodes, k, numberOfUavs)

    # Print results
    print("After clustering")
    for node in clustered_nodes:
        print("Node:", node.id, "Weight:", node.weight, "Neighbours:", [neighbour.id for neighbour in node.neighbours])

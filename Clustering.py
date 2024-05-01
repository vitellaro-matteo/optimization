import random
import math
import matplotlib.pyplot as plt
import copy

class UAV:
    def __init__(self, id, weight):
        self.id = id
        self.weight = weight
        self.position = self.generate_random_point()
        self.neighbors = []

    def generate_random_point(self):
        """Generate a random point in 2D space."""
        return (random.uniform(0, 10), random.uniform(0, 10))

    def distance(self, other_uav):
        """Calculate the Euclidean distance between this UAV and another UAV."""
        return math.sqrt((self.position[0] - other_uav.position[0])**2 + (self.position[1] - other_uav.position[1])**2)

    def communicate_within_radius(self, other_uav, radius):
        """Check if this UAV and another UAV are within communication radius."""
        if self.distance(other_uav) <= radius:
            return True
        else:
            return False

    def update_neighbors(self, all_uavs, communication_radius):
        """Update neighbors for this UAV based on communication radius."""
        self.neighbors = [uav for uav in all_uavs if self.communicate_within_radius(uav, communication_radius)]

    def delete_self_neighbors(self):
        self.neighbors = [neighbor for neighbor in self.neighbors if neighbor.id != self.id]

def printUavs(uavs):
    for uav in uavs:
        neighbor_ids = [neighbor.id for neighbor in uav.neighbors]
        print(f"UAV ID: {uav.id}, Weight: {uav.weight}, Neighbors: {neighbor_ids}")

def uavGraph(uavs,index,k):
    plt.figure(figsize=(8, 8))
    for uav in uavs:
        if uav.weight == k:
            plt.plot(uav.position[0], uav.position[1], 'ro', markersize=18)  # Plot UAVs
        else:
            plt.plot(uav.position[0], uav.position[1], 'bo', markersize=15)  # Plot UAVs
        plt.text(uav.position[0], uav.position[1], str(uav.id), fontsize=12, ha='center', va='center', color = 'white')  # Add node ID
        plt.text(uav.position[0], uav.position[1] - 0.4, str(k - uav.weight), fontsize=8, ha='center', va='bottom', color='black')  # Add node weight
        # plt.text(uav.position[0], uav.position[1], f'{uav.weight:.1f}', fontsize=8, ha='center', va='center', color='white')  # Add node weight inside the point
        for neighbor in uav.neighbors:
            if index == 0:
                plt.plot([uav.position[0], neighbor.position[0]], [uav.position[1], neighbor.position[1]], 'k-')  # Plot communication links
            elif index == 1:
                if (uav.weight - neighbor.weight) == 1:
                    plt.plot([uav.position[0], neighbor.position[0]], [uav.position[1], neighbor.position[1]], 'k-')  # Plot communication links
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('UAV Communication Network')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def khop_clustering(uavs, k, num_uavs):
    converged = False
    counter = 1
    while not converged:
        prevUavConfig = uavs
        print("counter", counter)
        printUavs(uavs)
        print("======================================================================================================================")
        converged = True
        for uav in uavs:

            if uav.neighbors:
                max_neighbor_weight = max(neighbor.weight for neighbor in uav.neighbors)
            else:
                max_neighbor_weight = 0

            if (max_neighbor_weight > uav.weight) and (uav.weight != (max_neighbor_weight - 1)):
                print("first if")
                print(uav.id)
                uav.weight = max_neighbor_weight - 1
                converged = False
                continue

            if (max_neighbor_weight == 0) and (uav.weight == 0):
                print("second if")
                uav.weight = k
                converged = False
                continue

            if (max_neighbor_weight <= uav.weight) and uav.weight != k:
                print("3 if")
                uav.weight -= 1
                converged = False
                continue
            
            if (max_neighbor_weight == k) and (uav.weight == k):
                print("4 if")
                uav.weight -= 1
                converged = False
                continue

        counter+=1

    return uavs


def cleanClusterInformation(finalClusters):
    modifiedClusters = {}
    for key, value in finalClusters.items():
        modified_list = []
        for sublist in value:
            modified_sublist = [node for node in sublist if node != key]
            modified_list.extend(modified_sublist)
        modifiedClusters[key] = modified_list
    return modifiedClusters

def getClusterInformation(uavs,k):
    clusters = {}
    uavClusterHeads = [uav for uav in uavs if uav.weight == k]
    print("Cluster Heads:")
    printUavs(uavClusterHeads)
    for head in uavClusterHeads:
        visited = set()
        visited_nodes = []
        clusterTraversal(head, k, visited,visited_nodes)

        # Initialize the list if it doesn't exist yet
        if head.id not in clusters:
            clusters[head.id] = []

        clusters[head.id].append(visited_nodes)

    finalClusters = cleanClusterInformation(clusters)
    return finalClusters

def clusterTraversal(node, k, visited, visited_nodes):
    visited.add(node.id)
    visited_nodes.append(node.id)
    for neighbor in node.neighbors:
        if neighbor.weight == node.weight - 1 and neighbor.id not in visited:
            clusterTraversal(neighbor,k, visited, visited_nodes)

# Generate a list of UAV objects
def cluster(num_uavs, communication_radius, k):
    uavs = [UAV(id+1, random.randint(0, k)) for id in range(num_uavs)]

    # Update neighbors for each UAV
    for uav in uavs:
        uav.update_neighbors(uavs, communication_radius)
        uav.delete_self_neighbors()

    print("Initial configuration")
    printUavs(uavs)
    print("======================================================================================================================")
    
    uavGraph(uavs,0,k)

    clustered_nodes = khop_clustering(uavs,k,num_uavs)
    printUavs(clustered_nodes)

    finalClusters = getClusterInformation(clustered_nodes, k)
    print("Final Clusters:")
    print(finalClusters)

    uavGraph(clustered_nodes,1,k)

    return clustered_nodes,finalClusters
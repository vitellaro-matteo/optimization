import random
import math
import matplotlib.pyplot as plt

class UAV:
    def __init__(self, id, weight):
        self.id = id
        self.weight = weight
        self.position = self.generate_random_point()
        self.neighbors = []

    def generate_random_point(self):
        """Generate a random point in 2D space."""
        return (random.uniform(-10, 10), random.uniform(0, 10))

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
        print(f"UAV ID: {uav.id}, Neighbors: {neighbor_ids}")

# Generate a list of UAV objects
def main():
    num_uavs = 5
    communication_radius = 6
    k = 3
    uavs = [UAV(id+1, random.randint(0, k)) for id in range(num_uavs)]

    # Update neighbors for each UAV
    for uav in uavs:
        uav.update_neighbors(uavs, communication_radius)
        uav.delete_self_neighbors()

    printUavs(uavs)

    plt.figure(figsize=(8, 8))
    for uav in uavs:
        plt.plot(uav.position[0], uav.position[1], 'bo', markersize=10)  # Plot UAVs
        plt.text(uav.position[0], uav.position[1], str(uav.id), fontsize=12, ha='center', va='center')  # Add node ID
        # plt.text(uav.position[0], uav.position[1], f'{uav.weight:.1f}', fontsize=8, ha='center', va='center', color='white')  # Add node weight inside the point
        for neighbor in uav.neighbors:
            plt.plot([uav.position[0], neighbor.position[0]], [uav.position[1], neighbor.position[1]], 'k--')  # Plot communication links

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('UAV Communication Network')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()
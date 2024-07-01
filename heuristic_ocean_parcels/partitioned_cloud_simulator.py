from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import datetime
import xarray

from simulation import simulation_main
from visual_analysis import plot_all_trajectories


def coordinates_of_cluster_centers(n_clusters, cloud_of_points):
    """
    Computes the coordinates of the center of mass of each partition,
    using the k-means clustering algorithm.
    :param n_clusters: int
    :param cloud_of_points: numpy array of 2d points
    :return: numpy array of the coordinates
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(cloud_of_points)
    centers_coordinates = kmeans.cluster_centers_
    # labels = kmeans.labels_
    return centers_coordinates #, labels


def read_last_coords_of_particles(k):
    file_i = f"file_logging/simulation_output_file/{k}_clusters.nc"

    file_i = xarray.open_dataset(file_i)
    ll_longituds = file_i.lon.values
    ll_latituds = file_i.lat.values

    set_of_last_coords = []
    for longitud, latitud in zip(ll_longituds, ll_latituds):
        set_of_last_coords.append((longitud[-1],latitud[-1]))

    return set_of_last_coords


AMOUNT_OF_PARTICLES = 100
X, Y = 40.6, 2.44
VARIANCE = 0.001
K_RANGE_MIN = 2
K_RANGE_MAX = 3

def main():
    """
    A function to simulate and plot sucessive partitions of an initial cloud of particles
    """

    # Initialise the cloud of particles
    mat = np.random.multivariate_normal([X, Y], np.diag(VARIANCE * np.ones(2)), size=AMOUNT_OF_PARTICLES)

    print("Simulating...")
    for k in range(K_RANGE_MIN,K_RANGE_MAX):
        centers_coordinates = coordinates_of_cluster_centers(n_clusters=k, cloud_of_points=mat)
        print("centers_coordinates=",centers_coordinates)
        simulation_main(
            start_coordinates=centers_coordinates,
            start_datetime=np.datetime64(datetime.datetime(2022, 3, 1, 2, 0, 0)),
            simulation_hours=24*10,
            cloud_of_particles=False)

    print("Plotting...")
    for k in range(K_RANGE_MIN,K_RANGE_MAX):
        fig = plot_all_trajectories(simulation_file_uri=f"file_logging/simulation_output_file/validation.nc",
                                    starting_particle_distribution=mat)
        plt.savefig(f"{k}_clusters_variance.png")
    print("Plotting done")

EPSILON = 0.03*3 # max distance allowed


def iterative_algorithm():
    for k in range(K_RANGE_MIN+1,K_RANGE_MAX): # k = 2, 3, 4, ...
        print(f"\nk={k} vs k_-1={k-1}")

        all_cm_have_a_previous_one_nearby = []
        current_final_positions = read_last_coords_of_particles(k)
        previous_final_positions = read_last_coords_of_particles(k-1)
        for final_position_i in current_final_positions:
            distances_to_previous_points = []
            for old_final_position_i in previous_final_positions:
                print(final_position_i,old_final_position_i)
                distance = np.sqrt((final_position_i[0] - old_final_position_i[0])**2 + (final_position_i[1] - old_final_position_i[1])**2)
                distances_to_previous_points.append(distance)
            print(distances_to_previous_points,"min=",min(distances_to_previous_points))
            if min(distances_to_previous_points) > EPSILON:
                print("Closest cm is too far away")
            else:
                print("The previous partitioning is representative")
                all_cm_have_a_previous_one_nearby.append(1)
        print("sum()=",sum(all_cm_have_a_previous_one_nearby))
        if sum(all_cm_have_a_previous_one_nearby) == k:
            print(f"Finish at k={k}")
            break
    print(f"End. k={k}")


def validate_origins():
    k = 11
    n_points = 10
    local_variance = 0.00001
    points = np.zeros((n_points*k,2))
    current_final_positions = read_last_coords_of_particles(k)
    for i, point_i in enumerate(current_final_positions):
        x,y = point_i
        cloud_i = np.random.multivariate_normal([y, x], np.diag(local_variance * np.ones(2)), size=n_points)
        points[n_points*i:n_points*i+n_points,:] = cloud_i
    simulation_main(
            start_coordinates=points,
            start_datetime=np.datetime64(datetime.datetime(2022, 2, 19, 2, 0, 0)),
            simulation_hours=24*10,
            cloud_of_particles=False)

    fig = plot_all_trajectories(simulation_file_uri=f"file_logging/simulation_output_file/validation.nc",
                                starting_particle_distribution=points)
    plt.show()

if __name__ == "__main__":
    ####### Test coordinates_of_cluster_centers() #######
    # k = 8
    # mat = np.random.multivariate_normal([10,10], np.diag(1 * np.array([1,1])),size=500)
    # # centers_coordinates, labels = coordinates_of_cluster_centers(n_clusters=k, cloud_of_points=mat)
    # centers_coordinates = coordinates_of_cluster_centers(n_clusters=k, cloud_of_points=mat)
    # for i in centers_coordinates:
    #     print(i)
    # plt.figure(figsize=(4,3))
    # # plt.scatter(mat[:,0],mat[:,1],marker=".",alpha=0.5,c=labels,cmap="viridis")
    # plt.scatter(centers_coordinates[:,0],centers_coordinates[:,1],c="r",marker="o")
    # plt.tight_layout()
    # plt.axis("equal")
    # # plt.savefig("round_cloud.pdf")
    # plt.show()

    ####### Run main() #######
    main()
    ####### Run iterative_algorithm() #######
    # iterative_algorithm()
    ####### Run validate_origins() #######
    # validate_origins()
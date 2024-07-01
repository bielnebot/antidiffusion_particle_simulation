import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(module)s | %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from parcels import ParticleSet, JITParticle, ErrorCode, Variable # AdvectionRK4

from utils.kernels import AdvectionRK4, StokesDrag, DiffusionUniformKh_custom, Advection_and_Diffusion_Backwards
import config
from utils import locate_utils


def simulation_main(cloud_of_particles,
                    start_coordinates, # [lat, lon]
                    start_datetime,
                    simulation_hours):

    # Load the currents fieldset
    fieldset = locate_utils.get_ibi_fieldset()

    # Consider diffusion
    # kh = 0.1  # 0.10-10
    # fieldset.add_constant_field("Kh_zonal", kh, mesh='spherical')
    # fieldset.add_constant_field("Kh_meridional", kh, mesh='spherical')

    # Define particle class
    class PlasticParticle(JITParticle):
        beached = Variable('beached', dtype=np.int32, initial=0.)

    # class PlasticParticle_backwards(JITParticle):
    #     beached = Variable('beached', dtype=np.int32, initial=0.)
    #     center_of_mass_lat = Variable("com_lat", initial=start_coordinates[0])
    #     center_of_mass_lon = Variable("com_lon", initial=start_coordinates[1])
    #     spatial_variance = Variable("spatial_variance", initial=0.0899**2)

    # Create particles
    if cloud_of_particles:  # cloud of particles
        radius_from_origin = 2
        amount_of_particles = 100
        lat, lon, times = locate_utils.point_cloud_from_coordinate(start_coordinates,
                                                                   radius_from_origin,
                                                                   amount_of_particles,
                                                                   start_datetime)
    elif not cloud_of_particles:  # single particle
        lat, lon, times = [], [], []
        for particle_i in start_coordinates:
            print(particle_i)
            aa = input("nk")
            lat.append(particle_i[0])
            lon.append(particle_i[1])
            times.append(start_datetime)

    # Create the particle set
    pset = ParticleSet.from_list(fieldset=fieldset,
                                 pclass=PlasticParticle,
                                 lon=lon,
                                 lat=lat,
                                 time=times)

    # Set kernels
    kernel = pset.Kernel(AdvectionRK4)
    # kernel += pset.Kernel(DiffusionUniformKh_custom)
    # kernel = pset.Kernel(Advection_and_Diffusion_Backwards)

    output_file = pset.ParticleFile(name=config.harbour_particles_sim_filename+f"validation.nc",
                                    outputdt=timedelta(minutes=1)
                                    )
    print("\nSimulation starts...")
    simulation_time_step = 5
    pset.execute(kernel,
                 runtime=timedelta(hours=simulation_hours),
                 dt=-timedelta(minutes=simulation_time_step),
                 output_file=output_file,
                 recovery={ErrorCode.ErrorOutOfBounds: locate_utils.DeleteParticleFunction},
                 verbose_progress=True
                 )

    print("Simulation terminated. Saving...")
    output_file.export()
    output_file.close()
    logger.info(f"Simulation results saved to: {config.harbour_particles_sim_filename}")


if __name__ == '__main__':
    simulation_main(
        # start_coordinates=[40.7,3],
        start_coordinates=[40.695118, 2.629246],
        # start_datetime=np.datetime64(datetime(2022,2,1,10,0,0)),
        start_datetime=np.datetime64(datetime(2022, 3, 1, 2, 0, 0)),
        simulation_hours=24*5,
        cloud_of_particles=True)
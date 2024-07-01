from parcels import rng as random
import math


def AdvectionRK4(particle, fieldset, time):
    if particle.beached == 0:
        (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        lon1, lat1 = (particle.lon + u1 * .5 * particle.dt, particle.lat + v1 * .5 * particle.dt)

        (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1]
        lon2, lat2 = (particle.lon + u2 * .5 * particle.dt, particle.lat + v2 * .5 * particle.dt)

        (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2]
        lon3, lat3 = (particle.lon + u3 * particle.dt, particle.lat + v3 * particle.dt)

        (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3]
        particle.lon += (u1 + 2 * u2 + 2 * u3 + u4) / 6. * particle.dt
        particle.lat += (v1 + 2 * v2 + 2 * v3 + v4) / 6. * particle.dt
        # particle.beached = 2


def DiffusionUniformKh_custom(particle, fieldset, time):
    if particle.beached == 0:
        dWx = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
        dWy = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))

        bx = math.sqrt(2 * fieldset.Kh_zonal[particle])
        by = math.sqrt(2 * fieldset.Kh_meridional[particle])

        particle.lon += bx * dWx
        particle.lat += by * dWy


def Advection_and_Diffusion_Backwards(particle, fieldset, time):
    if particle.beached == 0:

        # Advect center of mass
        (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        particle.com_lon += u1 * particle.dt
        particle.com_lat += v1 * particle.dt

        # Generate cloud around
        particle.spatial_variance += 2 * particle.dt * 10
        F = math.sqrt(particle.spatial_variance)
        Rx = ParcelsRandom.normalvariate(0, 1)
        Ry = ParcelsRandom.normalvariate(0, 1)
        Sx = (ParcelsRandom.randint(0, 1) * 2) - 1
        Sy = (ParcelsRandom.randint(0, 1) * 2) - 1
        Qx = Rx * Sx
        Qy = Ry * Sy

        # Update position
        particle.lon = particle.com_lon + F * Qy
        particle.lat = particle.com_lat + F * Qx


def BeachTesting_2D(particle, fieldset, time):
    # if particle.beached == 2 or particle.beached == 3:
    if particle.beached == 0:
        (u, v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        if fabs(u) < 1e-16 and fabs(v) < 1e-16:
            particle.beached = 1
        else:
            particle.beached = 0


def Ageing(particle, fieldset, time):
    particle.age += particle.dt


def DeleteParticle(particle, fieldset, time):
    print("Particle [%d] lost !! (%g %g %g %g)" % (
    particle.id, particle.lon, particle.lat, particle.depth, particle.time))
    particle.delete()


def StokesDrag(particle, fieldset, time):
    if particle.beached == 0:
        (u_uss, v_uss) = fieldset.UVuss[time, particle.depth, particle.lat, particle.lon]
        particle.lon += u_uss * particle.dt  # les partícules guanyen u_ss m/s
        particle.lat += v_uss * particle.dt


def Leeway(particle, fieldset, time):
    if particle.beached == 0:
        (u_wind, v_wind) = fieldset.UVwind[time, particle.depth, particle.lat, particle.lon]

        rho_air = 1.29
        rho_water = 1025
        C_d = 0.8
        C_w = 1.2
        I = 0.7

        relation_factor = math.sqrt((rho_air * C_d * (1 - I)) / (rho_water * C_w * I))

        u_leeway = u_wind * relation_factor
        v_leeway = v_wind * relation_factor

        particle.lon += u_leeway * particle.dt
        particle.lat += v_leeway * particle.dt

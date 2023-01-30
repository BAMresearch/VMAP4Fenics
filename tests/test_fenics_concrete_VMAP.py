# the order of imports is IMPORTANT
from vmap4fenics import VMAP4Fenics
from dolfin import *
import fenics_concrete
import numpy as np

# define problem
parameters = fenics_concrete.Parameters()
parameters['E'] = 30000
parameters['nu'] = 0.2
parameters['height'] = 300
parameters['radius'] = 75
parameters['mesh_density'] = 6
parameters['log_level'] = 'WARNING'
parameters['bc_setting'] = 'fixed'
parameters['dim'] = 3
experiment = fenics_concrete.ConcreteCylinderExperiment(parameters)
problem = fenics_concrete.LinearElasticity(experiment, parameters)
# define sensors
problem.add_sensor(fenics_concrete.sensors.ReactionForceSensorBottom())

# define wrapper
wrapper = VMAP4Fenics.VMAP4Fenics(filename = 'test', output_path = 'resultsCyl')
wrapper.write_metadata()
wrapper.write_unitsystem()
wrapper.write_coordinatesystem()
wrapper.set_material(name = 'Linear_Concrete',
					material_state = 'solid',
					material_type = 'concrete',
					material_description= 'linear elastic model',
					material_id = 'linear_concrete_model',  # unique material id
					material_idealization = 'continuum',
					physics = 'solid mechanics')
wrapper.set_geometry(problem.V, [problem.a, problem.L])

# define displacement
displacement_list = [1,5,10]
# loop over all measured displacements
for displacement in displacement_list:
	# apply displacement
	problem.experiment.apply_displ_load(displacement)
	# solve problem
	wrapper.next_state()
	problem.solve()  # solving this
	for sensorname, sensor in problem.sensors.items():
		wrapper.set_variable(sensorname, sensor.data[sensor.dataoffset[-1]:len(sensor.data)], sensor.LOCATION)
	wrapper.write_state()
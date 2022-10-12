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
problem.set_material(name = 'Linear_Concrete',
					state = 'solid',
					type= 'concrete',
					description= 'linear elastic model',
					material_id = 'linear_concrete_model',  # unique material id
					idealization = 'continuum',
					physics = 'solid mechanics'
					)
# define sensors
problem.add_sensor(fenics_concrete.sensors.ReactionForceSensorBottom())

# define wrapper
wrapper = VMAP4Fenics.VMAP4Fenics(filename = 'test', paraview_output = False, output_path = 'resultsCyl')
wrapper.set_metadata(user_id='ahannes')
wrapper.setup(problem)

# define displacement
displacement_list = [1,5,10]
# loop over all measured displacements
for displacement in displacement_list:
	# apply displacement
	problem.experiment.apply_displ_load(displacement)
	# solve problem
	problem.solve()  # solving this
	# define States
	def evaluation_function(dict):
		location, data = dict['ReactionForceSensorBottom']
		data = np.asarray(data).reshape(-1,1)/(np.pi*parameters['radius']**2)
		dict['SPECIMEN_STRESS'] = location, data
		return dict
	wrapper.write_state(problem, evaluation_function)

# write wrapper
wrapper.export_to_vmap()

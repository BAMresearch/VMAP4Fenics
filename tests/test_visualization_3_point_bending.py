# the order of imports is IMPORTANT
from vmap4fenics import VMAP4Fenics
import dolfin as df
import fenics_concrete

def three_point_bending_example(E, nu):
	"""Example of a linear elastic three point bending test
	Parameters
	----------
		E : float
			Young's modulus in N/mm²
		nu : float
			Poisson's ratio
	Returns
	-------
		stress_in_x : float
			Stress in x direction in the center at the bottom, where the maximum stress is expected
	"""
	# setting up the simulation parameters
	parameters = fenics_concrete.Parameters()  # using the current default values
	# input values for the material
	parameters['E'] = E
	parameters['nu'] = nu
	# definition of the beam and mesh
	parameters['dim'] = 3
	parameters['mesh_density'] = 4  # number of elements in vertical direction
	parameters['height'] = 300  # in mm
	parameters['length'] = 2000  # in mm
	parameters['width'] = 150  # in mm
	parameters['log_level'] = 'WARNING'

	# displacement load in the center of the beam
	displacement = -10  # displacement load in the center of the beam in mm

	# setting up the problem
	experiment = fenics_concrete.ConcreteBeamExperiment(parameters)
	problem = fenics_concrete.LinearElasticity(experiment, parameters)
	problem.set_material(name = 'Linear_Concrete_Beam',
					state = 'solid',
					type= 'concrete',
					description= 'linear elastic model',
					material_id = 'linear_concrete_model',  # unique material id
					idealization = 'continuum',
					physics = 'solid mechanics'
					)
	wrapper = VMAP4Fenics.VMAP4Fenics(filename = 'test', paraview_output = False, output_path = 'resultsBeam')
	wrapper.write_metadata(user_id='ahannes')
	wrapper.setup(problem)

	# applying the load
	problem.experiment.apply_displ_load(displacement)

	# applying the stress sensor
	stress_sensor = fenics_concrete.sensors.StressSensor(df.Point(parameters.length/2, 0, 0))
	problem.add_sensor(stress_sensor)

	# solving the problem
	problem.solve(t=0)  # solving this
	def evaluation_function(dict):
		location, data = dict['StressSensor']
		data = data[0]
		dict['StressSensor'] = location, data
		return dict
	wrapper.write_state(problem, evaluation_function)
	wrapper.export_to_vmap()


# example of how to use this function
# defining the material parameters
E = 30000  # N/mm²
nu = 0.2
three_point_bending_example(E, nu)
# resulting stress in x direction in the bottom center of the beam

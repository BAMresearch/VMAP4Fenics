from datetime import datetime  # to get current time
from os import mkdir
from os.path import isdir

import dolfin as df  # to get some solution fields... maybe...
import numpy as np  # to do some vector cross products etc
import PyVMAP as VMAP  # VMAP Python Interface
from ffc.fiatinterface import \
    create_quadrature  # to estimate number of integration points
from ufl import Form as Form  # only for type anotations
from ufl.algorithms import estimate_total_polynomial_degree, expand_derivatives

# current limitations:
# only one geometry id, one coordinate system, element type, material, etc...

class VMAP4Fenics():
	def __init__(self, filename = 'fenics_to_vmap', output_path = ''):

		# initiale the vmap object
		VMAP.Initialize()
		if not isdir(output_path): mkdir(output_path)
		self.vmap_file = VMAP.VMAPFile(f'{output_path}/{filename}_VMAP.h5')

		self.state_id = 0
		self.time = 0

	def write_metadata(self, user_id = 'unknown', description = 'FEM Simulation'):
		meta_info = VMAP.sMetaInformation()
		# tool/solver name
		meta_info.setExporterName('Fenics to VMAP Wrapper')
		# current time and date
		now = datetime.now()
		current_time = now.strftime('%H:%M:%S')
		meta_info.setFileTime(current_time)
		today = datetime.today()
		current_date = today.strftime('%Y-%m-%d')
		meta_info.setFileDate(current_date)
		# file description
		meta_info.setDescription(description)
		# add user id
		meta_info.setUserId(user_id)
		# analysis type
		meta_info.setAnalysisType('FEM Analysis')
		self.vmap_file.writeMetaInformation(meta_info)

	def write_unitsystem(self):
		unitSystem=VMAP.sUnitSystem()
		unitSystem.getLengthUnit().setUnitSymbol(             "m")
		unitSystem.getMassUnit().setUnitSymbol(              "kg")
		unitSystem.getTimeUnit().setUnitSymbol(               "s")
		unitSystem.getCurrentUnit().setUnitSymbol(            "A")
		unitSystem.getTemperatureUnit().setUnitSymbol(        "K")
		unitSystem.getAmountOfSubstanceUnit().setUnitSymbol("mol")
		unitSystem.getLuminousIntensityUnit().setUnitSymbol( "cd")
		self.vmap_file.writeUnitSystem(unitSystem)

	def write_coordinatesystem(self):
		self.coordinatesystem_id = 1
		systems = []
		systems.append(np.array((self.coordinatesystem_id, VMAP.sCoordinateSystem.CARTESIAN_LEFT_HAND,  (0., 0., 0.), (1., 0., 0., 0., 1., 0., 0., 0., 1.)), dtype=VMAP.sCoordinateSystem))
		vsystems=VMAP.VectorTemplateCoordinateSystem()

		for item in systems:
			csys = VMAP.sCoordinateSystem()
			csys.myIdentifier = item[0]
			csys.myType = item[1]
			csys.setReferencePoint(item[2])
			csys.setAxisVector(0,item[3][0:3])
			csys.setAxisVector(1,item[3][3:6])
			csys.setAxisVector(2,item[3][6:9])
		vsystems.push_back(csys)
		self.vmap_file.writeCoordinateSystems("/VMAP/SYSTEM",vsystems)	

	def set_material(self, material_name,
					material_id, 
					material_type = 'unknown', 
					material_description = 'unknown', 
					material_state = 'solid', 
					material_idealization = 'unknown', 
					physics = 'unknown', 
					solution = 'unknown', 
					paramters = None):
		self.material = VMAP.sMaterial()
		# TODO automatic material counter, maybe with optional direct number attribuite and a check if id exists...
		self.material.setMaterialName(material_name.upper())
		# solid, liquid, gas ...
		self.material.setMaterialState(material_state)
		self.material.setMaterialType(material_type)
		self.material.setMaterialDescription(material_description)
		self.material.setMaterialSupplier('not applicable')

		# material card, supposed to be solver specific???
		material_card = VMAP.sMaterialCard()
		material_card.setIdentifier(material_id)

		# Provides the idealisation of the material e.g. shell, beam, solid etc.
		material_card.setIdealization(material_idealization)
		material_card.setModelName(material_id)     # one more name...
		material_card.setPhysics(physics)           # Explains the physics behind the material e.g. solid mechanics, fluid mechanics, heat transfer etc.
		material_card.setSolution(solution)         # Explains the type of solution e.g. implicit, explicit etc.
		material_card.setSolver('Fenics/Dolfin')    # Provides the solver from which the material card has been taken.
		material_card.setSolverVersion(df.__version__)

		if paramters:
			material_parameter = VMAP.sParameter()
			vector_parameter = VMAP.VectorTemplateParameter()
			for paramter in paramters:
				material_parameter.setName(str(paramter[0]))
				material_parameter.setDescription(str(paramter[1]))
				material_parameter.setValue(str(paramter[2]))
				vector_parameter.push_back(material_parameter)
			material_card.setParameters(vector_parameter)
		self.material.setMaterialCard(material_card)
		self.vector_material = VMAP.VectorTemplateMaterial()
		self.vector_material.push_back(self.material)
		self.vmap_file.writeMaterialBlock(self.vector_material)

	def next_state(self, dt = 0, incr = 1, state_name = None, time = None):
		self.state = VMAP.VectorTemplateStateVariable()
		self.state_id += incr
		self.time = time if time else self.time + dt
		if not state_name: state_name = f'Loading at time {self.time}'
		self.variable_id = 0

		# add new vmap state
		self.vmap_file.createVariablesGroup(self.state_id, self.geometry_id)
		# increment is preset to 1. values larger 1 would allow for multiple increments in one step (for whatever reason)
		self.vmap_file.setVariableStateInformation(stateId = self.state_id, stateName = state_name,
										 totalTime = self.time, stepTime = dt, increment = 1)

	def write_state(self):
		self.vmap_file.writeVariablesBlock(f"/VMAP/VARIABLES/STATE-{self.state_id}/{self.geometry_id}", self.state)

	def set_geometry(self, vector_function_space : df.VectorFunctionSpace, functions : list[str, Form] = None, geometry_name : str = 'Mesh'):
		# includes data for gemoetry: nodes, element, element type
		self.vector_function_space = vector_function_space

		# so far only one id
		self.geometry_id = 1
		element_type_id = 1
		material_id = 1

		# ------------------------------------------------------------------------
		# EXTRACT MESH DATA
		# ------------------------------------------------------------------------
		# element dimension (1D, 2D, 3D)
		self.spacial_dimension = len(self.vector_function_space.mesh().coordinates()[0])
		# number of elements in mesh
		n_elements = self.vector_function_space.mesh().num_cells()
		# number of dofs per "node"/vertex
		self.n_node_dofs = self.vector_function_space.dofmap().block_size()
		# type of elements, eg triangle, tetrahedron
		self.element_shape = self.vector_function_space.ufl_element().cell()
		# polynomial degree
		self.element_polynomial_degree = self.vector_function_space.ufl_element().degree()
		# get the vmap element type and number of nodes based on fenics information
		vmap_element_type_number, n_element_nodes = self._determine_vmap_element(self.element_shape, self.element_polynomial_degree)
		# get a list of nodal coordinates
		node_list, _element_list = self._determine_nodes_and_connectivity(n_element_nodes)
		# number of nodes in mesh
		n_nodes = len(node_list)
		# ------------------------------------------------------------------------
		# SET ELEMENT TYPE DATA
		# ------------------------------------------------------------------------
		# set elementtype thing, without integration information...
		# currently element dim = mesh dimension
		# ---------------------------------------------
		# set the interploation type
		interpolatonType = self._determine_vmap_interpolation(self.element_polynomial_degree)

		# ------------------------------------------------------------------------
		# SET GEOMETRY
		# ------------------------------------------------------------------------
		# svmap geometry group
		self.vmap_file.createGeometryGroup(self.geometry_id, geometry_name)

		# points
		points = VMAP.sPointsBlock(n_nodes)
		# add coordinates to points
		for i in range(n_nodes):
			points.setPoint(i, i, node_list[i])
		self.vmap_file.writePointsBlock(f"/VMAP/GEOMETRY/{self.geometry_id}", points)

		# elements
		element_block = VMAP.sElementBlock(n_elements)
		for i in range(n_elements):
			# generating vmap element object
			element = VMAP.sElement()
			# set element type
			element.myElementType = vmap_element_type_number
			# determine the correct order of elemental nodes
			_vmap_connectivity = self._determine_order_of_nodes(vmap_element_type_number, _element_list[i], node_list)
			# set the connectivity for each element
			element.setConnectivity(_vmap_connectivity)
			# define element id
			element.myIdentifier = i
			# global coordinate system
			element.myCoordinateSystem = self.coordinatesystem_id # as defined earlier...
			# set material, currently preset to 1
			element.myMaterialType = material_id  # currently only one material
			# add element to geometry object
			element_block.setElement(i, element)
		# write elements to file
		self.vmap_file.writeElementsBlock(f"/VMAP/GEOMETRY/{self.geometry_id}", element_block)

		# elementtype_args
		elementtype_args = [self.spacial_dimension,vmap_element_type_number, interpolatonType, self.set_integrationtype(functions)]

		# integrationtype
		vector_integrationtypes = VMAP.VectorTemplateIntegrationType()
		integr_type = VMAP.VMAPIntegrationTypeFactory_createVMAPIntegrationType(elementtype_args[3])
		vector_integrationtypes.push_back(integr_type)
		self.vmap_file.writeIntegrationTypes(vector_integrationtypes)

		# elementtype
		element_type = VMAP.VMAPElementTypeFactory.createVMAPElementType(*elementtype_args)
		element_type.setIdentifier(element_type_id)
		vector_elementtype = VMAP.VectorTemplateElementType()
		vector_elementtype.push_back(element_type)
		self.vmap_file.writeElementTypes(vector_elementtype)

	def set_variable(self,name, values, location, coordinatesystem = 1, description = None):
		if isinstance(values, np.ndarray):
			if values.ndim == 1:
				dimension = len(values)
				value_list = values.tolist()
			elif values.ndim == 2:
				dimension = values.shape[1]
				value_list = values.flatten().tolist()
			else: raise NotImplementedError(f"Dim {values.ndim} not supported")
		elif isinstance(values, list):
			dimension = len(values)
			value_list = values
		elif isinstance(values, float):
			dimension = 1
			value_list = [values]
		else:
			raise NotImplementedError(f"Type {type(values)} not supported")

		# generate description from name
		if not description:
			description = 'Data field for ' + name.lower()

		# TODO ask what is the most user friendly way...
		# convert location description to VMAP loaction number
		if isinstance(location, str): location_int = self._get_location_int_from_location(location)
		else: raise NotImplementedError()

		# TODO setGeometryIds for which this data is defined.... how.... ?????
		# displacement.setGeometryIds(0) ???

		# define variable field parameter
		variable = VMAP.sStateVariable()
		variable.setIdentifier(self.variable_id)
		variable.setCoordinateSystem(coordinatesystem)
		variable.setVariableName(name + str(self.variable_id))
		variable.setVariableDescription(description)
		variable.setDimension(dimension)
		variable.setLocation(loc = location_int)
		variable.setValues(value_list)

		# MYMULTIPLICITY refers to the numer of columns in the MYVALUES dataset,
		# the majority of state variables have a multiplicity of 1
		variable.setMultiplicity(1)

		# add variable to state object
		self.state.push_back(variable)
		self.variable_id += 1

	def _get_location_int_from_location(self, location):
		loc2locint = {'GLOBAL': 1, 'NODE': 2, 'ELEMENT': 3, 'INTEGRATION POINT': 4, 'ELEMENT FACE': 5}
		if location.upper() in loc2locint: return loc2locint[location.upper()]
		raise ValueError(f'''{location} is an unexpected location string. 
					Use 'GLOBAL', 'NODE', 'INTEGRATION POINT' or 'ELEMENT FACE' as Input.''')

	# TODO currently not used
	def get_location_int_from_num(self, location):
		mesh = self.vector_function_space.mesh()
		num2locint = {1: 1, mesh.num_vertices(): 2, mesh.num_cells(): 3, self.num_integration_points: 4, mesh.num_faces(): 5}
		if location in num2locint: return num2locint[location]
		else: raise ValueError(location)

	def set_integrationtype(self, functions : list[str, Form]):
		# for each form passed to function, test maximun degree
		max_degree = 0
		for item in functions:
			# get estimated degree from form
			degree = estimate_total_polynomial_degree(expand_derivatives(item))
			# test maximum
			if degree > max_degree:
				max_degree = degree

		# get numuber of quadrature points for estimated degree
		# currently this number is applied to standard VMAP library
		# currently set to default
		# TODO get the integration scheme from fenics data
		# it would be possible to use the data to generate further integration schemes
		self.num_integration_points = len(create_quadrature(self.element_shape, max_degree, 'default')[1])

		# set VMAP integration type number
		if str(self.element_shape) == 'triangle':
			if self.num_integration_points == 1: integration_id = 500 	#GAUSS_TRIANGLE_1
			elif self.num_integration_points == 3: integration_id = 501	#GAUSS_TRIANGLE_3
			elif self.num_integration_points == 4: integration_id = 502	#GAUSS_TRIANGLE_4
			elif self.num_integration_points == 6: integration_id = 503	#GAUSS_TRIANGLE_6
			else: raise ValueError(f'{self.num_integration_points} is an unexpected number of integration points for a triangle. Use 1, 3, 4 or 6 as Input.')
		elif str(self.element_shape) == 'tetrahedron':
			if self.num_integration_points == 1: integration_id = 800 		#GAUSS_TETRAHEDRON_1
			elif self.num_integration_points == 4: integration_id = 801 	#GAUSS_TETRAHEDRON_4
			elif self.num_integration_points == 8: integration_id = 802 	#GAUSS_TETRAHEDRON_8
			elif self.num_integration_points == 11: integration_id = 803	#GAUSS_TETRAHEDRON_11
			elif self.num_integration_points == 15: integration_id = 804	#GAUSS_TETRAHEDRON_15
			else: raise ValueError(f'{self.num_integration_points} is an unexpected number of integration points for a tetrahedron. Use 1, 4, 8, 11, 15 as Input.')
		else: raise ValueError(f'''{str(self.element_shape)} is an unexpected element shape. Use 'triangle' or 'tetrahedron'.''')
		return integration_id

	def _determine_vmap_element(self, element_type, element_degree):
		# get number of "nodes" from element type and polynomial degree
		if str(element_type) == 'triangle':
			if element_degree == 1: 	return VMAP.sElementType.TRIANGLE_3, 3
			elif element_degree == 2: 	return VMAP.sElementType.TRIANGLE_6, 6
			else: raise ValueError(f'{element_degree} is an unexpected element degree for a triangle. Use 1 or 2 as Input.')
		elif str(element_type) == 'tetrahedron':
			if element_degree == 1: 	return VMAP.sElementType.TETRAHEDRON_4, 4
			elif element_degree == 2: 	return VMAP.sElementType.TETRAHEDRON_10, 10
			else: raise ValueError(f'{element_degree} is an unexpected element degree for a tetrahedron. Use 1 or 2 as Input.')
		else: raise ValueError(f'{str(element_type)} is an unexpected element type.')

	def _determine_vmap_interpolation(self, element_degree):
		# function to return the VMAP interpolation ID
		# currently only based on the polynomial degree of the element, can be extended
		if element_degree == 1: return 2	# linear
		elif element_degree == 2: return 5	# quadratic
		else: raise ValueError(f'{element_degree} is an unexpected element degree. Use 1 or 2 as Input.')

	def _determine_nodes_and_connectivity(self, n_element_nodes):

		# generating a list per element with nodal coordinates
		# generating a dictionary with coordinates as keys and node numbers as values
		dof_coordinates = self.vector_function_space.tabulate_dof_coordinates()

		# list with coordinates of "nodes" in order of dofs
		node_list = dof_coordinates[::self.n_node_dofs].tolist()

		# number of elements
		self.n_elements = self.vector_function_space.mesh().num_cells()

		# get the connectivity in terms of nodal number instead of dofs
		element_conectivity = []
		# loop over all elements
		for i in range(self.n_elements):
			# get the dof numbers for element, divide to convert to nude number
			element_dofs = self.vector_function_space.dofmap().cell_dofs(i)/self.n_node_dofs
			element_conectivity.append(element_dofs[:n_element_nodes].astype(int).tolist())

		return node_list, element_conectivity


	def _determine_order_of_nodes(self, element_type, fenics_connectivity, nodes):
		# the order of the nodes, defining the connectivity is standardized in vmap
		# fenics changes the order to optimize integration, therefore the correct order must be determined

		if  element_type == VMAP.sElementType.TRIANGLE_3:
			# 3 nodes, 2D, check orientation
			# calculating dirrection of the normal vector on the plane
			p1 = np.array(nodes[fenics_connectivity[0]])  # corrdinate 1
			p2 = np.array(nodes[fenics_connectivity[1]])  # coordinate 2
			p3 = np.array(nodes[fenics_connectivity[2]])  # coordinate 3
			# calculating the cross product of the difference
			cp = np.cross(p3 - p1, p2 - p1)
			return fenics_connectivity if cp < 0 else [fenics_connectivity[0], fenics_connectivity[2], fenics_connectivity[1]]
		elif  element_type == VMAP.sElementType.TRIANGLE_6:
			# 6 nodes, 2D, check orientation
			# calculating dirrection of the normal vector on the plane
			p1 = np.array(nodes[fenics_connectivity[0]])  # corrdinate 1
			p2 = np.array(nodes[fenics_connectivity[1]])  # coordinate 2
			p3 = np.array(nodes[fenics_connectivity[2]])  # coordinate 3
			# calculating the cross product of the difference
			cp = np.cross(p3 - p1, p2 - p1)
			if cp < 0:
				return [fenics_connectivity[0],
						fenics_connectivity[1],
						fenics_connectivity[2],
						fenics_connectivity[5],
						fenics_connectivity[3],
						fenics_connectivity[4]]
			return [fenics_connectivity[0],
					fenics_connectivity[2],
					fenics_connectivity[1],
					fenics_connectivity[4],
					fenics_connectivity[3],
					fenics_connectivity[5]]
		elif  element_type == VMAP.sElementType.TETRAHEDRON_4:
			p1 = np.array(nodes[fenics_connectivity[0]])  # corrdinate 1
			p2 = np.array(nodes[fenics_connectivity[1]])  # coordinate 2
			p3 = np.array(nodes[fenics_connectivity[2]])  # coordinate 3
			p4 = np.array(nodes[fenics_connectivity[3]])  # coordinate 4
			# compute normal
			nv = np.cross(p3 - p1, p2 - p1)
			# checking on which side of the plane P4 lies
			result = nv[0] * (p4[0] - p1[0]) + nv[1] * (p4[1] - p1[1]) + nv[2] * (p4[2] - p1[2])
			if result < 0:  # "correct side", there right numbering
				return fenics_connectivity
			return [fenics_connectivity[0],
					fenics_connectivity[2],
					fenics_connectivity[1],
					fenics_connectivity[3]]
		elif  element_type == VMAP.sElementType.TETRAHEDRON_10:
			p1 = np.array(nodes[fenics_connectivity[0]])  # corrdinate 1
			p2 = np.array(nodes[fenics_connectivity[1]])  # coordinate 2
			p3 = np.array(nodes[fenics_connectivity[2]])  # coordinate 3
			p4 = np.array(nodes[fenics_connectivity[3]])  # coordinate 3

			# compute normal
			nv = np.cross(p3 - p1, p2 - p1)
			# checking on which side of the plane P4 lies
			result = nv[0] * (p4[0] - p1[0]) + nv[1] * (p4[1] - p1[1]) + nv[2] * (p4[2] - p1[2])
			if result < 0:  # "correct side", there right numbering
				return [fenics_connectivity[0],
						fenics_connectivity[1],
						fenics_connectivity[2],
						fenics_connectivity[3],
						fenics_connectivity[9],
						fenics_connectivity[6],
						fenics_connectivity[8],
						fenics_connectivity[7],
						fenics_connectivity[5],
						fenics_connectivity[4]]
			return [fenics_connectivity[0],
					fenics_connectivity[2],
					fenics_connectivity[1],
					fenics_connectivity[3],
					fenics_connectivity[8],
					fenics_connectivity[6],
					fenics_connectivity[9],
					fenics_connectivity[7],
					fenics_connectivity[4],
					fenics_connectivity[5]]
		else: raise ValueError(f'{element_type} is an unexpected element type.')
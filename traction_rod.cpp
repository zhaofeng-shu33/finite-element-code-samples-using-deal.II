#include "post_processing.h"
/** fem computation of an  engineering structure
*/
namespace TractionRodComputation{
/** static strength analysis of a traction rod, which is an import part in bogie.
* no analytical solution available.
* Displacement Unit:mm.
*/
    using namespace post_processing;
template <int dim>
class TractionRod
{
    typedef ComputeStressField<dim,TractionRod<dim>> BeamPostProcessing;    
	friend class BeamPostProcessing;
public:
	TractionRod(double given_normal_force_per_area,char* inputMshFileName);
	void run();
private:
	static const double YoungModulus;
	static const double PoissonRatio;
	static const double shearModulus;
	static const double LamesFirstParameter;
	void get_neumann_value(const Point<dim>   &p, Vector<double>  &values)const;
	void make_grid();
	void read_grid(char* fileName);
	void write_grid(char* fileName);
	void setup_system();
	void setup_system_from_saved_solution();
	void storeGradientToMap();
	void assemble_system();
	void test_boundary();//!< output boundary infomation to console
	void solve();
	void output_results(char* fileName) const;
	Triangulation<dim>     triangulation;
	FESystem<dim>               fe;
	ConstraintMatrix     constraints;
	DoFHandler<dim>        dof_handler;
	SparsityPattern      sparsity_pattern;
	SparseMatrix<double> system_matrix;
	Vector<double>       solution;
	Vector<double>       system_rhs;
	double normal_force_per_area;
    char mshFileName[20];
};
template <int dim>
const double TractionRod<dim>::YoungModulus = 2.1;
template <int dim>
const double TractionRod<dim>::PoissonRatio = 0.3;
template <int dim>
const double TractionRod<dim>::shearModulus = YoungModulus / (2 * (1 + PoissonRatio));
template <int dim>
const double TractionRod<dim>::LamesFirstParameter = YoungModulus*PoissonRatio / ((1 + PoissonRatio)*(1 - 2 * PoissonRatio));
template <int dim>
TractionRod<dim>::TractionRod(double given_normal_force_per_area,char* inputMshFileName) :
	fe(FE_Q<dim>(1), dim),
	dof_handler(triangulation)
{
	normal_force_per_area = given_normal_force_per_area;
    strcpy_s(mshFileName,inputMshFileName);//destination<=src
}
template <int dim>
void TractionRod<dim>::test_boundary() {//Faces with external force with boundary id=1
	typename DoFHandler<dim>::active_cell_iterator
		cell = dof_handler.begin_active(),
		endc = dof_handler.end();
	for (; cell != endc; ++cell) {
		for (unsigned int face_number = 0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
			if (cell->face(face_number)->at_boundary()
				&&
				(cell->face(face_number)->boundary_id() == 1)) {
				std::cout << "one face has boundary id =1\n";
				for (int vertex_number = 0; vertex_number < GeometryInfo<dim>::vertices_per_face; vertex_number++) {
					std::cout << "vertex: " << vertex_number << ' ' << cell->face(face_number)->vertex(vertex_number);
				}
			}
	}
}
template <int dim>
void TractionRod<dim>::storeGradientToMap() {
	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
	DoFHandler<dim>::active_cell_iterator endc = dof_handler.end();
	QGaussLobatto<dim>  quadrature_formula(2);//2^dim quadrature points
	FEValues<dim> fe_values(fe, quadrature_formula, update_gradients);
	int cell_index = 0;
	for (; cell != endc; ++cell) {
		cell_index += 1;
		fe_values.reinit(cell);
		cell->get_dof_indices(local_dof_indices);
		for (int i = 0; i < GeometryInfo<dim>::vertices_per_cell; i++) {
			Tensor<2, dim> myGradient;
			for (int j = 0; j < dofs_per_cell; j++) {
				const unsigned int
					component_j = fe.system_to_component_index(j).first;
				myGradient[component_j] += solution[local_dof_indices[j]] * fe_values.shape_grad(j, i);
			}
			for (int k = 0; k<dim; k++) {
				if (my_gradient_map<dim>.find(local_dof_indices[dim*i + k]) == my_gradient_map<dim>.end()) {//not found
					my_gradient_map<dim>[local_dof_indices[dim*i + k]] = myGradient[k];
					my_gradient_sum[local_dof_indices[dim*i + k]] = 1;
				}
				else {
					my_gradient_map<dim>[local_dof_indices[dim*i + k]] += myGradient[k];
					my_gradient_sum[local_dof_indices[dim*i + k]]++;
				}
			}

		}
	}
	//std::cout << "cellIndex:" << cell_index << '\n';
	//std::cout << "vertices per cell:" << GeometryInfo<dim>::vertices_per_cell << '\n';
}
template <int dim>
void TractionRod<dim>::get_neumann_value(const Point<dim>   &p, Vector<double>  &values) const {//vector valued function
//	std::cout << p[1] << '\n';
	values[0] = 0;
	values[2] = 0;
	if (p[1]<0) {// sigma_x=6*a*y
		values[1] =-normal_force_per_area;
	}
	else {
		values[1] = normal_force_per_area;
	}
}

template <int dim>
void TractionRod<dim>::read_grid(char* fileName) {
	GridIn<dim> grid_in;
	grid_in.attach_triangulation(triangulation);
	std::ifstream input_file(fileName);
	grid_in.read_msh(input_file);
	//triangulation.refine_global(2);
	std::cout << "Number of active cells: "
		<< triangulation.n_active_cells()
		<< std::endl;
}
template <int dim>
void TractionRod<dim>::write_grid(char* fileName) {
	std::ofstream out(fileName);
	GridOut grid_out;
	grid_out.write_vtk(triangulation, out);

	std::cout << "Grid written to grid-" << fileName << std::endl;
}
template <int dim>
void TractionRod<dim>::run()
{
	read_grid(mshFileName);
//	test_boundary();
	//no need to refine the mesh
		setup_system();
	assemble_system();
	std::cout << "assemble system finished...\n";
	solve();
	std::cout << "solve finished...\n";
	//setup_system_from_saved_solution();
	storeGradientToMap();
	output_results("solution_TractionRod.vtk");
}
template <int dim>
void TractionRod<dim>::make_grid()
{
	GridGenerator::hyper_shell(triangulation, Point<dim>(), inner_radius, outer_radius);
	static const SphericalManifold<dim> boundary;
	triangulation.set_all_manifold_ids_on_boundary(0);
	triangulation.set_manifold(0, boundary);
	triangulation.refine_global(2);
	std::cout << "Number of active cells: "
		<< triangulation.n_active_cells()
		<< std::endl;
}template <int dim>
void TractionRod<dim>::setup_system_from_saved_solution() {
	dof_handler.distribute_dofs(fe);
	std::cout << "Number of degrees of freedom: "
		<< dof_handler.n_dofs()
		<< std::endl;
	std::ifstream file{ "solution_archive.txt" };
	boost::archive::text_iarchive ia(file);
	solution.load(ia, 1);
}
template <int dim>
void TractionRod<dim>::setup_system()
{
	dof_handler.distribute_dofs(fe);
	std::cout << "Number of degrees of freedom: "
		<< dof_handler.n_dofs()
		<< std::endl;
	DynamicSparsityPattern dsp(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp);
	sparsity_pattern.copy_from(dsp);
	system_matrix.reinit(sparsity_pattern);
	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());
}
template <int dim>
void TractionRod<dim>::assemble_system()
{
	constraints.clear();
	QGauss<dim>  quadrature_formula(2);
	QGauss<dim - 1> face_quadrature_formula(2);
	const RightHandSide<dim> right_hand_side;
	FEValues<dim> fe_values(fe, quadrature_formula,
		update_values | update_gradients | update_JxW_values | update_quadrature_points);
	FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
		update_values | update_quadrature_points |
		update_normal_vectors | update_JxW_values);
	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	const unsigned int   n_q_points = quadrature_formula.size();
	const unsigned int n_face_q_points = face_quadrature_formula.size();
	FullMatrix<double>   cell_matrix(dofs_per_cell, dofs_per_cell);
	Vector<double>       cell_rhs(dofs_per_cell);
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
	DoFHandler<dim>::active_cell_iterator endc = dof_handler.end();
	std::vector<Vector<double>> rhs_values(n_q_points);
	for (; cell != endc; ++cell)
	{
		cell_matrix = 0;
		cell_rhs = 0;
		fe_values.reinit(cell);
		right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_values);
		//assemble local matrix per cell
		for (unsigned int i = 0; i<dofs_per_cell; ++i)
		{
			const unsigned int
				component_i = fe.system_to_component_index(i).first;
			for (unsigned int j = 0; j<dofs_per_cell; ++j)
			{
				const unsigned int
					component_j = fe.system_to_component_index(j).first;
				for (unsigned int q_point = 0; q_point<n_q_points;
					++q_point)
				{
					cell_matrix(i, j)
						+=
						(
						(fe_values.shape_grad(i, q_point)[component_i] *
							fe_values.shape_grad(j, q_point)[component_j] *
							LamesFirstParameter)
							+
							(fe_values.shape_grad(i, q_point)[component_j] *
								fe_values.shape_grad(j, q_point)[component_i] *
								shearModulus)
							+
							((component_i == component_j) ?
							(fe_values.shape_grad(i, q_point) *
								fe_values.shape_grad(j, q_point) *
								shearModulus) :
								0)
							)
						*
						fe_values.JxW(q_point);
				}
			}
		}
		//assemble RHS per cell
		for (unsigned int i = 0; i<dofs_per_cell; ++i)
		{
			const unsigned int
				component_i = fe.system_to_component_index(i).first;
			for (unsigned int q_point = 0; q_point<n_q_points; ++q_point)
				cell_rhs(i) += fe_values.shape_value(i, q_point) *
				rhs_values[q_point][component_i] *
				fe_values.JxW(q_point);
		}
		Vector<double> neumann_value(dim);
		//assemble neumann bc per cell
		for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
			if (cell->face(face_number)->at_boundary() && cell->face(face_number)->boundary_id() == 1)
			{
				fe_face_values.reinit(cell, face_number);
				for (unsigned int q_point = 0; q_point<n_face_q_points; ++q_point)
				{
					get_neumann_value(fe_face_values.quadrature_point(q_point), neumann_value);
					for (unsigned int i = 0; i < dofs_per_cell; ++i) {
						const unsigned int
							component_face_i = fe.system_to_component_index(i).first;

						cell_rhs(i) += (neumann_value[component_face_i] *
							fe_face_values.shape_value(i, q_point) *
							fe_face_values.JxW(q_point));
					}
				}
			}
		cell->get_dof_indices(local_dof_indices);
		//assemble dirichlet bc per cell
		/*for (int vertex_number = 0; vertex_number<GeometryInfo<dim>::vertices_per_cell; vertex_number++) {
			Point<dim> one_vertex = cell->vertex(vertex_number);
			if (abs(one_vertex[0] - TractionRod::length_of_TractionRod / 2) < 1e-5 && abs(one_vertex[1]) < 1e-5) {
				constraints.add_line(local_dof_indices[dim*vertex_number + 1]);
			}
			else if (abs(one_vertex[0] + TractionRod::length_of_TractionRod / 2) < 1e-5 && abs(one_vertex[1]) < 1e-5) {
				constraints.add_line(local_dof_indices[dim*vertex_number]);
				constraints.add_line(local_dof_indices[dim*vertex_number + 1]);
			}
		}*/
		for (unsigned int i = 0; i<dofs_per_cell; ++i)
		{
			for (unsigned int j = 0; j<dofs_per_cell; ++j)
				system_matrix.add(local_dof_indices[i],
					local_dof_indices[j],
					cell_matrix(i, j));
			system_rhs(local_dof_indices[i]) += cell_rhs(i);
		}
	}
	constraints.close();//the first vertex is fixed
	//std::cout << "Constraint Matrix: ";
	//constraints.print(std::cout);
	constraints.condense(system_matrix, system_rhs);
}
template <int dim>
void TractionRod<dim>::solve()
{
	//non symetric matrix
	SolverControl           solver_control(4000, 1e-5);
	SolverGMRES<>              solver(solver_control);
	solver.solve(system_matrix, solution, system_rhs,
		PreconditionIdentity());
	constraints.distribute(solution);
	std::ofstream file{ "solution_archive.txt" };
	boost::archive::text_oarchive oa(file);
	solution.save(oa, 1);
}
template <int dim>
void TractionRod<dim>::output_results(char* fileName) const//post processing
{
	DataOut<dim> data_out;
	data_out.attach_dof_handler(dof_handler);
	BeamPostProcessing my_stress_field;
	data_out.add_data_vector(solution, my_stress_field);
	data_out.build_patches();
	std::ofstream output(fileName);
	data_out.write_vtk(output);
}
}
int main(int argc, char** argv)
{
	if (argc == 1) {
		std::cerr << "no normal force per area is given!";
		exit(-1);
	}
	if (argc == 2) {
		std::cerr << "no input msh file is given!";
		exit(-1);
	}
    
	TractionRodComputation::TractionRod<3> spacial_traction_rod(atof(argv[1]),argv[2]);
	spacial_traction_rod.run();
	return 0;
}

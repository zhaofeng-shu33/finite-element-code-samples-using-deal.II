#include "post_processing.h"
#include "beam_parameter.h"
/** namespace planar_beam_update contains updated code to solve a planar stress problem.
 *  For more info, see the namespace planar_beam.
 */
namespace planar_beam_update{
/** implementing symmetric constraints.
* reducing computation cost
*/
template <int dim>
class Beam
{
    typedef post_processing::ComputeStressField<dim,Beam<dim>> BeamPostProcessing;    
	friend class BeamPostProcessing;
public:
	Beam();
	void run();
private:
	static const double YoungModulus;
	static const double PoissonRatio;
	static const double shearModulus;
	static const double LamesFirstParameter;
	double neumann_coefficient;
	void get_neumann_value(const Point<dim>   &p, Vector<double>  &values)const;
	void make_grid();
	void read_grid(char* fileName);
	void write_grid(char* fileName);
	void setup_system();
	void setup_system_from_saved_solution();
	void storeGradientToMap();
	void assemble_system();
	void test_boundary();
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
};
template <int dim>
const double Beam<dim>::YoungModulus = 1.0;
template <int dim>
const double Beam<dim>::PoissonRatio = 0.1;
template <int dim>
const double Beam<dim>::shearModulus = YoungModulus / (2 * (1 + PoissonRatio));
template <int dim>
const double Beam<dim>::LamesFirstParameter = YoungModulus*PoissonRatio / ((1 + PoissonRatio)*(1 -PoissonRatio));
template <int dim>
Beam<dim>::Beam() :
	fe(FE_Q<dim>(1), dim),
	dof_handler(triangulation)
{
	neumann_coefficient = beam::force_couple_moment * 2 / pow(beam::height_of_beam,3.0);
}
template <int dim>
void Beam<dim>::test_boundary() {
	typename DoFHandler<dim>::active_cell_iterator
		cell = dof_handler.begin_active(),
		endc = dof_handler.end();
	for (; cell != endc; ++cell) {
		for (unsigned int face_number = 0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
			if (cell->face(face_number)->at_boundary()
				&&
				(cell->face(face_number)->boundary_id() == 1)) {
				std::cout << "one face has boundary id =1\n";
			}
	}
}
template <int dim>
void Beam<dim>::storeGradientToMap() {
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
	std::cout << "cellIndex:" << cell_index << '\n';
	std::cout << "vertices per cell:" << GeometryInfo<dim>::vertices_per_cell << '\n';
}
template <int dim>
void Beam<dim>::get_neumann_value(const Point<dim>   &p, Vector<double>  &values) const {//vector valued function
	std::cout << p[1] << '\n';
	values[1] = 0;
	if (p[0]<0) {// sigma_x=6*a*y
		values[0] = -6 * neumann_coefficient*p[1];		
	}
	else {
		values[0] = 6 * neumann_coefficient*p[1];
	}
}

template <int dim>
void Beam<dim>::read_grid(char* fileName) {
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
void Beam<dim>::write_grid(char* fileName) {
	std::ofstream out(fileName);
	GridOut grid_out;
	grid_out.write_vtk(triangulation, out);

	std::cout << "Grid written to grid-" << fileName << std::endl;
}
template <int dim>
void Beam<dim>::run()
{
	read_grid("D:/cmake/CGAL/build/cgal_solution/Release/beam_triangulation.msh");
	//test_boundary();
	//no need to refine the mesh
	setup_system();
	assemble_system();
	std::cout << "assemble system finished...\n";
	solve();
	std::cout << "solve finished...\n";
/*	setup_system_from_saved_solution();*/
	storeGradientToMap();
	output_results("solution_Beam.vtk");

}
template <int dim>
void Beam<dim>::make_grid()
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
void Beam<dim>::setup_system_from_saved_solution() {
	dof_handler.distribute_dofs(fe);
	std::cout << "Number of degrees of freedom: "
		<< dof_handler.n_dofs()
		<< std::endl;
	std::ifstream file{ "solution_archive.txt" };
	boost::archive::text_iarchive ia(file);
	solution.load(ia, 1);
}
template <int dim>
void Beam<dim>::setup_system()
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
void Beam<dim>::assemble_system()
{
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
	std::map<types::global_dof_index, double> boundary_values;
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
			if (cell->face(face_number)->at_boundary() &&cell->face(face_number)->boundary_id()==1)
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
		for(int vertex_number=0;vertex_number<GeometryInfo<dim>::vertices_per_cell; vertex_number++){
			Point<dim> one_vertex = cell->vertex(vertex_number);
			if (abs(one_vertex[0] - beam::length_of_beam / 2) < 1e-5 && abs(one_vertex[1]) < 1e-5) {
				boundary_values[local_dof_indices[dim*vertex_number+1]]=0;
			}
			else if (abs(one_vertex[0] + beam::length_of_beam / 2) < 1e-5 && abs(one_vertex[1]) < 1e-5) {
				boundary_values[local_dof_indices[dim*vertex_number + 1]]=0;
			}
		}
		for (unsigned int i = 0; i<dofs_per_cell; ++i)
		{
			for (unsigned int j = 0; j<dofs_per_cell; ++j)
				system_matrix.add(local_dof_indices[i],
					local_dof_indices[j],
					cell_matrix(i, j));
			system_rhs(local_dof_indices[i]) += cell_rhs(i);
		}
	}
	FEValuesExtractors::Scalar displacement_1(0);
	ComponentMask displacement_1_mask = fe.component_mask(displacement_1);
	VectorTools::interpolate_boundary_values(dof_handler,
		2,/*boundary_id=2, component_mask=1*/
		ZeroFunction<2>(2),
		boundary_values,displacement_1_mask);
	MatrixTools::apply_boundary_values(boundary_values,
		system_matrix,
		solution,
		system_rhs);/*apply dirichlet bc*/
}
template <int dim>
void Beam<dim>::solve()
{
	//non symetric matrix
	SolverControl           solver_control(4000, 1e-5);
	SolverGMRES<>              solver(solver_control);
	solver.solve(system_matrix, solution, system_rhs,
		PreconditionIdentity());
	std::ofstream file{ "solution_archive.txt" };
	boost::archive::text_oarchive oa(file);
	solution.save(oa, 1);
}
template <int dim>
void Beam<dim>::output_results(char* fileName) const//post processing
{
	DataOut<dim> data_out;
	data_out.attach_dof_handler(dof_handler);
	ComputeStressField<dim> my_stress_field;
	data_out.add_data_vector(solution, my_stress_field);
	data_out.build_patches();
	std::ofstream output(fileName);
	data_out.write_vtk(output);
}
}
int main(int argc)
{
	planar_beam_update::Beam<2> planar_beam;
	planar_beam.run();
	return 0;
}

#define _USE_MATH_DEFINES
#include <math.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/pointer_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/fe/fe_system.h>
#include <fstream>
#include <deal.II/base/mpi.h>
#include <iostream>
using namespace dealii;
/** namespace NeumannBoundarySingularity contains test code to \f$ y''=1 \f$ with known derivative at \f$ y(0)=1,y(1)=-1\f$.
 *  
 */
namespace NeumannBoundarySingularity{
/**
 * function to actually solve mysparse*mySolution=myRHS
 */
void ConstaintMatrixTest(SparseMatrix<double>& mysparse, Vector<double>& myRHS, Vector<double>& mySolution);
/**
 * function to assemble the equation system
 */
void PureNeumannBound() {
	const int MD = 20;
	double h = 2.0 / (MD - 1);
	Vector<double> myRHS(MD);
	Vector<double> mySolution(MD);
	myRHS[0] = -h;
	myRHS[MD - 1] = -h;
	for (int i = 1; i < MD - 1; i++) {
		myRHS[i] = h*h;
	}

	std::vector<std::map<int, double>> vector_for_sparse_matrix;
	std::vector<std::vector<unsigned int> > column_indices(MD);
	for (int j = 0; j < MD; j++) {

		std::map<int, double> tmp_map;
		column_indices[j].push_back(j);
		if (j ==MD-1) {
			tmp_map.insert(std::pair<int, double>(j, 1.0));
			tmp_map.insert(std::pair<int, double>(j-1, -1.0));
			column_indices[j].push_back(j-1);
		}
		else if (j ==0) {
			tmp_map.insert(std::pair<int, double>(j, 1.0));
			tmp_map.insert(std::pair<int, double>(j+1, -1.0));
			column_indices[j].push_back(j+1);
		}
		else {
			tmp_map.insert(std::pair<int, double>(j, 2.0));
			tmp_map.insert(std::pair<int, double>(j-1, -1.0));
			tmp_map.insert(std::pair<int, double>(j+1, -1.0));
			column_indices[j].push_back(j - 1);
			column_indices[j].push_back(j + 1);
		}
		vector_for_sparse_matrix.push_back(tmp_map);
	}
	SparseMatrix<double> A;
	SparsityPattern As;
	As.copy_from(MD, MD , column_indices.begin(), column_indices.end());//create entry
	A.reinit(As);//tri-diagnal;
	A.copy_from(vector_for_sparse_matrix.begin(), vector_for_sparse_matrix.end());//add value
	ConstaintMatrixTest(A, myRHS, mySolution);
}
void ConstaintMatrixTest(SparseMatrix<double>& mysparse,Vector<double>& myRHS,Vector<double>& mySolution) {
	std::cout << "SPARSEMATRIX:\n";
	mysparse.print(std::cout);
	std::cout << "RHS:\n";
	myRHS.print(std::cout);
	ConstraintMatrix     constraints;
	constraints.clear();
	constraints.add_line(0);
	//constraints.add_entry(0, 1, 4);
	constraints.set_inhomogeneity(0, 0);
	constraints.close();
	std::cout << "CONSTRAINTMATRIX:\n";
	constraints.print(std::cout);
	constraints.condense(mysparse,myRHS);
	std::cout << "SPARSEMATRIX after condense:\n";
	mysparse.print(std::cout);
	std::cout << "RHS after condense:\n";
	myRHS.print(std::cout);

	SolverControl           my_solver_control(4000, 1e-5);
	SolverGMRES<>              my_solver(my_solver_control);
	my_solver.solve(mysparse, mySolution, myRHS,
		PreconditionIdentity());
	std::cout << "Solution before distribute:\n";
	mySolution.print(std::cout);
	constraints.distribute(mySolution);
	std::cout << "Solution after distribute:\n";
	mySolution.print(std::cout);
}

}
/** namespace hollowSphereOriginal contains code to solve 3D fem problem based on linear elastic model.
 *  For more info, see <a href="http://www.dealii.org/developer/doxygen/deal.II/step_8.html">step 8 of deal.II tutorial</a> 
 */
namespace hollowSphereOriginal{
template<int dim>
std::map<int, Tensor<1,dim>> my_gradient_map;
std::map<int, int>my_gradient_sum;
template <int dim>
//! learn how to use deal.II tensor algebra
void TensorAlgebraTest() {
	double aaa[dim][dim] = { {1,2,3},{4,5,6},{7,8,9} };
	Tensor<2, dim> myTensor(aaa);
	for (int i = 0; i < myTensor.dimension; i++)
		std::cout << myTensor[i] << std::endl;
	std::cout << "Tensor transpose:\n" << transpose(myTensor);
	std::cout << "Tensor mean value:\n" << (myTensor+transpose(myTensor)) / 2;
}
//! user provided class to deal with scalar type DataPostprocessor
template <int dim>
class ComputeRadiusDisplacement : public DataPostprocessorScalar<dim>
{
public:
	ComputeRadiusDisplacement();
	virtual
		void
		evaluate_vector_field
		(const DataPostprocessorInputs::Vector<dim> &inputs,
			std::vector<Vector<double> >               &computed_quantities) const;
};
template <int dim>
ComputeRadiusDisplacement<dim>::ComputeRadiusDisplacement()
	:
	DataPostprocessorScalar<dim>("RadiusDisplacement",
		update_values| update_quadrature_points)
{}
template <int dim>
void
ComputeRadiusDisplacement<dim>::evaluate_vector_field
(const DataPostprocessorInputs::Vector<dim> &inputs,
	std::vector<Vector<double> >               &computed_quantities) const
{
	Assert(computed_quantities.size() == inputs.solution_values.size(),
		ExcDimensionMismatch(computed_quantities.size(), inputs.solution_values.size()));
	for (unsigned int i = 0; i<computed_quantities.size(); i++)
	{
		Assert(computed_quantities[i].size() == 1,
			ExcDimensionMismatch(computed_quantities[i].size(), 1));
		Assert(inputs.solution_values[i].size() == dim,
			ExcDimensionMismatch(inputs.solution_values[i].size(), dim));
		computed_quantities[i](0) = 0;
		for (int j = 0; j < dim; j++)
			computed_quantities[i](0) += inputs.solution_values[i](j)*inputs.evaluation_points[i](j);
		computed_quantities[i](0) /= inputs.evaluation_points[i].norm();

	}
}
//! user provided class to deal with vector type DataPostprocessor
template <int dim>
class ComputeStressField : public DataPostprocessorVector<dim>//compute stress field along the radius outwards
{
public:
	ComputeStressField();
	virtual
		void
		evaluate_vector_field
		(const DataPostprocessorInputs::Vector<dim> &inputs,
			std::vector<Vector<double> >               &computed_quantities) const;
};
template <int dim>
ComputeStressField<dim>::ComputeStressField()
	:
	DataPostprocessorVector<dim>("StressField",
		update_values)
{}
template <int dim>
void
ComputeStressField<dim>::evaluate_vector_field
(const DataPostprocessorInputs::Vector<dim> &inputs,
	std::vector<Vector<double> >               &computed_quantities) const
{
	Assert(computed_quantities.size() == inputs.solution_values.size(),
		ExcDimensionMismatch(computed_quantities.size(), inputs.solution_values.size()));
	std::vector<types::global_dof_index> local_dof_indices(inputs.solution_values.size()*dim);
	auto current_cell = inputs.get_cell<DoFHandler<dim>>();
	current_cell->get_dof_indices(local_dof_indices);
	for (unsigned int i = 0; i<computed_quantities.size(); i++)
	{
		Assert(computed_quantities[i].size() == dim,
			ExcDimensionMismatch(computed_quantities[i].size(), dim));
		Assert(inputs.solution_values[i].size() == dim,
			ExcDimensionMismatch(inputs.solution_values[i].size(), dim));
		//construct a rank 2 strain tensor from C-style array
		Tensor<2, dim> strainTensor, identityTensor, stressTensor;
		for (int j = 0; j < dim; j++) {
			strainTensor[j] = my_gradient_map<dim>[local_dof_indices[3*i+j]] / my_gradient_sum[local_dof_indices[3 * i + j]];//inputs.solution_gradients[i][j];
			identityTensor[j][j] = 1;
		}
		double divergent_value = trace(strainTensor);
		strainTensor = (strainTensor + transpose(strainTensor)) / 2;
		stressTensor = (hollowSphere<dim>::LamesFirstParameter * divergent_value)*identityTensor + (2 * hollowSphere<dim>::shearModulus)*strainTensor;
		//check whether the stressTensor is symmetric
		Assert(stressTensor == transpose(stressTensor),
			ExcInternalError());
		Tensor<1,dim> stress_along_radius= stressTensor*(inputs.evaluation_points[i] / inputs.evaluation_points[i].norm());
		for (int j = 0; j < dim; j++)
			computed_quantities[i][j] = stress_along_radius[j];
	}
}
template <int dim>
class RightHandSide : public Function<dim>
{
public:
	RightHandSide() : Function<dim>() {}
	virtual void vector_value(const Point<dim>   &p,
		Vector<double>  &values) const;
};
template <int dim>
void RightHandSide<dim>::vector_value(const Point<dim>   &p,
	Vector<double>  &values) const {//vector valued function
	values.reinit(dim);
	for (int i = 0; i < dim; i++)
		values[i] = 0;
}

//! Solver class
template <int dim>
class hollowSphere
{
	friend class ComputeStressField<dim>;
public:
	hollowSphere();
/** if userDefined equals true, read custom grid file from disk,
*   otherwise generate mesh with deal.II built-in functionality.
*/
	void run(bool userDefined);
private:
	static const double epsilon;
	static const double inner_radius;
	static const double outer_radius;
	static const double outer_pressure;
	static const double inner_pressure;
	static const double YoungModulus;
	static const double PoissonRatio;
	static const double shearModulus;
	static const double LamesFirstParameter;
	void get_neumann_value(const Point<dim>   &p,Vector<double>  &values)const;
	void make_grid();
	void read_grid(char* fileName);
	void write_grid(char* fileName);
	void setup_system();
	void setup_system_from_saved_solution();
//! compute stess field at node    
	void storeGradientToMap();
	void assemble_system(bool userDefined);
	void solve();
	void output_results(char* fileName) const;
	Triangulation<dim>     triangulation;
	FESystem<dim>               fe;
	DoFHandler<dim>        dof_handler;
	SparsityPattern      sparsity_pattern;
	SparseMatrix<double> system_matrix;
	ConstraintMatrix     constraints;
	Vector<double>       solution;
	Vector<double>       system_rhs;
};
template <int dim>
const double hollowSphere<dim>::epsilon = 0.1;
template <int dim>
const double hollowSphere<dim>:: inner_radius = 1.0;
template <int dim>
const double hollowSphere<dim>:: outer_radius = 2.0;
template <int dim>
const double hollowSphere<dim>:: outer_pressure = 0.5;
template <int dim>
const double hollowSphere<dim>:: inner_pressure = 0.6;
template <int dim>
const double hollowSphere<dim>:: YoungModulus = 1.0;
template <int dim>
const double hollowSphere<dim>:: PoissonRatio = 0.1;
template <int dim>
const double hollowSphere<dim>:: shearModulus = YoungModulus / (2 * (1 + PoissonRatio));
template <int dim>
const double hollowSphere<dim>:: LamesFirstParameter = YoungModulus*PoissonRatio / ((1 + PoissonRatio)*(1 - 2 * PoissonRatio));
template <int dim>
hollowSphere<dim>::hollowSphere() :
	fe(FE_Q<dim>(1), dim),
	dof_handler(triangulation)
{}
template <int dim>
void hollowSphere<dim>::storeGradientToMap() {
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
			for(int k=0;k<dim;k++){
			if (my_gradient_map<dim>.find(local_dof_indices[dim*i+k]) == my_gradient_map<dim>.end()) {//not found
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
void hollowSphere<dim>::get_neumann_value(const Point<dim>   &p,Vector<double>  &values) const {//vector valued function
	if (abs(p.norm() - inner_radius) < abs(p.norm() - outer_radius)) {// p at inner boundary
		for (int i = 0; i<dim; i++)
			values[i] = inner_pressure*p[i] / p.norm();
		return;
	}
	else {
		for (int i = 0; i<dim; i++)
			values[i] = -outer_pressure*p[i] / p.norm();
		return;
	}
	/*else{
	Assert(false, ExcNotImplemented());
	return;
	}*/
}
template <int dim>
void hollowSphere<dim>::read_grid(char* fileName) {
	GridIn<dim> grid_in;
	grid_in.attach_triangulation(triangulation);
	std::ifstream input_file(fileName);
	grid_in.read_msh(input_file);
	triangulation.refine_global(2);
	std::cout << "Number of active cells: "
		<< triangulation.n_active_cells()
		<< std::endl;
}
template <int dim>
void hollowSphere<dim>::write_grid(char* fileName) {
	std::ofstream out(fileName);
	GridOut grid_out;
	grid_out.write_vtk(triangulation, out);

	std::cout << "Grid written to grid-" << fileName << std::endl;
}
template <int dim>
void hollowSphere<dim>::run(bool userDefined)
{
	//make_grid();
	if (userDefined)
		read_grid("D:/gmesh/tutorial/t1.msh");
	else
		make_grid();
	//write_grid("t1.vtk");
	setup_system();
	assemble_system(userDefined);
	std::cout << "assemble system finished...\n";
	solve();
	std::cout << "solve finished...\n";
	//setup_system_from_saved_solution();
	storeGradientToMap();
	if (userDefined)
		output_results("solution_hollowSphere_userDefined.vtk");
	else
		output_results("solution_hollowSphere.vtk");
	
}
template <int dim>
void hollowSphere<dim>::make_grid()
{
	GridGenerator::hyper_shell(triangulation,Point<dim>(), inner_radius,outer_radius);
	static const SphericalManifold<dim> boundary;
	triangulation.set_all_manifold_ids_on_boundary(0);
	triangulation.set_manifold(0, boundary);
	triangulation.refine_global(2);
	std::cout << "Number of active cells: "
		<< triangulation.n_active_cells()
		<< std::endl;
}template <int dim>
void hollowSphere<dim>::setup_system_from_saved_solution() {
	dof_handler.distribute_dofs(fe);
	std::cout << "Number of degrees of freedom: "
		<< dof_handler.n_dofs()
		<< std::endl;
	std::ifstream file{ "solution_archive.txt" };
	boost::archive::text_iarchive ia(file);
	solution.load(ia, 1);
}
template <int dim>
void hollowSphere<dim>::setup_system()
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
void hollowSphere<dim>::assemble_system(bool userDefined)
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
	for (; cell != endc; ++cell)
	{
		cell_matrix = 0;
		cell_rhs = 0;
		fe_values.reinit(cell);
		right_hand_side.vector_value_list(fe_values.get_quadrature_points(), rhs_values);
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
		for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
			if (cell->face(face_number)->at_boundary())
			{
			/*	for(unsigned int vertex_number=0;vertex_number<GeometryInfo<dim>::vertices_per_face;vertex_number++){
					std::cout<<cell->face(face_number)->vertex(vertex_number)<<'*';
				}
				std::cout << '\n';*/
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
		for (unsigned int i = 0; i<dofs_per_cell; ++i)
		{
			for (unsigned int j = 0; j<dofs_per_cell; ++j)
				system_matrix.add(local_dof_indices[i],
					local_dof_indices[j],
					cell_matrix(i, j));
			system_rhs(local_dof_indices[i]) += cell_rhs(i);
		}
	}
/*	constraints.clear();
	constraints.add_line(0);
	constraints.set_inhomogeneity(0, 0);
	constraints.close();//the first vertex is fixed
	constraints.condense(system_matrix, system_rhs);
	*/
}
template <int dim>
void hollowSphere<dim>::solve()
{
	//non symetric matrix
	SolverControl           solver_control(4000, 1e-5);
	SolverGMRES<>              solver(solver_control);
	solver.solve(system_matrix, solution, system_rhs,
		PreconditionIdentity());
	std::ofstream file{ "solution_archive.txt" };
	boost::archive::text_oarchive oa(file);
	solution.save(oa, 1);
	//constraints.distribute(solution);
}
template <int dim>
void hollowSphere<dim>::output_results(char* fileName) const//post processing
{
	DataOut<dim> data_out;
	data_out.attach_dof_handler(dof_handler);
	std::vector<std::string> solution_names;
	switch (dim)
	{
	case 1:
		solution_names.push_back("displacement");
		break;
	case 2:
		solution_names.push_back("x_displacement");
		solution_names.push_back("y_displacement");
		break;
	case 3:
		solution_names.push_back("x_displacement");
		solution_names.push_back("y_displacement");
		solution_names.push_back("z_displacement");
		break;
	default:
		Assert(false, ExcNotImplemented());
	}
	ComputeRadiusDisplacement<dim> my_radius_displacement;
	ComputeStressField<dim> my_stress_field;
	data_out.add_data_vector(solution, solution_names);
	data_out.add_data_vector(solution, my_radius_displacement);
	data_out.add_data_vector(solution, my_stress_field);
	data_out.build_patches();
	std::ofstream output(fileName);
	data_out.write_vtk(output);
}
}
int main(int argc)
{
	hollowSphereOriginal::hollowSphere<3> myhollowSphere;
	if (argc>1)
		myhollowSphere.run(true);//use user-defined input mesh file
	else
		myhollowSphere.run(false);//use grid-generator to produce a hollowSphere and refine it
	
	return 0;
}

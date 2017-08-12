#define _USE_MATH_DEFINES
#include <math.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
using namespace dealii;
/** namespace practice5 contains the class of my coursework of course 
 *  PDE numerical solution 
 *  see the problem description: <a href="../practice5.pdf">practice5.pdf</a>
 */
namespace practice5 {
/**
 * problem const
 */
const double epsilon = 0.1;
/**
 * right hand side of pde: \f$\sin(\pi x)\sin(\pi y)\f$
 */
class RightHandSide : public Function<2>
{
public:
	RightHandSide() : Function<2>() {}//!< constructor
	virtual double value(const Point<2>   &p,
		const unsigned int  component = 0) const;//!< evaluate at a two dimentional point
};
double RightHandSide::value(const Point<2>   &p,
	const unsigned int) const {
	return std::sin(M_PI*p[0])*std::sin(M_PI*p[1]);
}
/**
 * Practice5 Solver
 * \author zhao feng
 */
class Practice5
{
public:
	Practice5();//!< constructor
	void run();//!< solver routine with finite difference method
private:
	void make_grid();//!< generate the grid
	void setup_system();//!< initialization
/**
 * if isFiniteDifference is true, use finite difference method to assemble the system;
 * otherwise use finite element method
 */    
	void assemble_system(bool isFiniteDifference);
	void solve();//!< solve non symetric matrix system equation
	void output_results() const;//!< output the solution in vtk format
	Triangulation<2>     triangulation;
	FE_Q<2>              fe;
	DoFHandler<2>        dof_handler;
	SparsityPattern      sparsity_pattern;
	SparseMatrix<double> system_matrix;
	Vector<double>       solution;
	Vector<double>       system_rhs;
	int refinementLevel=5;//!< grid refinement level
};
Practice5::Practice5() :
	fe(1),
	dof_handler(triangulation) 
{}
void Practice5::run()
{
	make_grid();
	setup_system();
	assemble_system(true);
	solve();
	output_results();
}
void Practice5::make_grid()
{
	GridGenerator::hyper_cube(triangulation,0, 1);
	triangulation.refine_global(refinementLevel);
	std::cout << "Number of active cells: "
		<< triangulation.n_active_cells()
		<< std::endl;
}
void Practice5::setup_system()
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
void Practice5::assemble_system(bool isFiniteDifference)
{
	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	FullMatrix<double>   cell_matrix(dofs_per_cell, dofs_per_cell);
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	DoFHandler<2>::active_cell_iterator cell = dof_handler.begin_active();
	DoFHandler<2>::active_cell_iterator endc = dof_handler.end();
	const RightHandSide right_hand_side;

	if (isFiniteDifference) {
		double h = 1 / pow(2.0,refinementLevel);
		double sigma = h*std::cosh(h / (2 * epsilon)) / (2 * std::sinh(h / (2 * epsilon))*epsilon);
		for (; cell != endc; ++cell) {
			cell_matrix = 0;
			cell_matrix(0, 0) = sigma*epsilon / (h*h);
			cell_matrix(0, 1) = -sigma*epsilon / (2*h*h)+1/(4*h);
			cell_matrix(0, 2) = -sigma*epsilon / (2 * h*h);
			cell_matrix(1, 0) = -1/(4*h) - sigma*epsilon / (2 * h*h);
			cell_matrix(1, 1) = sigma*epsilon / (h*h);
			cell_matrix(1, 3) = -sigma*epsilon / (2 * h*h);
			cell_matrix(2, 0) = -sigma*epsilon / (2 * h*h);
			cell_matrix(2, 2) = sigma*epsilon / (h*h);
			cell_matrix(2, 3) = -sigma*epsilon / (2 * h*h) + 1 / (4 * h);
			cell_matrix(3, 1) = -sigma*epsilon / (2 * h*h);
			cell_matrix(3, 2) = -sigma*epsilon / (2 * h*h) - 1 / (4 * h);
			cell_matrix(3, 3) = sigma*epsilon / (h*h);
			cell->get_dof_indices(local_dof_indices);
			for (unsigned int i = 0; i<dofs_per_cell; ++i)
				for (unsigned int j = 0; j<dofs_per_cell; ++j)
					system_matrix.add(local_dof_indices[i],
						local_dof_indices[j],
						cell_matrix(i, j));
			for (unsigned int i = 0; i<dofs_per_cell; ++i)
				system_rhs(local_dof_indices[i]) += right_hand_side.value(cell->vertex(i))/4;
		}
	}
	else{
	Vector<double>       cell_rhs(dofs_per_cell);
	QGauss<2>  quadrature_formula(2);
	FEValues<2> fe_values(fe, quadrature_formula,
		update_values | update_gradients | update_JxW_values | update_quadrature_points);
	const unsigned int   n_q_points = quadrature_formula.size();
	for (; cell != endc; ++cell)
	{
		fe_values.reinit(cell);
		cell_matrix = 0;
		cell_rhs = 0;
		for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
		{
			for (unsigned int i = 0; i < dofs_per_cell; ++i) {
				for (unsigned int j = 0; j < dofs_per_cell; ++j) {
					cell_matrix(i, j) += epsilon*(fe_values.shape_grad(i, q_index) *
						fe_values.shape_grad(j, q_index) *
						fe_values.JxW(q_index));
					cell_matrix(i, j) += (fe_values.shape_grad(j, q_index)[0] *
						fe_values.shape_value(i, q_index) *
						fe_values.JxW(q_index));
				}
			}
			for (unsigned int i = 0; i<dofs_per_cell; ++i)
				cell_rhs(i) += (fe_values.shape_value(i, q_index) *
					right_hand_side.value(fe_values.quadrature_point(q_index))  *
					fe_values.JxW(q_index));
		}
		cell->get_dof_indices(local_dof_indices);
		for (unsigned int i = 0; i<dofs_per_cell; ++i)
			for (unsigned int j = 0; j<dofs_per_cell; ++j)
				system_matrix.add(local_dof_indices[i],
					local_dof_indices[j],
					cell_matrix(i, j));
		for (unsigned int i = 0; i<dofs_per_cell; ++i)
			system_rhs(local_dof_indices[i]) += cell_rhs(i);
	}
	}
	std::map<types::global_dof_index, double> boundary_values;
	VectorTools::interpolate_boundary_values(dof_handler,
		0,
		ZeroFunction<2>(),
		boundary_values);
	MatrixTools::apply_boundary_values(boundary_values,
		system_matrix,
		solution,
		system_rhs);
	if (isFiniteDifference) {
		std::ofstream out("system_matrix.txt");
		system_matrix.print(out);
		out.close();
	}
}
void Practice5::solve()
{
	
	SolverControl           solver_control(4000, 1e-5);
	SolverGMRES<>              solver(solver_control);
	solver.solve(system_matrix, solution, system_rhs,
		PreconditionIdentity());
}
void Practice5::output_results() const
{
	DataOut<2> data_out;
	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(solution, "solution");
	data_out.build_patches();
	std::ofstream output("solution_practice_new_5.vtk");
	data_out.write_vtk(output);
}
}
int main()
{
    using namespace practice5;
	Practice5 myPractice5;
	myPractice5.run();
	return 0;
}
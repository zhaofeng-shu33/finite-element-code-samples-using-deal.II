#pragma once
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_enriched.h>
#include <deal.II/base/polynomials_p.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/config.h>
#include <deal.II/fe/fe.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/hp/dof_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/sparse_matrix.h>

/** namespace fe_enriched_test contains code to verify that with knowledge of analytical behaviour of solution, FEM accuracy can be improved
* by using shape functions similar to the solution component.
*
* deal.II implementing FE_Enriched finite element to accomplish this.
*/
namespace fe_enriched_test{
using namespace dealii;
/** Solution function is \f$ f(x,y)=(3x+4y+2xy+1)exp(-\sqrt{x^2+y^2}) \f$,
* and we project it on the FEM space and compute the error.
*/
template <int dim>
class SolutionFunction : public Function<dim>
{
public:
	SolutionFunction()
		: Function<dim>(1)
	{}

	virtual double value(const Point<dim> &point,
		const unsigned int component = 0) const
	{
		return (point[0]*3+point[1]*4+2*point[0]*point[1]+1)*exp(-1*point.norm());
	}

	virtual Tensor<1, dim> gradient(const Point<dim> &point,
		const unsigned int component = 0) const
	{
		Tensor<1, dim> res = point;
		Assert(point.norm() > 0,
			ExcMessage("gradient is not defined at zero"));
		res[0] = (3 + 2 * point[1]-point[0]*(point[0] * 3 + point[1] * 4 + 2 * point[0] * point[1] + 1)/point.norm())*exp(-1 * point.norm());
		res[1] = (4 + 2 * point[0]- (point[0] * 3 + point[1] * 4 + 2 * point[0] * point[1] + 1)*point[1]/point.norm())*exp(-1 * point.norm());
		if (dim == 3)res[2] = 0;
		return res;
	}
};
/** XFEM requires to use special shape functions near crack tip to simulate the singularity of the solution.
*
* This class is not accomplished yet.
*/
class NearTipEnrichmentFunction : public Function<2>
{
public:
	NearTipEnrichmentFunction()
		: Function<2>(1)
	{}

	virtual double value(const Point<2> &point,
		const unsigned int component = 0) const
	{
		//sin(theta/2)
		double r = point.norm();
		double sin_theta_half = sqrt((1 - point[0] / r) / 2);
		return sqrt(r)*sin_theta_half;
	}

	virtual Tensor<1, 2> gradient(const Point<2> &point,
		const unsigned int component = 0) const
	{
		Tensor<1, 2> res = point;
		Assert(point.norm() > 0,
			dealii::ExcMessage("gradient is not defined at zero"));
		res[0] *= -value(point);
		return res;
	}
};

template <int dim>
void test_enriched()
{
	std::cout << "enriched finite element\n";
	SparseMatrix<double>                    system_matrix;
	Vector<double>                          solution;
	Vector<double>                          system_rhs;
	SparsityPattern                         sparsity_pattern;



	SolutionFunction<dim> mySolution;

	Triangulation<dim> triangulation;
	hp::DoFHandler<dim> dof_handler(triangulation);
	NearTipEnrichmentFunction<dim> function;
	hp::FECollection<dim> fe_collection;
	fe_collection.push_back(FE_Enriched<dim>(FE_Q<dim>(1)));
	fe_collection.push_back(FE_Enriched<dim>(FE_Q<dim>(1),
		FE_Q<dim>(1),
		&function));
	GridGenerator::hyper_cube(triangulation);
	dof_handler.begin_active()->set_active_fe_index(1);
	dof_handler.distribute_dofs(fe);


	DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp);
	sparsity_pattern.copy_from(dsp);
	system_matrix.reinit(sparsity_pattern);
	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());



	QGauss<dim> quadrature(4);
	FEValues<dim> fe_values(fe, quadrature,
		update_values | update_quadrature_points |
		update_JxW_values);

	FullMatrix<double>   cell_matrix;
	Vector<double>       cell_rhs;
	std::vector<types::global_dof_index> local_dof_indices;

	typename DoFHandler<dim>::active_cell_iterator
		cell = dof_handler.begin_active(),
		endc = dof_handler.end();
	for (; cell != endc; ++cell) {
		fe_values.reinit(cell);
		std::cout << cell->active_cell_index() << ' ';
		const unsigned int n_q_points = quadrature.size();
		const unsigned int dofs_per_cell = fe.dofs_per_cell;
		const std::vector<Point<dim> > q_points = fe_values.get_quadrature_points();
		cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
		cell_matrix = 0;
		cell_rhs.reinit(dofs_per_cell);
		cell_rhs = 0;
		for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
			for (unsigned int i = 0; i < dofs_per_cell; ++i) {
			//	std::cout << "at point " << q_points[q_point] << "shape function id "<<i<<" value is " << fe_values.shape_value(i, q_point) << std::endl;
				for (unsigned int j = 0; j < dofs_per_cell; ++j) {
					cell_matrix(i, j) += fe_values.shape_value(i, q_point)* fe_values.shape_value(j, q_point)*fe_values.JxW(q_point);// 
				}
				cell_rhs(i) += fe_values.shape_value(i, q_point)*mySolution.value(q_points[q_point])*fe_values.JxW(q_point);
			}
		}
		local_dof_indices.resize(dofs_per_cell);
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
	std::cout << '\n';
	std::cout << "System matrix: \n";
	system_matrix.print(std::cout, false, false);
	std::cout << "RHS:\n" << std::endl;
	system_rhs.print();
	SolverControl           solver_control(system_rhs.size(),
		1e-9);
	SolverCG<>              cg(solver_control);
	PreconditionSSOR<> preconditioner;
	preconditioner.initialize(system_matrix, 1.2);
	cg.solve(system_matrix, solution, system_rhs,
		preconditioner);
	std::cout << "Solution:\n" << solution << std::endl;


	Vector<float> difference_per_cell(triangulation.n_active_cells());
	VectorTools::integrate_difference(dof_handler,
		solution,
		SolutionFunction<dim>(),
		difference_per_cell,
		QGauss<dim>(3),
		VectorTools::L2_norm);
	const double L2_error = VectorTools::compute_global_error(triangulation,
		difference_per_cell,
		VectorTools::L2_norm);
	VectorTools::integrate_difference(dof_handler,
		solution,
		SolutionFunction<dim>(),
		difference_per_cell,
		QGauss<dim>(3),
		VectorTools::H1_seminorm);
	const double H1_error = VectorTools::compute_global_error(triangulation,
		difference_per_cell,
		VectorTools::H1_seminorm);
	const QTrapez<1>     q_trapez;
	const QIterated<dim> q_iterated(q_trapez, 5);
	VectorTools::integrate_difference(dof_handler,
		solution,
		SolutionFunction<dim>(),
		difference_per_cell,
		q_iterated,
		VectorTools::Linfty_norm);
	const double Linfty_error = VectorTools::compute_global_error(triangulation,
		difference_per_cell,
		VectorTools::Linfty_norm);

	const unsigned int n_dofs = dof_handler.n_dofs();

	std::cout << std::endl
		<< "   Number of degrees of freedom: "
		<< n_dofs
		<< std::endl
		<< "L2_error: " << L2_error << std::endl
		<< "H1_error: " << H1_error << std::endl
		<< "Linfty_error: " << Linfty_error << std::endl;
}
template <int dim>
void test_normal()
{
	//FE_Fracture<2, 2> my_fe_fracture;
	//std::cout << my_fe_fracture.get_name()<<std::endl;
	std::cout << "normal finite element"<<std::endl;
	SparseMatrix<double>                    system_matrix;
	Vector<double>                          solution;
	Vector<double>                          system_rhs;
	SparsityPattern                         sparsity_pattern;



	SolutionFunction<dim> mySolution;

	Triangulation<dim> triangulation;
	DoFHandler<dim> dof_handler(triangulation);
	FE_Q<dim> fe(1);//linear in 2 dimension
	GridGenerator::hyper_cube(triangulation);
	triangulation.refine_global(1);
	dof_handler.distribute_dofs(fe);


	DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp);
	sparsity_pattern.copy_from(dsp);
	system_matrix.reinit(sparsity_pattern);
	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());



	QGauss<dim> quadrature(2);
	FEValues<dim> fe_values(fe, quadrature,
		update_values| update_quadrature_points|
		update_JxW_values);

	FullMatrix<double>   cell_matrix;
	Vector<double>       cell_rhs;
	std::vector<types::global_dof_index> local_dof_indices;

	typename DoFHandler<dim>::active_cell_iterator
		cell = dof_handler.begin_active(),
		endc = dof_handler.end();
	for (; cell != endc; ++cell) {
		fe_values.reinit(cell);
		std::cout<<cell->active_cell_index()<<' ';
		const unsigned int n_q_points = quadrature.size();
		const unsigned int dofs_per_cell = fe.dofs_per_cell;
		const std::vector<Point<dim> > q_points = fe_values.get_quadrature_points();
		cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
		cell_matrix = 0;
		cell_rhs.reinit(dofs_per_cell);
		cell_rhs = 0;
		for (unsigned int q_point = 0; q_point < n_q_points; ++q_point){
			for (unsigned int i = 0; i < dofs_per_cell; ++i){
				for (unsigned int j = 0; j < dofs_per_cell; ++j) {
					cell_matrix(i, j) += fe_values.shape_value(i, q_point)* fe_values.shape_value(j, q_point)*fe_values.JxW(q_point);// 
				}
				cell_rhs(i) += fe_values.shape_value(i,q_point)*mySolution.value(q_points[q_point])*fe_values.JxW(q_point);
			}
		}
		local_dof_indices.resize(dofs_per_cell);
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
	std::cout << '\n';

	SolverControl           solver_control(system_rhs.size(),
		1e-12);
	SolverCG<>              cg(solver_control);
	PreconditionSSOR<> preconditioner;
	preconditioner.initialize(system_matrix, 1.2);
	cg.solve(system_matrix, solution, system_rhs,
		preconditioner);
	std::cout <<"Solution:\n"<< solution << std::endl;


	Vector<float> difference_per_cell(triangulation.n_active_cells());
	VectorTools::integrate_difference(dof_handler,
		solution,
		SolutionFunction<dim>(),
		difference_per_cell,
		QGauss<dim>(3),
		VectorTools::L2_norm);
	const double L2_error = VectorTools::compute_global_error(triangulation,
		difference_per_cell,
		VectorTools::L2_norm);
	VectorTools::integrate_difference(dof_handler,
		solution,
		SolutionFunction<dim>(),
		difference_per_cell,
		QGauss<dim>(3),
		VectorTools::H1_seminorm);
	const double H1_error = VectorTools::compute_global_error(triangulation,
		difference_per_cell,
		VectorTools::H1_seminorm);
	const QTrapez<1>     q_trapez;
	const QIterated<dim> q_iterated(q_trapez, 5);
	VectorTools::integrate_difference(dof_handler,
		solution,
		SolutionFunction<dim>(),
		difference_per_cell,
		q_iterated,
		VectorTools::Linfty_norm);
	const double Linfty_error = VectorTools::compute_global_error(triangulation,
		difference_per_cell,
		VectorTools::Linfty_norm);

	const unsigned int n_dofs = dof_handler.n_dofs();

	std::cout << std::endl
		<< "   Number of degrees of freedom: "
		<< n_dofs
		<< std::endl
		<< "L2_error: " << L2_error << std::endl
		<< "H1_error: " << H1_error << std::endl
		<< "Linfty_error: " << Linfty_error << std::endl;

}
}
int main() {
	deallog << std::setprecision(4);
	deallog << std::fixed;
	deallog.attach(std::cout);
	deallog.depth_console(0);
	deallog.threshold_double(1.e-10);


	try
	{

		test_enriched<2>();
		test_normal<2>();

	}
	catch (std::exception &exc)
	{
		std::cerr << std::endl << std::endl
			<< "----------------------------------------------------"
			<< std::endl;
		std::cerr << "Exception on processing: " << std::endl
			<< exc.what() << std::endl
			<< "Aborting!" << std::endl
			<< "----------------------------------------------------"
			<< std::endl;

		return 1;
	}
	catch (...)
	{
		std::cerr << std::endl << std::endl
			<< "----------------------------------------------------"
			<< std::endl;
		std::cerr << "Unknown exception!" << std::endl
			<< "Aborting!" << std::endl
			<< "----------------------------------------------------"
			<< std::endl;
		return 1;
	};
	return 0;
}
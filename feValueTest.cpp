#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/base/quadrature_lib.h>
#include <iostream>
using namespace dealii;
/** namespace feValuetest contains test code to learn polynomial shape function supported on stardard cube.
 *  
 */
namespace feValueTest {
const int dim = 2;
const int quadrature_formula_size = 2;
const int polynomial_degree = 2;
}
void main()
{
	Triangulation<dim>   triangulation;
	FE_Q<dim>            fe(polynomial_degree);
	GridGenerator::hyper_cube(triangulation);
	DoFHandler<dim>      dof_handler(triangulation);
	dof_handler.distribute_dofs(fe);

	QGauss<dim>  quadrature_formula(quadrature_formula_size);
	QGauss<1>  quadrature_formula_1d(quadrature_formula_size);
	std::cout << quadrature_formula.size() << ' ' << quadrature_formula_1d.size() << std::endl;
	for (int i = 0; i < quadrature_formula_size; i++) {
		std::cout << "At (" << quadrature_formula_1d.point(i) << "),weight is " << quadrature_formula_1d.weight(i)<<std::endl;
	}
	FEValues<dim> fe_values(fe, quadrature_formula,
		update_values | update_gradients |
		update_quadrature_points | update_JxW_values);
	unsigned int   n_q_points = quadrature_formula.size();
	unsigned int   dofs_per_cell = fe.dofs_per_cell;
	DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
	fe_values.reinit(cell);
	for(int t=0;t<dofs_per_cell;t++){
	for (int i = 0; i < n_q_points; i++) {
		std::cout << "function Number:" << t << " At (" << fe_values.quadrature_point(i) << "),value:" << fe_values.shape_value(t, i);
			std::cout<<" weight: "<<fe_values.JxW(i)<<std::endl;
	}
	}
	//std::cout << n_q_points << std::endl;
}
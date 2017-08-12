#pragma once
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
/** namespace post_processing contains code to post processing
 *  fem computation result.
 *  displacement field and stress tensor field at each degree of freedom are calculated
 */
namespace post_processing{
template<int dim>
std::map<int, Tensor<1, dim>> my_gradient_map;
std::map<int, int>my_gradient_sum;
/** compute displacement and stress field
*/
template <int dim, class T>
class ComputeStressField : public DataPostprocessor<dim>//compute stress tensor
{
public:
	ComputeStressField();
	virtual
		void
		evaluate_vector_field
		(const DataPostprocessorInputs::Vector<dim> &inputs,
			std::vector<Vector<double> >               &computed_quantities) const;
	virtual std::vector<std::string> get_names() const;
	virtual
		std::vector<DataComponentInterpretation::DataComponentInterpretation>
		get_data_component_interpretation() const;
	virtual UpdateFlags get_needed_update_flags() const;
};
template <int dim,class T>
UpdateFlags
ComputeStressField<dim,T>::get_needed_update_flags() const
{
	return update_values;
}
template <int dim,class T>
std::vector<std::string>
ComputeStressField<dim,T>::get_names() const
{
	std::vector<std::string> solution_names(dim, "displacement");
	for(int i=0;i<dim;i++)
		for (int j = 0; j < dim; j++) {
			char newName[20];
			sprintf_s(newName, "tensor_component_%d", i);
			solution_names.push_back(newName);
		}
	return solution_names;
}
template <int dim,class T>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
ComputeStressField<dim,T>::get_data_component_interpretation() const
{
	std::vector<DataComponentInterpretation::DataComponentInterpretation>
		interpretation((dim+1)*dim,
			DataComponentInterpretation::component_is_part_of_vector);
	return interpretation;
}
template <int dim,class T>
ComputeStressField<dim,T>::ComputeStressField()
{}
template <int dim,class T>
void
ComputeStressField<dim,T>::evaluate_vector_field
(const DataPostprocessorInputs::Vector<dim> &inputs,
	std::vector<Vector<double> >               &computed_quantities) const
{
	//for debugging purpose
	Assert(computed_quantities.size() == inputs.solution_values.size(),/*=number of vertex per cell*/
		ExcDimensionMismatch(computed_quantities.size(), inputs.solution_values.size()));
	std::vector<types::global_dof_index> local_dof_indices(inputs.solution_values.size()*dim);
	auto current_cell = inputs.get_cell<DoFHandler<dim>>();
	current_cell->get_dof_indices(local_dof_indices);
	for (unsigned int i = 0; i<computed_quantities.size(); i++)
	{
		for (int j = 0; j < dim; j++)
			computed_quantities[i][j] = inputs.solution_values[i][j];
		//construct a rank 2 strain tensor from C-style array
		Tensor<2, dim> strainTensor, identityTensor, stressTensor;//rank=2
		for (int j = 0; j < dim; j++) {
			strainTensor[j] = my_gradient_map<dim>[local_dof_indices[dim * i + j]] / my_gradient_sum[local_dof_indices[dim * i + j]];//inputs.solution_gradients[i][j];
			identityTensor[j][j] = 1;
		}
		double divergent_value = trace(strainTensor);
		strainTensor = (strainTensor + transpose(strainTensor)) / 2;
		stressTensor = (T::LamesFirstParameter * divergent_value)*identityTensor + (2 * T::shearModulus)*strainTensor;
		//check whether the stressTensor is symmetric
		Assert(stressTensor == transpose(stressTensor),
			ExcInternalError());
		for (int j = dim; j < (dim+1)*dim; j++)
			computed_quantities[i][j] = stressTensor[(j-dim)/dim][j%dim];
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

}
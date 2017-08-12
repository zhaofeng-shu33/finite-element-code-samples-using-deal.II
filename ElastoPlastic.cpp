#include <math.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <fstream>
#include <iostream>
#include <strstream>
#include <vector>
/** Solve ideal elasto-plastic model problem based on von-mise yield criterion.
* the constitutive relation between stress increment and strain increment
* can be found at [elastoplastic.pdf](../elastoplastic.pdf) 
*/
namespace ElastoPlastic
{
	using namespace dealii;
template <int dim>
/** Implementation of incremental FEM for elasto-plastic model problem
* mathematical aspect can be found at [femfoundation.pdf](../femfoundation.pdf) 
*/
class SPHERICAL_SHELL
{
	public:
        /** user-provided parameter to initialize the class
        * \param elementType_i input element Type, the number means DOF at each dimension, usually 2 or 3.
        * see [parameter.txt](../parameter.txt) for typical configuration.
        *
        * \param givenMaxNewtonStep maximal allowed Newton steps
        * \param quadratureCnt_i quadrature points at each dimension
        */
		SPHERICAL_SHELL(char* fileName, double inner_radius_i, double outer_radius_i, double inner_pressure_i,
			double YoungModulus_i, double PoissonRatio_i, int elementType_i, int quadratureCnt_i,int givenMaxNewtonStep,double yield_strength_i);
		~SPHERICAL_SHELL();
		void run(bool readSolution);
	private:
		int elementType;//1 or 2
		char inputFileName[20];//!< inputFileName such as hollowSphere.msh, same with hollowSphereUpdate::hollowSphere input mesh file
		int quadratureCnt;
		double inner_radius;
		double outer_radius;
		double yield_strength;
		double inner_pressure;
		double YoungModulus;
		double PoissonRatio;
		double shearModulus;
		double LamesFirstParameter;
		int MaxNewtonStep;
		void setup_system();
        /** given a rank 2 tensor 
        * if the von-mises stress exceeds the yielding limit, then the region is yielded.
        */
		bool isYield(Tensor<2, dim>& stressTensor);
        /** relax the stressTensorIncrement to make sure the yielding limit is not exceeded.
        */
		double compute_revision_parameter(Tensor<2, dim>& stressTensor,Tensor<2, dim>& stressTensorIncrement);
		void output_results();
		void setupOutput();
		void assemble_system(bool isFirstStep);
		void solve(bool isFirstStep);
		void compute_stress_field(bool isFirstStep);
		double compute_residual(const double alpha) const;
        /** implementation of constitutive model of ideal plastic
        */
		double Dp(int i, int j, int k, int l,Tensor<2,dim>& stress_tensor_last_step);//compute plastic tangent modulus
		double determine_step_length() const;
		bool get_neumann_value(const Point<dim>   &p, Vector<double>  &values)const;
		Triangulation<dim>   triangulation;

		DoFHandler<dim>      dof_handler;
		FESystem<dim>           fe;

		ConstraintMatrix     hanging_node_constraints;

		SparsityPattern      sparsity_pattern;
		SparseMatrix<double> system_matrix;

		Vector<double>       present_solution;
		Vector<double>       newton_update;
		Vector<double>       system_rhs;
		std::vector<std::vector<Tensor<2, dim>>> stress_field;
		Tensor<2, dim>		identityTensor;
		bool* isPlasticRegion;
		std::map<int, Point<dim>> PointMap;
		std::vector<std::pair<Point<dim>, Tensor<2, dim>>> StressMap;
	
	};
	template <int dim>
	void SPHERICAL_SHELL<dim>::setupOutput() {
		const unsigned int   dofs_per_cell = fe.dofs_per_cell;
		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
		DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
		DoFHandler<dim>::active_cell_iterator endc = dof_handler.end();
		std::vector<Point<dim>> point_collection = fe.get_unit_support_points();
		MappingQ1<dim> mapping_instance;
		Point<dim> center_point;
		for (int i = 0; i < dim; i++) {
			center_point[i] = 0.5;			
		}
		Quadrature<dim> quadrature_instance(center_point);
		FEValues<dim> fe_values(fe, quadrature_instance, update_gradients | update_quadrature_points);
		int cell_index = 0;
		const unsigned int n_q_points = dim==3? quadratureCnt*quadratureCnt*quadratureCnt:quadratureCnt*quadratureCnt;
		for (; cell != endc; ++cell) {
			Assert(dofs_per_cell == point_collection.size(), ExcMessage("dofs_per_cell!=point_collection.size()"));
			//map point_collection to the current cell
			fe_values.reinit(cell);
			cell->get_dof_indices(local_dof_indices);
			StressMap.push_back(std::pair<Point<dim>, Tensor<2, dim>>(fe_values.quadrature_point(0), stress_field[cell_index][n_q_points]));
			for (int i = 0; i < point_collection.size(); i++) {
				PointMap[local_dof_indices[i]] = mapping_instance.transform_unit_to_real_cell(cell, point_collection[i]);
				//use the center point of unit cube to calculate stress tensor
				Assert(local_dof_indices[i] % 3 == 0, ExcMessage("local_dof_indices[i]/3!=0"));
				i += dim - 1;
				//store the index of global dof index of the first coordinate of this point
			}
			cell_index++;
		}
	}
	template <int dim>
	inline double SPHERICAL_SHELL<dim>::compute_revision_parameter(Tensor<2, dim>& stressTensor, Tensor<2, dim>& stressTensorIncrement) {
		Tensor<2, dim> deviator = stressTensor - trace(stressTensor)*identityTensor / 3;
		double a3 = double_contract(deviator, deviator)- 2 * pow(yield_strength, 2) / 3;
		Tensor<2, dim> deviator_increment = stressTensorIncrement - trace(stressTensorIncrement)*identityTensor / 3;
		double a1 = double_contract(deviator_increment, deviator_increment);
		double a2=2* double_contract(deviator, deviator_increment);
		return (-a2 + sqrt(a2*a2 - 4 * a1*a3)) / (2 * a1);			
	}
	template <int dim>
	inline double SPHERICAL_SHELL<dim>::Dp(int i, int j, int k, int l,Tensor<2,dim>& stress_tensor_last_step) {
		//implementation of perfect plastic material with Ep=0
		Tensor<2, dim> deviator_tensor=stress_tensor_last_step - trace(stress_tensor_last_step)*identityTensor / 3;;
		return 3 * shearModulus*deviator_tensor[i][j] * deviator_tensor[k][l] / pow(yield_strength, 2);
	}
	template <int dim>
	void SPHERICAL_SHELL<dim>::compute_stress_field(bool isFirstStep) {
		const QGauss<dim>  quadrature_formula(quadratureCnt);
		const unsigned int n_q_points = quadrature_formula.size();

		const unsigned int   dofs_per_cell = fe.dofs_per_cell;
		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
		//average over all stress tensor in quadrature points to get an aprroximated stress tensor for the current cell
		DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
		DoFHandler<dim>::active_cell_iterator endc = dof_handler.end();
		FEValues<dim> fe_values(fe, quadrature_formula, update_gradients);
		
		
		
		int cell_index = 0;
		for (; cell != endc; ++cell) {
			std::vector<Tensor<2, dim>> stress_field_tmp;//stress tensor increment
			stress_field_tmp.reserve(n_q_points);
			fe_values.reinit(cell);
			cell->get_dof_indices(local_dof_indices);
			Tensor<2, dim> myStressTotal;
			for (int i = 0; i < n_q_points; i++) {
				Tensor<2, dim> myGradient;
				for (int j = 0; j < dofs_per_cell; j++) {
					const unsigned int
						component_j = fe.system_to_component_index(j).first;
					myGradient[component_j] += newton_update[local_dof_indices[j]] * fe_values.shape_grad(j, i);
				}
				Tensor<2, dim> strainTensorIncrement = (myGradient + transpose(myGradient)) / 2;
				
					Tensor<2, dim> stressTensor = (LamesFirstParameter * trace(strainTensorIncrement))*identityTensor + (2 * shearModulus)*strainTensorIncrement;
					if (!isFirstStep && isPlasticRegion[cell_index]) {
						for (int i1 = 0; i1 < dim; i1++)
							for (int j1 = 0; j1 < dim; j1++)
								for (int k = 0; k < dim; k++)
									for (int L1 = 0; L1 < dim; L1++)
										stressTensor[i1][j1] -= Dp(i1,j1,k,L1, stress_field[cell_index][i])*strainTensorIncrement[k][L1];
					}
					//von Mise criterion
					stress_field_tmp.push_back(stressTensor);	
					myStressTotal += stressTensor;
			}
			myStressTotal /= n_q_points;
			stress_field_tmp.push_back(myStressTotal);
			bool currentState = isFirstStep?isYield(myStressTotal):isYield(stress_field[cell_index][n_q_points]+myStressTotal);
			if (currentState && !isPlasticRegion[cell_index]) {//last step is elastic,current step is elasto-plastic
				//revise the stress increment
				//stress_field_tmp: stress_increment
				//stress_field[cell_index][n_q_points]: stress_last_step
				double alpha;
				if (isFirstStep) {
					alpha = 0.99*compute_revision_parameter(Tensor<2,dim>(), myStressTotal);
					//verify the new stress:stress_field[cell_index][n_q_points]+alpha*myStressTotal
					//satisfy Plastic Potential F(sigma)=0 approximately
					Assert(isYield(alpha*myStressTotal)==false, ExcMessage("revision parameter not work!"));
				}
				else{
					alpha = compute_revision_parameter(stress_field[cell_index][n_q_points], myStressTotal);
				}
				for (int i = 0; i <= n_q_points; i++) {
					stress_field_tmp[i] *= alpha;
				}
				isPlasticRegion[cell_index] = currentState;

			}
			//if plastic transfers to elastic, assert fails
			//Assert(!(currentState==false && isPlasticRegion[cell_index]), ExcMessage("plastic transfers to elastic!"));

			if(isFirstStep){
			stress_field.push_back(stress_field_tmp);
			}
			else {
				for (int i = 0; i <= n_q_points; i++) {
					stress_field[cell_index][i] += stress_field_tmp[i];
				}
			}
			cell_index++;
		}
	}
	template <int dim>
	inline bool SPHERICAL_SHELL<dim>::isYield(Tensor<2,dim>& stressTensor) {
		Tensor<2, dim> deviator=stressTensor-trace(stressTensor)*identityTensor/3;
		double J2 = double_contract(deviator, deviator);
		return (J2>=(2*pow(yield_strength,2)/3));
	}
	template <int dim>
	SPHERICAL_SHELL<dim>::SPHERICAL_SHELL(char* fileName, double inner_radius_i, double outer_radius_i, double inner_pressure_i,
		double YoungModulus_i, double PoissonRatio_i, int elementType_i, int quadratureCnt_i, int givenMaxNewtonStep,double yield_strength_i)
		:
		elementType(elementType_i),
		inner_radius(inner_radius_i),
		outer_radius(outer_radius_i),
		MaxNewtonStep(givenMaxNewtonStep),
		yield_strength(yield_strength_i),
		inner_pressure(inner_pressure_i),
		YoungModulus(YoungModulus_i),
		PoissonRatio(PoissonRatio_i),
		quadratureCnt(quadratureCnt_i),
		dof_handler(triangulation),
		fe(FE_Q<dim>(elementType_i), dim)
	{
		strcpy_s(inputFileName, fileName);
		for (int j = 0; j < dim; j++) {
			identityTensor[j][j] = 1;
		}
		shearModulus = YoungModulus / (2 * (1 + PoissonRatio));
		LamesFirstParameter = YoungModulus*PoissonRatio / ((1 + PoissonRatio)*(1 - 2 * PoissonRatio));
	}

	template <int dim>
	bool SPHERICAL_SHELL<dim>::get_neumann_value(const Point<dim>   &p, Vector<double>  &values) const{
		if(abs(p.norm() - inner_radius)< abs(p.norm() - outer_radius)){
		for (int i = 0; i<dim; i++)
			values[i] = inner_pressure*p[i] / p.norm();
		return true;
		}
		return false;
	}
	template <int dim>
	SPHERICAL_SHELL<dim>::~SPHERICAL_SHELL()
	{
		delete[] isPlasticRegion;
		dof_handler.clear();
	}

	


	template <int dim>
	void SPHERICAL_SHELL<dim>::output_results() {//output result in vtk format
		//output data in csv format
		//write csv for pycharm analysis  
		std::ofstream output("hollowSphere.csv");
		output << "Point:0" << ',' << "Point:1" << ',';
		if (dim == 3)output << "Point:2" << ',';
		output << "x_displacement" << ',' << "y_displacement";
		if (dim == 3)output << ',' << "z_displacement";
		output << std::endl;
		for (auto item = PointMap.begin(); item != PointMap.end(); item++) {
			int i = item->first;
			Point<dim> currentPoint = item->second;
			output << currentPoint(0) << ',' << currentPoint(1) << ',';
			if (dim == 3)output << currentPoint(2) << ',';
			output << present_solution[i] << ',' << present_solution[i + 1];
			if (dim == 3)output << ',' << present_solution[i + 2];
			output << std::endl;
		}
		output.close();
		std::string stressOutput("hollowSphere.csv");
		stressOutput.replace(stressOutput.length() - 4, stressOutput.length() - 1, "2.csv");
		std::ofstream output2(stressOutput.c_str());
		output2 << "Point:0" << ',' << "Point:1" << ',';
		if (dim == 3)output2 << "Point:2" << ',';
		output2 << "tensor_xx" << ',' << "tensor_xy" << ',' << "tensor_yy";
		if (dim == 3)output2 << ',' << "tensor_xz" << ',' << "tensor_yz" << ',' << "tensor_zz";
		output2 << std::endl;
		for (auto i = StressMap.begin(); i != StressMap.end(); i++) {
			Point<dim> currentPoint = i->first;
			output2 << currentPoint(0) << ',' << currentPoint(1) << ',';
			if (dim == 3)output2 << currentPoint(2) << ',';
			Tensor<2, dim> currentTensor = i->second;
			output2 << currentTensor[0][0] << ',' << currentTensor[0][1] << ',' << currentTensor[1][1];
			if (dim == 3)output2 << ',' << currentTensor[0][2] << ',' << currentTensor[1][2] << ',' << currentTensor[2][2];
			output2 << std::endl;
		}
		output2.close();
		//output in vtk format
		const unsigned int   dofs_per_cell = fe.dofs_per_cell;
		const unsigned int   n_q_points = (dim == 2 ? quadratureCnt*quadratureCnt : quadratureCnt*quadratureCnt*quadratureCnt);
		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
		const std::string filename = "solution-elastoPlastic-sphere.vtk";
		std::ofstream out(filename.c_str());
		DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
		DoFHandler<dim>::active_cell_iterator endc = dof_handler.end();
		int cell_index = 0;
		std::stringstream vertexInfo;
		std::stringstream cellInfo;
		std::stringstream DisplacementData;
		std::stringstream StressTensorData;
		std::stringstream isPlasticRegionData;
		out << "# vtk DataFile Version 3.0"
			<< '\n'
			<< "#This file was generated by the deal.II library";
			out << " on " << Utilities::System::get_date()
				<< " at " << Utilities::System::get_time();
		out << '\n'
			<< "ASCII"
			<< '\n';
		// now output the data header
		out << "DATASET UNSTRUCTURED_GRID\n"
			<< '\n';

		for (; cell != endc; ++cell) {
			for (int i = 0; i < GeometryInfo<dim>::vertices_per_cell; i++) {
				if(dim==3){
					vertexInfo<<cell->vertex(i) << '\n';
				}
				else if (dim == 2) {
					vertexInfo << cell->vertex(i)<<" 0" << '\n';
				}
			}
			if(dim==3){
				cellInfo << 8 << '\t'
					<< cell_index * 8 << '\t' << cell_index * 8 + 1 << '\t'
					<< cell_index * 8 + 3 << '\t' << cell_index * 8 + 2<<'\t'
					<< cell_index * 8+4 << '\t' << cell_index * 8 + 5 << '\t'
					<< cell_index * 8 + 7 << '\t' << cell_index * 8 + 6 << '\n';
			}
			else if (dim == 2) {
				cellInfo << 4 << '\t' 
					<< cell_index * 4<<'\t' << cell_index * 4+1 << '\t' 
					<< cell_index * 4+3 << '\t' << cell_index * 4+2<<'\n';
			}
			cell->get_dof_indices(local_dof_indices);
			int* dof_rearrange_array;
			if (dim == 2) {
				int dof_rearrange_tmp_array[4] = { 0,1,3,2 };
				dof_rearrange_array = new int[4];
				memcpy(dof_rearrange_array, dof_rearrange_tmp_array, sizeof(int) * 4);
			}
			else if (dim == 3) {
				int dof_rearrange_tmp_array[8] = { 0,1,3,2,4,5,7,6};
				dof_rearrange_array = new int[8];
				memcpy(dof_rearrange_array, dof_rearrange_tmp_array, sizeof(int) * 8);
			}
			for (int i = 0; i < (dim==2?4:8); i++) {
				DisplacementData << present_solution[local_dof_indices[dim*dof_rearrange_array[i]]] << ' '
					<< present_solution[local_dof_indices[dim*dof_rearrange_array[i]] + 1] << ' '
					<< (dim == 2 ? '0' : present_solution[local_dof_indices[dim*dof_rearrange_array[i]] + 2]) << '\n';
			}
			StressTensorData << stress_field[cell_index][n_q_points]<<'\n';
			isPlasticRegionData << isPlasticRegion[cell_index] << ' ';
			cell_index++;
		}
		out << "POINTS " << cell_index*GeometryInfo<dim>::vertices_per_cell << " double\n"
			<< vertexInfo.str();
		out << "CELLS " << cell_index << ' ' << cell_index*(dim == 2 ? 5 : 9) << '\n'
			<< cellInfo.str();
		out << "CELL_TYPES " << cell_index << '\n';
		int cell_type = (dim==2?9:12);
		for (int i = 0; i < cell_index; i++)
			out << cell_type << ' ';
		out << "\nPOINT_DATA " << cell_index*GeometryInfo<dim>::vertices_per_cell << '\n';
		out << "VECTORS displacement double \n"
			<< DisplacementData.str();
		out << "CELL_DATA " << cell_index << '\n';
		out << "SCALARS isPlasticRegion bit\n"
			<< "LOOKUP_TABLE default\n"
			<< isPlasticRegionData.str();
		out << "TENSORS stress double \n"
			<< StressTensorData.str();
		out.close();
		//output stress_filed and isPlastic as cell_data
	}
	template <int dim>
	void SPHERICAL_SHELL<dim>::setup_system()
	{
	
		dof_handler.distribute_dofs(fe);
		std::cout << "Number of degrees of freedom: "
			<< dof_handler.n_dofs()
			<< std::endl;
		isPlasticRegion = new bool[triangulation.n_active_cells()];
		memset(isPlasticRegion, 0, triangulation.n_active_cells());
		present_solution.reinit(dof_handler.n_dofs());
		newton_update.reinit(dof_handler.n_dofs());
		system_rhs.reinit(dof_handler.n_dofs());

		stress_field.reserve(dof_handler.n_dofs());
		DynamicSparsityPattern dsp(dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(dof_handler, dsp);
		sparsity_pattern.copy_from(dsp);
		system_matrix.reinit(sparsity_pattern);
	}

	template <int dim>
	void SPHERICAL_SHELL<dim>::assemble_system(bool isFirstStep)
	{
		const QGauss<dim>  quadrature_formula(quadratureCnt);
		const QGauss<dim - 1> face_quadrature_formula(quadratureCnt);
		system_matrix = 0;
		system_rhs = 0;
		FEValues<dim> fe_values(fe, quadrature_formula,
			update_values|
			update_gradients |
			update_quadrature_points |
			update_JxW_values);
		FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
			update_values | update_quadrature_points |
			update_normal_vectors | update_JxW_values);

		const unsigned int           dofs_per_cell = fe.dofs_per_cell;
		const unsigned int           n_q_points = quadrature_formula.size();
		const unsigned int n_face_q_points = face_quadrature_formula.size();
		FullMatrix<double>           cell_matrix(dofs_per_cell, dofs_per_cell);
		Vector<double>               cell_rhs(dofs_per_cell);

		std::vector<types::global_dof_index>    local_dof_indices(dofs_per_cell);

		typename DoFHandler<dim>::active_cell_iterator
			cell = dof_handler.begin_active(),
			endc = dof_handler.end();
		int cell_index = 0;
		for (; cell != endc; ++cell)
		{
			bool yielding_status =  isFirstStep? false:isPlasticRegion[cell_index];
			cell_matrix = 0;
			cell_rhs = 0;

			fe_values.reinit(cell);
			for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
			{			
				for (unsigned int i = 0; i < dofs_per_cell; ++i)
				{
					const unsigned int
						component_i = fe.system_to_component_index(i).first;
					for (unsigned int j = 0; j < dofs_per_cell; ++j)
					{
						const unsigned int
							component_j = fe.system_to_component_index(j).first;
						//use elastic constitutive model

						cell_matrix(i, j) += (
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


						if (yielding_status) {//revise the constitutive model with plastic tangent modulus;
							for(int j1=0;j1<dim;j1++)
								for(int L1=0;L1<dim;L1++)
								cell_matrix(i, j) -= fe_values.shape_grad(i, q_point)[j1] *
								Dp(component_i,j1,component_j,L1,stress_field[cell_index][q_point])*
								fe_values.shape_grad(j, q_point)[L1] *
									fe_values.JxW(q_point);
						}
					}

					if (!isFirstStep) {//revise the RHS with previous stress
						cell_rhs(i) -= (fe_values.shape_grad(i, q_point)
							* stress_field[cell_index][q_point][component_i]/*q_point info?*/
							* fe_values.JxW(q_point));
					}
				}
			}
			for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point){
				Vector<double> neumann_value(dim);
				for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
					if (cell->face(face_number)->boundary_id()==0)
					{
					
						fe_face_values.reinit(cell, face_number);
						
							if(get_neumann_value(fe_face_values.quadrature_point(q_point), neumann_value)){
								for (unsigned int i = 0; i < dofs_per_cell; ++i) {
									const unsigned int
										component_face_i = fe.system_to_component_index(i).first;

									cell_rhs(i) += (neumann_value[component_face_i] *
										fe_face_values.shape_value(i, q_point) *
										fe_face_values.JxW(q_point));
									
							}
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
			cell_index++;
		}
		std::map<types::global_dof_index, double> boundary_values;
		VectorTools::interpolate_boundary_values(dof_handler,
			1,/*boundary_id*/
			ZeroFunction<dim>(dim),
			boundary_values,fe.component_mask(FEValuesExtractors::Scalar(0)));
		VectorTools::interpolate_boundary_values(dof_handler,
			2,/*boundary_id*/
			ZeroFunction<dim>(dim),
			boundary_values, fe.component_mask(FEValuesExtractors::Scalar(1)));
		VectorTools::interpolate_boundary_values(dof_handler,
			3,/*boundary_id*/
			ZeroFunction<dim>(dim),
			boundary_values, fe.component_mask(FEValuesExtractors::Scalar(2)));
		MatrixTools::apply_boundary_values(boundary_values,
			system_matrix,
			newton_update,
			system_rhs);	
	}
	template <int dim>
	void SPHERICAL_SHELL<dim>::solve(bool isFirstStep)
	{
		SolverControl           solver_control(4000, 1e-5);
		SolverGMRES<>              solver(solver_control);
		solver.solve(system_matrix, newton_update, system_rhs,
			PreconditionIdentity());
		const double alpha = isFirstStep? 1:determine_step_length();
		present_solution.add(alpha, newton_update);
		
		std::ofstream file{ "solution_archive_ElastoPlastic.txt" };
		boost::archive::text_oarchive oa(file);
		present_solution.save(oa, 1);
	}
	template <int dim>
	double SPHERICAL_SHELL<dim>::compute_residual(const double alpha) const
	{	
		return 0;
	}
	template <int dim>
	double SPHERICAL_SHELL<dim>::determine_step_length() const
	{
		return 1.0;
	}

	template <int dim>
	void SPHERICAL_SHELL<dim>::run(bool readSolution)
	{
		bool         first_step = true;
		GridIn<dim> grid_in;
		grid_in.attach_triangulation(triangulation);
		std::ifstream input_file(inputFileName);
		grid_in.read_msh(input_file);
		std::cout << "Number of active cells: "
			<< triangulation.n_active_cells()
			<< std::endl;
		double previous_res = 1;
		static unsigned int inner_iteration = 0;
		setup_system();
		while (first_step || (previous_res>1e-2))
		{
			if (first_step == true)
			{
				std::cout << "******** Initial Solution with Elastic Equation "
					<< " ********"
					<< std::endl;
				
				if(readSolution){
				std::ifstream file{ "solution_archive_ElastoPlastic.txt" };
				boost::archive::text_iarchive ia(file);
				newton_update.load(ia, 1);
				present_solution.add(newton_update);
				}
				else{
				assemble_system(true);
				solve(true);
				}
			
				compute_stress_field(true);
				
				first_step = false;
				inner_iteration += 1;
				continue;
			}
			if (inner_iteration > MaxNewtonStep)
				break;
			
			assemble_system(false);
			previous_res = system_rhs.l2_norm();
			solve(false);
			std::cout << "At Newton Step "<<inner_iteration<<"  Residual is: "
					<< previous_res
					<< std::endl;
			compute_stress_field(false);
			inner_iteration += 1;	
		}
		setupOutput();
		output_results();
	}
}
int main (int argc,char** argv)
{
    using namespace ElastoPlastic;
    if (argc == 1) {
        std::cout << "No input mesh file provided!" << std::endl;
        exit(0);
    }
    if (argc == 2) {
        std::cout << "No parameter file provided!" << std::endl;
    }
    std::ifstream fin(argv[2]);
    if (fin.fail()) {
        std::cout << "parameter file " << argv[2] << " is invalid!" << std::endl;
    }
    double inner_radius, outer_radius, inner_pressure, YoungModulus, PoissonRatio, yield_strength;
    int elementType, quadratureCnt,MaxNewtonStep;
    bool useSavedSolution;
    int initializationCnt = 0;
    char buffer[20];
    while (!fin.eof()) {
        fin >> buffer;
        if (strcmp(buffer, "inner_radius") == 0) {
            fin >> inner_radius;
            initializationCnt = initializationCnt | 1;
        }
        else if (strcmp(buffer, "outer_radius") == 0) {
            fin >> outer_radius;
            initializationCnt = initializationCnt | 2;
        }
        else if (strcmp(buffer, "MaxNewtonStep") == 0) {
            fin >> MaxNewtonStep;
            initializationCnt = initializationCnt | 4;
        }
        else if (strcmp(buffer, "inner_pressure") == 0) {
            fin >> inner_pressure;
            initializationCnt = initializationCnt | 8;
        }
        else if (strcmp(buffer, "YoungModulus") == 0) {
            fin >> YoungModulus;
            initializationCnt = initializationCnt | 16;
        }
        else if (strcmp(buffer, "PoissonRatio") == 0) {
            fin >> PoissonRatio;
            initializationCnt = initializationCnt | 32;
        }
        else if (strcmp(buffer, "elementType") == 0) {
            fin >> elementType;
            initializationCnt = initializationCnt | 64;
        }
        else if (strcmp(buffer, "quadratureCnt") == 0) {
            fin >> quadratureCnt;
            initializationCnt = initializationCnt | 128;
        }
        else if (strcmp(buffer, "useSavedSolution") == 0) {
            fin >> useSavedSolution;
            initializationCnt = initializationCnt | 256;
        }
        else if (strcmp(buffer, "yield_strength") == 0) {
            fin >> yield_strength;
            initializationCnt = initializationCnt | 512;
        }

    }
    //check for parameter integrity
    {
        if (!(initializationCnt & 1)) {
            std::cerr << "inner_radius is missing" << std::endl;
        }
        else if (!(initializationCnt & 2)) {
            std::cerr << "outer_radius is missing" << std::endl;
        }
        else if (!(initializationCnt & 4)) {
            std::cerr << "MaxNewtonStep is missing" << std::endl;
        }
        else if (!(initializationCnt & 8)) {
            std::cerr << "inner_pressure is missing" << std::endl;
        }
        else if (!(initializationCnt & 16)) {
            std::cerr << "YoungModulus is missing" << std::endl;
        }
        else if (!(initializationCnt & 32)) {
            std::cerr << "PoissonRatio is missing" << std::endl;
        }
        else if (!(initializationCnt & 64)) {
            std::cerr << "elementType is missing" << std::endl;
        }
        else if (!(initializationCnt & 128)) {
            std::cerr << "quadratureCnt is missing" << std::endl;
        }
        else if (!(initializationCnt & 256)) {
            std::cerr << "useSavedSolution is missing" << std::endl;
        }
        else if (!(initializationCnt & 512)) {
            std::cerr << "yield_strength is missing" << std::endl;
        }
        
        else {
            std::cout << "parse input parameter file successfully." << std::endl;
        }
        if (!(initializationCnt & (0x3ff))) {
            std::cerr << "Please fill the missing input parameter" << std::endl;
            exit(0);
        }
    }

    SPHERICAL_SHELL<3> sphere3D(argv[1], inner_radius, outer_radius, inner_pressure, YoungModulus, PoissonRatio, elementType, quadratureCnt, MaxNewtonStep,yield_strength);
    sphere3D.run(useSavedSolution);


    return 0;
}

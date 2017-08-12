#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <fstream>
#include <iostream>
/** namespace ClassicalFracture contains code to solve a classical fracture problem
*/
namespace ClassicalFracture
{
	using namespace dealii;
template <int dim>
/** use normal fe method to solve the filed of a plate with mode I fracture.
* XFEM method is not implemented yet.
* for problem description, see [classical_fracture_problem_description.png](../classical_fracture_problem_description.png)
*/
class FractureProblem
{
	public:
		FractureProblem(
			double YoungModulus_i, double PoissonRatio_i, double height_i, double width_i,
			double fracture_width_i, char* input_mesh_fileName_i, char* output_result_fileName_i,double sigma_i);
		~FractureProblem();
		void run();
	private:
		void setup_system();
		void assemble_system();
		void solve();
		void read_grid();
		void output_results() const;
		void refine_grid();
		Triangulation<dim>   triangulation;
		DoFHandler<dim>      dof_handler;
		FESystem<dim>        fe;
		ConstraintMatrix     hanging_node_constraints;
		SparsityPattern      sparsity_pattern;
		SparseMatrix<double> system_matrix;
		Vector<double>       solution;
		Vector<double>       system_rhs;
		double YoungModulus;
		double PoissonRatio;
		double shearModulus;
		double LamesFirstParameter;
		double height;
		double width;
		double fracture_width;
		double sigma;
		char input_mesh_fileName[20];
		char output_result_fileName[20];
	};
	template <int dim>
	FractureProblem<dim>::FractureProblem(double YoungModulus_i, double PoissonRatio_i, double height_i, double width_i,
		double fracture_width_i, char* input_mesh_fileName_i, char* output_result_fileName_i,double sigma_i)
		:
		YoungModulus(YoungModulus_i),
		PoissonRatio(PoissonRatio_i),
		height(height_i),
		width(width_i),
		fracture_width(fracture_width_i),
		dof_handler(triangulation),
		sigma(sigma_i),
		fe(FE_Q<dim>(1), dim)
	{
		strcpy_s(input_mesh_fileName, input_mesh_fileName_i);
		strcpy_s(output_result_fileName, output_result_fileName_i);
		shearModulus = YoungModulus / (2 * (1 + PoissonRatio));
		LamesFirstParameter = YoungModulus*PoissonRatio / ((1 + PoissonRatio)*(1 - PoissonRatio));
	}
	template <int dim>
	FractureProblem<dim>::~FractureProblem()
	{
		dof_handler.clear();
	}
	template <int dim>
	void FractureProblem<dim>::setup_system()
	{
		dof_handler.distribute_dofs(fe);
		hanging_node_constraints.clear();
		DoFTools::make_hanging_node_constraints(dof_handler,
			hanging_node_constraints);
		hanging_node_constraints.close();
		DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
		DoFTools::make_sparsity_pattern(dof_handler,
			dsp,
			hanging_node_constraints,
			/*keep_constrained_dofs = */ true);
		std::cout << "   Number of degrees of freedom: "
			<< dof_handler.n_dofs()
			<< std::endl;

		sparsity_pattern.copy_from(dsp);
		system_matrix.reinit(sparsity_pattern);
		solution.reinit(dof_handler.n_dofs());
		system_rhs.reinit(dof_handler.n_dofs());
	}
	template <int dim>
	void FractureProblem<dim>::assemble_system()
	{
		QGauss<dim>  quadrature_formula(2);
		QGauss<dim - 1> face_quadrature_formula(2);

		FEValues<dim> fe_values(fe, quadrature_formula,
			update_values | update_gradients |
			update_quadrature_points | update_JxW_values);
		FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
			update_values | update_quadrature_points |
			update_normal_vectors | update_JxW_values);

		const unsigned int   dofs_per_cell = fe.dofs_per_cell;
		const unsigned int   n_q_points = quadrature_formula.size();
		const unsigned int n_face_q_points = face_quadrature_formula.size();

		FullMatrix<double>   cell_matrix(dofs_per_cell, dofs_per_cell);
		Vector<double>       cell_rhs(dofs_per_cell);
		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
		typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
			endc = dof_handler.end();
		for (; cell != endc; ++cell)
		{
			cell_matrix = 0;
			cell_rhs = 0;
			fe_values.reinit(cell);
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
			auto point_first = cell->vertex(0);
			for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
				if (cell->face(face_number)->at_boundary() && cell->face(face_number)->boundary_id()!=0)
				{
					fe_face_values.reinit(cell, face_number);
					if(cell->face(face_number)->boundary_id()==1){
					for (unsigned int q_point = 0; q_point<n_face_q_points; ++q_point)
					{
						for (unsigned int i = 0; i < dofs_per_cell; ++i) {
							const unsigned int
								component_face_i = fe.system_to_component_index(i).first;
							if(component_face_i!=0)
							cell_rhs(i) += (sigma *(point_first[1]>0?1:-1)*
								fe_face_values.shape_value(i, q_point) *
								fe_face_values.JxW(q_point));
						}
					}
					}
					else if (cell->face(face_number)->boundary_id() == 2) {
						for (unsigned int q_point = 0; q_point<n_face_q_points; ++q_point)
						{
							for (unsigned int i = 0; i < dofs_per_cell; ++i) {
								const unsigned int
									component_face_i = fe.system_to_component_index(i).first;
								if (component_face_i != 1)
									cell_rhs(i) += (sigma *(point_first[0]>0 ? 1 : -1)*
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
		}
		hanging_node_constraints.condense(system_matrix);
		hanging_node_constraints.condense(system_rhs);
	}
	template <int dim>
	void FractureProblem<dim>::solve()
	{
		SolverControl           solver_control(system_rhs.size(), 1e-8);
		SolverCG<>              cg(solver_control);
		PreconditionSSOR<> preconditioner;
		preconditioner.initialize(system_matrix, 1.2);
		cg.solve(system_matrix, solution, system_rhs,
			preconditioner);
		hanging_node_constraints.distribute(solution);
	}
	template <int dim>
	void FractureProblem<dim>::read_grid()
	{
		GridIn<dim> grid_in;
		grid_in.attach_triangulation(triangulation);
		std::ifstream input_file(input_mesh_fileName);
		grid_in.read_msh(input_file);
		std::cout << "Number of active cells: "
			<< triangulation.n_active_cells()
			<< std::endl;

	}
	template <int dim>
	void FractureProblem<dim>::output_results() const
	{
		std::ofstream output(output_result_fileName);
		DataOut<dim> data_out;
		data_out.attach_dof_handler(dof_handler);
		std::vector<std::string> solution_names;
		std::vector< DataComponentInterpretation::DataComponentInterpretation >data_component_interpretation;
		data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
		data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
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
		data_out.add_data_vector(dof_handler,solution, solution_names,data_component_interpretation);
		data_out.build_patches();
		data_out.write_vtk(output);
	}
	template <int dim>
	void FractureProblem<dim>::refine_grid()
	{
		Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
		KellyErrorEstimator<dim>::estimate(dof_handler,
			QGauss<dim - 1>(2),
			typename FunctionMap<dim>::type(),
			solution,
			estimated_error_per_cell);
		GridRefinement::refine_and_coarsen_fixed_number(triangulation,
			estimated_error_per_cell,
			0.3, 0.03);
		triangulation.execute_coarsening_and_refinement();
		std::cout << "   Number of active cells:       "
			<< triangulation.n_active_cells()
			<< std::endl;
	}

	template <int dim>
	void FractureProblem<dim>::run()
	{
			read_grid();
			for (unsigned int cycle = 0; cycle < 5; ++cycle)
			{
				std::cout << "Cycle " << cycle << ':' << std::endl;
				if (cycle != 0) refine_grid();
				setup_system();
				assemble_system();
				solve();
			}
			output_results();
	}
}
int main(int argc, char** argv)
{
	if (argc == 1) {
		std::cout << "No parameter file provided!" << std::endl;
		exit(0);
	}
	std::ifstream fin(argv[1]);
	if (fin.fail()) {
		std::cout << "parameter file " << argv[1] << " is invalid!" << std::endl;
	}
	;
	char input_mesh_fileName[20];
	char output_result_fileName[20];
	double height, width, fracture_width, YoungModulus, PoissonRatio,sigma;
	int elementType, quadratureCnt;
	bool useSavedSolution;
	int initializationCnt = 0;
	char buffer[40];
	while (!fin.eof()) {
		fin >> buffer;
		if (strcmp(buffer, "height") == 0) {
			fin >> height;
			initializationCnt = initializationCnt | 1;
		}
		else if (strcmp(buffer, "width") == 0) {
			fin >> width;
			initializationCnt = initializationCnt | 2;
		}
		else if (strcmp(buffer, "fracture_width") == 0) {
			fin >> fracture_width;
			initializationCnt = initializationCnt | 4;
		}
		else if (strcmp(buffer, "input_mesh_fileName") == 0) {
			fin >> input_mesh_fileName;
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
		else if (strcmp(buffer, "output_result_fileName") == 0) {
			fin >> output_result_fileName;
			initializationCnt = initializationCnt | 64;
		}
		else if (strcmp(buffer, "sigma") == 0) {
			fin >> sigma;
			initializationCnt = initializationCnt | 128;
		}
		else if (strcmp(buffer, "useSavedSolution") == 0) {
			fin >> useSavedSolution;
		//	initializationCnt = initializationCnt | 256;
		}
	}
	//check for parameter integrity
	{
		if (!(initializationCnt & 1)) {
			std::cerr << "height is missing" << std::endl;
		}
		else if (!(initializationCnt & 2)) {
			std::cerr << "width is missing" << std::endl;
		}
		else if (!(initializationCnt & 4)) {
			std::cerr << "fracture_width is missing" << std::endl;
		}
		else if (!(initializationCnt & 8)) {
			std::cerr << "input_mesh_fileName is missing" << std::endl;
		}
		else if (!(initializationCnt & 16)) {
			std::cerr << "YoungModulus is missing" << std::endl;
		}
		else if (!(initializationCnt & 32)) {
			std::cerr << "PoissonRatio is missing" << std::endl;
		}
		else if (!(initializationCnt & 64)) {
			std::cerr << "output_result_fileName is missing" << std::endl;
		}
		else if (!(initializationCnt & 128)) {
			std::cerr << "stress BC is missing" << std::endl;
		}
		//else if (!(initializationCnt & 256)) {
			//std::cerr << "useSavedSolution is missing" << std::endl;
		//}
		else {
			std::cout << "parse input parameter file successfully." << std::endl;
		}
		if ((initializationCnt & 255) != 255)
			exit(0);
	}

	try
	{
		ClassicalFracture::FractureProblem<2> elastic_problem_2d(YoungModulus,PoissonRatio,height,width,fracture_width,
			input_mesh_fileName,output_result_fileName,sigma);
		elastic_problem_2d.run();
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
	}
	return 0;
}
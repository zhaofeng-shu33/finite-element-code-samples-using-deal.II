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
#include <tuple>
using namespace dealii;
/** namespace hollowSphereUpdate contains updated code to solve a hollow elastic sphere model problem.
 *  For more info, see the namespace hollowSphereOriginal.
 */
namespace hollowSphereUpdate{
template <int dim>
class hollowSphere
{
public:
	hollowSphere(char* fileName, double inner_radius_i, double outer_radius_i, double outer_pressure_i, double inner_pressure_i,
		double YoungModulus_i, double PoissonRatio_i, int elementType_i,int quadratureCnt_i);
	void run(bool useSavedSolution);
private:
	int elementType;//1 or 2
	char input_fileName[20];//!< use methods in draw_hollowSphere to generate the input mesh file
	double inner_radius;
	double outer_radius;
	double outer_pressure;
	double inner_pressure;
	double YoungModulus;
	double PoissonRatio;
	double shearModulus;
	double LamesFirstParameter;
	int quadratureCnt;
	void get_neumann_value(const Point<dim>   &p,Vector<double>  &values)const;
/** read grid from python generated gmsh file,
 *  which is 1/8 of the sphere.
 */    
	void read_grid(char* fileName);
	void setup_system();
/** for debugging matrix solver and output results,
 *  quickly set up system from the last saved solution.
 */      
	void setup_system_from_saved_solution();
	void setupOutput();
	void assemble_system();
	void solve();
/** output results in both csv and vtk format.
 *  csv format is used for accuracy analysis script written in Python.
 */          
	void output_results(char*fileNamecsv) const;
	Triangulation<dim>     triangulation;
	FESystem<dim>               fe;
	DoFHandler<dim>        dof_handler;
	SparsityPattern      sparsity_pattern;
	SparseMatrix<double> system_matrix;
	ConstraintMatrix     constraints;
	Vector<double>       solution;
	Vector<double>       system_rhs;
	std::map<int, Point<dim>> PointMap;
	std::vector<std::pair<Point<dim>, Tensor<2, dim>>> StressMap;
};
template <int dim>
hollowSphere<dim>::hollowSphere(char* fileName, double inner_radius_i, double outer_radius_i, double outer_pressure_i, double inner_pressure_i,
	double YoungModulus_i, double PoissonRatio_i, int elementType_i, int quadratureCnt_i):
	elementType(elementType_i),
    inner_radius(inner_radius_i),
    outer_radius(outer_radius_i),
	outer_pressure(outer_pressure_i),
	inner_pressure(inner_pressure_i),
	YoungModulus(YoungModulus_i),
	PoissonRatio(PoissonRatio_i),
	quadratureCnt(quadratureCnt_i),
	fe(FE_Q<dim>(elementType_i), dim),
	dof_handler(triangulation)
{
	strcpy_s(input_fileName, fileName);
	shearModulus = YoungModulus / (2 * (1 + PoissonRatio));
	LamesFirstParameter = YoungModulus*PoissonRatio / ((1 + PoissonRatio)*(1 - 2 * PoissonRatio));
}
template <int dim>
void hollowSphere<dim>::setupOutput() {
	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
	DoFHandler<dim>::active_cell_iterator endc = dof_handler.end();
	std::vector<Point<dim>> point_collection = fe.get_unit_support_points();
	MappingQ1<dim> mapping_instance;
	Point<dim> center_point;
	Tensor<2, dim> identityTensor;
	for (int i = 0; i < dim; i++) {
		center_point[i] = 0.5;
		identityTensor[i][i] = 1;
	}
	Quadrature<dim> quadrature_instance(center_point);
	FEValues<dim> fe_values(fe, quadrature_instance, update_gradients|update_quadrature_points);
		for (; cell != endc; ++cell) {
		Assert(dofs_per_cell==point_collection.size(), ExcMessage("dofs_per_cell!=point_collection.size()"));
		//map point_collection to the current cell
		fe_values.reinit(cell);
		cell->get_dof_indices(local_dof_indices);
		Tensor<2, dim> myGradient;
		for (int j = 0; j < dofs_per_cell; j++) {
			const unsigned int
				component_j = fe.system_to_component_index(j).first;
			myGradient[component_j] += solution[local_dof_indices[j]] * fe_values.shape_grad(j, 0);
		}//get Gradient * u
		double divergent_value = trace(myGradient);
		Tensor<2, dim> strainTensor = (myGradient + transpose(myGradient)) / 2;
		Tensor<2, dim> stressTensor = (LamesFirstParameter * divergent_value)*identityTensor + (2 * shearModulus)*strainTensor;
		StressMap.push_back(std::pair<Point<dim>, Tensor<2, dim>>(fe_values.quadrature_point(0),stressTensor));
		for (int i = 0; i < point_collection.size(); i++) {
			PointMap[local_dof_indices[i]] = mapping_instance.transform_unit_to_real_cell(cell, point_collection[i]);
			//use the center point of unit cube to calculate stress tensor
			Assert(local_dof_indices[i] % 3 == 0, ExcMessage("local_dof_indices[i]/3!=0"));
			i += dim-1;
			//store the index of global dof index of the first coordinate of this point
		}
	}
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
}
template <int dim>
void hollowSphere<dim>::read_grid(char* fileName) {
	GridIn<dim> grid_in;
	grid_in.attach_triangulation(triangulation);
	std::ifstream input_file(fileName);
	grid_in.read_msh(input_file);
	std::cout << "Number of active cells: "
		<< triangulation.n_active_cells()
		<< std::endl;
}
template <int dim>
void hollowSphere<dim>::run(bool useSavedSolution)
{
		read_grid(input_fileName);
		if (useSavedSolution){
			setup_system_from_saved_solution();
			std::cout << "Load solution from solution_archive.txt finished\n";
		}
		else{
		setup_system();
		assemble_system();
		std::cout << "assemble system finished...\n";
		std::cout << "rhs norm: " << system_rhs.l2_norm() << '\n';
		solve();
		std::cout << "solve finished...\n";
		}
	
	setupOutput();
	output_results("hollowSphere.csv");
}
template <int dim>
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
	std::cout << "Memory consumption estimation of sparsity pattern: " << sparsity_pattern.memory_consumption()<<'B' << std::endl;
	std::cout << "Does the sparsity pattern compressed: ? " << sparsity_pattern.is_compressed() << std::endl;
	std::cout << "Max entries per row: " << sparsity_pattern.max_entries_per_row() << std::endl;
	system_matrix.reinit(sparsity_pattern);
	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());
}
template <int dim>
void hollowSphere<dim>::assemble_system()
{
	std::map<types::global_dof_index, double> boundary_values;
	VectorTools::interpolate_boundary_values(dof_handler,
		1,/*boundary_id*/
		ZeroFunction<dim>(dim),
		boundary_values, fe.component_mask(FEValuesExtractors::Scalar(0)));
	VectorTools::interpolate_boundary_values(dof_handler,
		2,/*boundary_id*/
		ZeroFunction<dim>(dim),
		boundary_values, fe.component_mask(FEValuesExtractors::Scalar(1)));
	VectorTools::interpolate_boundary_values(dof_handler,
		3,/*boundary_id*/
		ZeroFunction<dim>(dim),
		boundary_values, fe.component_mask(FEValuesExtractors::Scalar(2)));
	QGauss<dim>  quadrature_formula(quadratureCnt);
	QGauss<dim - 1> face_quadrature_formula(quadratureCnt);
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
		
		Vector<double> neumann_value(dim);
		for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
			if (cell->face(face_number)->boundary_id()==0)
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
		for (unsigned int i = 0; i<dofs_per_cell; ++i)
		{
			for (unsigned int j = 0; j<dofs_per_cell; ++j)
				system_matrix.add(local_dof_indices[i],
					local_dof_indices[j],
					cell_matrix(i, j));
			system_rhs(local_dof_indices[i]) += cell_rhs(i);
		}
	}
		
		MatrixTools::apply_boundary_values(boundary_values,
			system_matrix,
			solution,
			system_rhs);	
		std::cout << "Memory consumption estimation of system matrix: " << system_matrix.memory_consumption() << 'B' << std::endl;

}
template <int dim>
void hollowSphere<dim>::solve()
{
	//non symetric matrix
	SolverControl           solver_control(4000, 1e-5);
	SolverGMRES<>              solver(solver_control);
	solver.solve(system_matrix, solution, system_rhs,
		PreconditionIdentity());
	std::ofstream file{ "solution_archive.txt"};
	boost::archive::text_oarchive oa(file);
	solution.save(oa, 1);
	//constraints.distribute(solution);
}
template <int dim>
void hollowSphere<dim>::output_results(char*fileNamecsv) const//post processing
{

	//write csv for pycharm analysis  
	std::ofstream output(fileNamecsv);
	output << "Point:0" << ',' << "Point:1" << ',';
	if (dim == 3)output << "Point:2" << ',';
	output << "x_displacement" << ',' << "y_displacement";
	if (dim == 3)output<<',' << "z_displacement";
	output << std::endl;
	for (auto item = PointMap.begin(); item != PointMap.end();item++) {
		int i = item->first;
		Point<dim> currentPoint = item->second;
		output << currentPoint(0) << ',' << currentPoint(1)<<',';
		if (dim == 3)output << currentPoint(2)<<',';
		output << solution[i] << ',' << solution[i+1];
		if (dim == 3)output << ','<<solution[i+2];
		output << std::endl;
	}
	output.close();
	std::string stressOutput(fileNamecsv);
	stressOutput.replace(stressOutput.length()-4, stressOutput.length()-1, "2.csv");
	std::ofstream output2(stressOutput.c_str());
	output2 << "Point:0" << ',' << "Point:1" << ',';
	if (dim == 3)output2 << "Point:2" << ',';
	output2 << "tensor_xx" << ',' << "tensor_xy"<<','<<"tensor_yy";
	if (dim == 3)output2 << ',' << "tensor_xz"<<','<<"tensor_yz"<<','<<"tensor_zz";
	output2 << std::endl;
	for (auto i = StressMap.begin(); i != StressMap.end();i++) {
		Point<dim> currentPoint = i->first;
		output2 << currentPoint(0) << ',' << currentPoint(1) << ',';
		if (dim == 3)output2 << currentPoint(2) << ',';
		Tensor<2, dim> currentTensor = i->second;
		output2 << currentTensor[0][0] << ',' << currentTensor[0][1] << ',' << currentTensor[1][1];
		if (dim == 3)output2 << ',' << currentTensor[0][2]<<','<< currentTensor[1][2]<<','<<currentTensor[2][2];
		output2 << std::endl;
	}
	output2.close();
	//wrtie displacement field to vtk
	DataOut<3> data_out;
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
	data_out.add_data_vector(solution, solution_names);
	data_out.build_patches();
	std::string vtkFileName(fileNamecsv);
	vtkFileName.replace(vtkFileName.length() - 3, vtkFileName.length() - 1, "vtk");
	std::ofstream output3(vtkFileName);
	data_out.write_vtk(output3);
	return;
}
}
int main(int argc,char** argv)
{
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
	double inner_radius, outer_radius, outer_pressure, inner_pressure, YoungModulus, PoissonRatio;
	int elementType, quadratureCnt;
	bool useSavedSolution;
	int initializationCnt = 0;
	char buffer[20];
	while (!fin.eof()) {
		fin >> buffer;
		if (strcmp(buffer,"inner_radius")==0) {
			fin >> inner_radius;
			initializationCnt= initializationCnt | 1;
		}
		else if (strcmp(buffer,"outer_radius")==0) {
			fin >> outer_radius;
			initializationCnt = initializationCnt | 2;
		}
		else if (strcmp(buffer,"outer_pressure")==0){
			fin >> outer_pressure;
			initializationCnt = initializationCnt | 4;
		}
		else if (strcmp(buffer,"inner_pressure")==0){
			fin >> inner_pressure;
			initializationCnt = initializationCnt | 8;
		}
		else if (strcmp(buffer,"YoungModulus")==0){
			fin >> YoungModulus;
			initializationCnt = initializationCnt | 16;
		}
		else if (strcmp(buffer,"PoissonRatio")==0){
			fin >> PoissonRatio;
			initializationCnt = initializationCnt | 32;
		}
		else if (strcmp(buffer,"elementType")==0){
			fin >> elementType;
			initializationCnt = initializationCnt | 64;
		}
		else if (strcmp(buffer,"quadratureCnt")==0){
			fin >> quadratureCnt;
			initializationCnt = initializationCnt | 128;
		}
		else if(strcmp(buffer,"useSavedSolution")==0){
			fin >> useSavedSolution;
			initializationCnt = initializationCnt | 256;
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
			std::cerr << "outer_pressure is missing" << std::endl;
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
		else {
			std::cout << "parse input parameter file successfully." << std::endl;
		}
	}
	hollowSphere<3> myhollowSphere(argv[1],inner_radius,outer_radius,outer_pressure,inner_pressure,YoungModulus,PoissonRatio,elementType,quadratureCnt);
	myhollowSphere.run(useSavedSolution);//use saved solution
	
	return 0;
}

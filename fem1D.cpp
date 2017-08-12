#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#define _USE_MATH_DEFINES
#include <math.h>
#include<deal.II\lac\vector.h>
#include<boost\archive\binary_oarchive.hpp>
#include<deal.II\lac\sparse_matrix.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/identity_matrix.h>
/** namespace mySolverClass contains the parameter to solve the BC value probelm
 *  \f$ -\epsilon f''(x)+\epsilon \pi^2 f(x) +f'(x)=\sin \pi x \f$
 *  This equation appears when solving practice5, and has analytical solution.
 * For more info, see <a href="../PDE_Numerical_Five.pdf">PDE_Numerical_Five.pdf</a> 
 */
namespace mySolverClass{
	const int library_offset = 1;
	double epsilon = 0.1;
	int n = 10;
	const int version = 1;
	int iteratitionTime = 1000;
	double errorLimit = 1.e-5;
	typedef dealii::Vector<double> myVector;
	typedef dealii::SparseMatrix<double> myMatrix;
	typedef dealii::SparsityPattern mySparsityPattern;
	typedef dealii::SolverGMRES<>  mySolver;
	typedef dealii::SolverControl mySolverControl;

}
int main(int argc,char** argv) {
	//step 0: given x_0,x_1,...x_n
	if (argc > 1) {//second argument is epsilon
		mySolverClass::epsilon = atof(argv[1]);
	}
	if (argc > 2) {//third argument, refinement
		mySolverClass::n = atoi(argv[2]);
	}
	if (argc > 3) {
		mySolverClass::iteratitionTime = atoi(argv[3]);
	}
	double* x= new double[mySolverClass::n+1];
	for (int i = 0; i <= (mySolverClass::n/2); i++) {
		x[i] = i*(1.0-5*mySolverClass::epsilon) / (mySolverClass::n / 2);
	}
	for (int i = (mySolverClass::n / 2)+1; i <= (mySolverClass::n); i++) {
		x[i] = x[(mySolverClass::n / 2)]+ 5*mySolverClass::epsilon*(i- (mySolverClass::n / 2))*
			1.0 / (mySolverClass::n/2);
	}

	//first we decleare RHS
	double* b_array = new double[mySolverClass::n];
	for (int i = 1; i < mySolverClass::n; i++) {
		b_array[i]=((x[i]-x[i+1])*std::sin(M_PI*x[i-1])+(x[i+1]-x[i-1])*std::sin(M_PI*x[i])
			+(x[i-1]-x[i])*std::sin(M_PI*x[i+1]))/(x[i-1]-x[i])/(x[i]-x[i+1])/pow(M_PI,2);
	}
	mySolverClass::myVector b(b_array+1,b_array+mySolverClass::n);
	//print the right hand side info
	std::cout << "b size:" << b.size() << std::endl;
	//b.print(std::cout,3,false);

    //then we initialize the sparse matrix
	std::vector<std::map<int, double>> vector_for_sparse_matrix;
	std::vector<std::vector<unsigned int> > column_indices(mySolverClass::n-1);
	for (int j = 1; j < mySolverClass::n; j++) {
		std::map<int, double> tmp_map;
		tmp_map.insert(std::pair<int, double>(j - mySolverClass::library_offset, mySolverClass::epsilon*(1 / (x[j] - x[j - 1]) + 1 / (x[j + 1] - x[j]) +
			pow(M_PI, 2) / 3 * (x[j + 1] - x[j - 1])
			)
			));
		column_indices[j - mySolverClass::library_offset].push_back(j - mySolverClass::library_offset);
		if (j < mySolverClass::n - 1) {
		tmp_map.insert(std::pair<int, double>(j + 1 - mySolverClass::library_offset, mySolverClass::epsilon*(-1 / (x[j + 1] - x[j]) +
			pow(M_PI, 2) / 6 * (x[j + 1] - x[j])
			) + 0.5
			));
		column_indices[j - mySolverClass::library_offset].push_back(j+1 - mySolverClass::library_offset);
		}
		if (j > 1) {
			tmp_map.insert(std::pair<int, double>(j - 1 - mySolverClass::library_offset, mySolverClass::epsilon*(-1 / (x[j] - x[j - 1]) +
				pow(M_PI, 2) / 6 * (x[j] - x[j - 1])
				) - 0.5
				));
			column_indices[j - mySolverClass::library_offset].push_back(j- 1 - mySolverClass::library_offset);
		}
		vector_for_sparse_matrix.push_back(tmp_map);
	}
	mySolverClass::myMatrix* A = new mySolverClass::myMatrix;
	mySolverClass::mySparsityPattern As = mySolverClass::mySparsityPattern();
	As.copy_from(mySolverClass::n - 1, mySolverClass::n - 1, column_indices.begin(), column_indices.end());//create entry
	A->reinit(As);//tri-diagnal;
	A->copy_from(vector_for_sparse_matrix.begin(), vector_for_sparse_matrix.end());//add value
    //print the matrix info
	std::cout << "matrix info\n";
	std::cout << "num of rows: " << A->m() << std::endl;
	std::cout << "num of columns: " << A->n() << std::endl;
	std::cout << "non-zero element counts: " << A->n_nonzero_elements() << std::endl;
	//A->print(std::cout);
	//save data
	//std::ofstream ofs("matrixData.dat");
	//boost::archive::binary_oarchive oa(ofs);
	//b.save(oa, mySolverClass::version);
	//std::cout << "b is serialized."<< std::endl;

	//initialize the solution vector
	mySolverClass::myVector y(mySolverClass::n-1);

    //initialize the solver class
	mySolverClass::mySolverControl* control = new mySolverClass::mySolverControl(
		mySolverClass::iteratitionTime,mySolverClass::errorLimit);
	//mySolverClass::myAdditionalData solver_param = mySolverClass::myAdditionalData();
	mySolverClass::mySolver solver(*control);
	
	solver.solve(*A,y,b,dealii::IdentityMatrix(y.size()));

	//output the solution
	//save data
	std::ofstream ofs("vector.csv");
	ofs.precision(4);
	for (int i = 0; i < y.size(); i++) {
		ofs << x[i+1] << ',' << y[i] << '\n';
	}
	//boost::archive::binary_oarchive oa(ofs);
	//y.print(ofs,4,false,false);
	ofs.close();
	//std::cout << "b is serialized."<< std::endl;
	std::cout << "The solution is saved to vector.csv\n";
	//y.print(std::cout);
	delete[] x;
	delete[] b_array;
	delete A;
	delete control;
	//Matrix
	//SolverGMRES
	return 0;
}
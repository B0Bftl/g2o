//
// @author Lukas Schneider, 18.03.2019.
//
#include <Eigen/StdVector>
#include <Eigen/SparseCholesky>
#include <iostream>
#include <stdint.h>

#include <sstream>
#include <string>


#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/jacobi_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/eigen/linear_solver_pcg_eigen.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include <fstream>

using namespace Eigen;
using namespace std;


template <typename Derived>
void readVector(const string& filename, Eigen::DenseBase<Derived>& _vector) {
	Eigen::VectorXd& vector = static_cast<Eigen::VectorXd&>( _vector);

	std::ifstream infile(filename);

    std::string line;

	getline(infile, line);

	istringstream iss(line);

	int rows;
	iss >> rows;

	vector.resize(rows);

	int currentRow = 0;
	while (std::getline(infile, line))
	{
		istringstream iss(line);
		iss >> vector.coeffRef(currentRow);
		++currentRow;
	}
}

template <typename Derived>
void readMatrix(const string& filename, Eigen::SparseMatrixBase<Derived>& _matrix) {

	Eigen::SparseMatrix<number_t>& matrix = static_cast<Eigen::SparseMatrix<number_t>&>( _matrix);

	std::ifstream infile(filename);

  std::string line;

  getline(infile, line);

  istringstream iss(line);

  int rows;
  int cols;
  iss >> rows;
  iss >> cols;

  matrix.resize(rows,cols);

  int currentRow = 0;
  number_t value;
  while (std::getline(infile, line))
  {
    istringstream iss(line);
    for (int col = 0; col<cols; ++col) {
      iss >> value;
      if (value != 0)
        matrix.coeffRef(currentRow, col) = value;
    }
    ++currentRow;
  }
}

template <typename Derived>
void printMatrix(const Eigen::MatrixBase<Derived>& matrix, std::string name, bool compact = true) {
	std::cout << "printing " << name << std::endl << "----------------------------" << std::endl;
	if(compact) {
		for (int j = 0; j < matrix.rows(); ++j) {
			for (int k = 0; k < matrix.cols(); ++k) {
				std::cout  << "   " << matrix.coeff(j,k);
			}
			std::cout << std::endl;
		}
	} else {
		for (int j = 0; j < matrix.cols(); ++j) {
			for (int k = 0; k < matrix.rows(); ++k) {
				std::cout <<  k << " x " <<  j << ": " << matrix.coeff(k,j) << std::endl;
			}
		}
	}
	std::cout << "----------------------------" << std::endl << std::endl;
}

template <typename Derived>
void printMatrix(const Eigen::SparseMatrixBase<Derived>& matrix, std::string name, bool compact = true) {
	Eigen::MatrixXd denseMat = matrix;
	printMatrix(denseMat, name, compact);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    cout << endl;
    cout << "Please type: " << endl;
    cout << "ba_benchmark [FILENAME_MATRIX] [FILENAME_VECTOR] [NUM_CAMS] [NUM_POINTS] [EDGE_INFO] [ITERATIONS] [STATFILE]" << endl;
    cout << endl;
    cout << "FILENAME_MATRIX: File to load matrix from" << endl;
    cout << "FILENAME_VECTOR: File to load vector b from" << endl;
    cout << "ITERATIONS: number of iterations" << endl;
    cout << "STATFILE: File to save stats to" << endl;
    cout << endl;
    cout << endl;
    exit(0);
  }

  SparseMatrix<number_t> mat;
  VectorXd b;


	std::string matFile;
  std::string vecFile;

  //int iterations = 5;
  std::string statsFile = "";


  if (argc < 4) {
    std::cerr << "No Files given." << std::endl;
    return -1;
  }

  matFile = (argv[1]);
  vecFile = (argv[2]);

  int numCams = atoi(argv[3]);
  int numPoints = atoi(argv[4]);

  readMatrix(matFile, mat);
  readVector(vecFile, b);
  VectorXd x(b.rows());

  printMatrix(mat, "matrix");
  printMatrix(b, "b");

  std::unique_ptr<g2o::JacobiSolver_6_3::LinearSolverType> linearSolverPCG = g2o::make_unique<g2o::LinearSolverPCGEigen<g2o::JacobiSolver_6_3::PoseMatrixType>>();

  linearSolverPCG->solve(mat, x.data() ,b.data(), numCams, numPoints, 2, 6, 3, 100, 0);

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<number_t >> cholSolver;
  SparseMatrix<number_t> hessian = mat.transpose() * mat;
  cholSolver.compute(hessian);
  VectorXd xChol = cholSolver.solve(b);

  std::cout << "norm of difference: " << (x-xChol).norm() << std::endl;



}



// g2o - General Graph Optimization
// Copyright (C) 2011 R. Kuemmerle, G. Grisetti, W. Burgard
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef G2O_LINEAR_SOLVER_PCG_EIGEN_H
#define G2O_LINEAR_SOLVER_PCG_EIGEN_H

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/SparseCore>


#include "g2o/core/linear_solver.h"
#include "g2o/core/batch_stats.h"
#include "g2o/stuff/timeutil.h"

#include <iostream>
#include <vector>
#include <Eigen/src/Core/Matrix.h>

namespace g2o {

/**
 * \brief linear solver which uses PCG.
 *
 * Has no dependencies except Eigen. Hence, should compile almost everywhere
 * without to much issues. Performance should be similar to CSparse, I guess.
 */
template <typename MatrixType>
class LinearSolverPCGEigen: public LinearSolver<MatrixType>
{
  public:
    typedef Eigen::SparseMatrix<number_t, Eigen::ColMajor> SparseMatrix;
    typedef Eigen::Triplet<number_t> Triplet;

  public:
    LinearSolverPCGEigen() :
      LinearSolver<MatrixType>(),
      _init(true), _writeDebug(false)
    {
    }

    virtual ~LinearSolverPCGEigen()
    {
    }

    virtual bool init()
    {
      _init = true;
      return true;
    }


    bool solve(const SparseBlockMatrix<MatrixType>& A, number_t* x, number_t* b) {
      (void) A;
      (void) x;
      (void) b;
      return false;
    }

    /**
     * Solve System Ax = b for x
     * @param Reference to A, sparseBlockMatrix in g2o form, system matrix of linear system
     * @param x pointer to parameter array to solve for
     * @param b pointer to array containing values to solve for
     * @return true if solving was successful, false otherwise
     */
    bool solve(Eigen::SparseMatrix<number_t>& J, number_t* x, number_t* b, int _numCams, int _numPoints,
               int _rowDim, int _colDimCam, int _colDimPoint, SparseOptimizer* optimizer, number_t lambda)
    {
	    number_t time = get_monotonic_time();
	    (void) _rowDim;
      _optimizer = optimizer;
      // Get Refs to x and b
      VectorX::MapType xVec(x, J.cols());
      VectorX::ConstMapType bVec(b, J.cols());

        // Compute Preconditioner R_inv, store in coeffR
      const Eigen::Ref<const Eigen::SparseMatrix<number_t>> Jc_tmp = J.leftCols(_numCams * _colDimCam);
      const Eigen::Ref<const Eigen::SparseMatrix<number_t>> Jp_tmp = J.rightCols(_numPoints * _colDimPoint);

      std::vector<Eigen::Triplet<number_t >> coeffR;
      coeffR.resize(static_cast<unsigned long>(_numCams * _colDimCam * _colDimCam + _numPoints * _colDimPoint * _colDimPoint));
      number_t timeQR = get_monotonic_time();

      computeRc_inverse(J, _numCams, coeffR);
      std::cout << "QRc " <<  get_monotonic_time() - timeQR << std::endl;

      timeQR = get_monotonic_time();
      computeRp_inverse(Jp_tmp, _colDimPoint, _numCams, coeffR, lambda);
      std::cout << "QRp " <<  get_monotonic_time() - timeQR << std::endl;

      timeQR = get_monotonic_time();
      Eigen::SparseMatrix<number_t> R_inv(J.cols(), J.cols());
      R_inv.setFromTriplets(coeffR.begin(), coeffR.end());
      std::cout << "QR matr" <<  get_monotonic_time() - timeQR << std::endl;

	    coeffR.clear();

      // TODO: here we apply precond to the whole system
      // apply Preconditioning with R_inv to J and b
      timeQR = get_monotonic_time();
      Eigen::SparseMatrix<number_t> _precondJ = J * R_inv;
      Eigen::VectorXd _precond_b = R_inv.transpose() * bVec;
      std::cout << "Apply QR " <<  get_monotonic_time() - timeQR << std::endl;

      // get Jc and Jp from preconditioned J
      const Eigen::Ref<const Eigen::SparseMatrix<number_t>> Jc = _precondJ.leftCols(_numCams * _colDimCam);
      const Eigen::Ref<const Eigen::SparseMatrix<number_t>> Jp = _precondJ.rightCols(_numPoints * _colDimPoint);

      // Map Vector x in Camera and Position Part. Writing to xC/xP writes to x
      VectorX::MapType xC(x, _numCams * _colDimCam);
      VectorX::MapType xP(x + _numCams * _colDimCam, _numPoints * _colDimPoint);
        // Reference b
      Eigen::Ref<VectorX> bC = _precond_b.segment(0, _numCams * _colDimCam);
      Eigen::Ref<VectorX> bP = _precond_b.segment( _numCams * _colDimCam, _numPoints * _colDimPoint);

      // Preconditioning is complete.
      // Initialize Algorithm.

      number_t e_2 = 0;
      number_t e_1 = 0;
      // previous x
      VectorX xC_diff(xC.rows());
      xC_diff.setZero();
      xC.setZero();

      xP = bP;
      xC.setZero();
      VectorX r(xVec.rows());

      Eigen::Ref<VectorX> rC = r.segment(0, _numCams * _colDimCam);
      Eigen::Ref<VectorX> rP = r.segment(_numCams* _colDimCam , _numPoints * _colDimPoint);
      rC = bC;
      rC.noalias() -= Jc.transpose() * (Jp * xP);
      //rC.noalias() = bC - (Jc.transpose() * (Jp * xP));
      rP.setZero();


      // Previous r
      VectorX r_1(xVec.rows());
      Eigen::Ref<VectorX> rC_1 = r_1.segment(0, _numCams * _colDimCam);
      Eigen::Ref<VectorX> rP_1 = r_1.segment(_numCams* _colDimCam , _numPoints * _colDimPoint);

      rC_1 = rC;
      rP_1 = rP;


      number_t q = 1;
      number_t q_1 = 1;
      number_t eta = 0.1; //TODO: eta value benchmarking

      size_t maxIter = J.rows();

      //maxIter = 200;
      size_t iteration = 0;
      bool isEven = false;

      // compute Initial error

      number_t scaledInitialError = eta * rC.dot(rC);

      number_t r_dot_r = scaledInitialError;
      number_t r1_dot_r1 = scaledInitialError;

	  std::cout << "preptime " <<  get_monotonic_time() - time << std::endl;

	  time = get_monotonic_time();

	  for (iteration = 1; iteration < maxIter + maxIter%2; ++iteration) {

	  	isEven = !static_cast<bool>(iteration % 2);
        if(isEven && r_dot_r + r1_dot_r1 < scaledInitialError)
        	break;

        q_1 = q;
        q = 1 - e_1;
        if (!isEven) {
            // odd
            rP.noalias() = Jp.transpose()  * (Jc* rC);
            rP -= (e_1 * rP_1);
            rP *= (1/q);
	        rP_1 = rP;
	        rC.setZero();
        } else {
            // even
            rC.noalias() = Jc.transpose() * (Jp * rP);
            rC -= (e_1 * rC_1 );
            rC *= (1/q);
	        rP.setZero();
        }
        e_2 = e_1;

	    r_dot_r =  r.dot(r);
	    r1_dot_r1 = r_1.dot(r_1);
        e_1 = q *( r_dot_r / r1_dot_r1);

        if(isEven) {
            //even
            xC_diff = e_1 * e_2 * xC_diff; //aliasing needed
            xC_diff += rC_1;
            xC_diff *= (1/(q * q_1));
	        rC_1 = rC;
	        xC = xC_diff + xC;
        }

      }
    //retrieve  xP
    xP.noalias() = bP - rP;
	xP.noalias() -=  Jp.transpose() * (Jc * xC);
	//xP.noalias() = bP - (Jp.transpose() * (Jc * xC)) - rP;

      xVec = R_inv * xVec;

      std::cout << "Looptime " <<  get_monotonic_time() - time << " with "<< iteration << " iterations" << std::endl;


      return true;
    }

    //! write a debug dump of the system matrix if it is not SPD in solve
    virtual bool writeDebug() const { return _writeDebug;}
    virtual void setWriteDebug(bool b) { _writeDebug = b;}

  protected:
    bool _init;
    bool _writeDebug;
    SparseOptimizer* _optimizer;

    void testing () {
    	std::cout << "testing" << std::endl;

    	std::vector<Eigen::Triplet<number_t>> T;

    	T.emplace_back(0,0,3);
	    T.emplace_back(0,1,5);
	    T.emplace_back(1,0,2);
	    T.emplace_back(1,1,3);

	    T.emplace_back(2,2,4);
	    T.emplace_back(2,3,2);
	    T.emplace_back(3,2,1);
	    T.emplace_back(3,3,1);

	    T.emplace_back(4,0,4);
	    T.emplace_back(4,1,4);
	    T.emplace_back(5,0,2);
	    T.emplace_back(5,1,0);

	    T.emplace_back(6,4,3);
	    T.emplace_back(6,5,1);
	    T.emplace_back(7,4,0);
	    T.emplace_back(7,5,4);

	    T.emplace_back(8,0,6);
	    T.emplace_back(9,1,6);
	    T.emplace_back(10,2,6);
	    T.emplace_back(11,3,6);
	    T.emplace_back(12,4,6);
	    T.emplace_back(13,5,6);

	    Eigen::SparseMatrix<number_t> mat (8+6,6);

	    mat.setFromTriplets(T.begin(), T.end());

	    Eigen::MatrixXd dense = mat;

	    Eigen::SparseMatrix<number_t> R1 = computeRc_inverse(mat,2);


	    //Eigen::SparseMatrix<number_t> R1 = computeRp_inverse(mat,2,0);
	    //Eigen::SparseMatrix<number_t> R2 = computeR_inverse(mat,0,0,0);



    }


    template <typename Derived>
    void computeRc_inverse(Eigen::SparseMatrixBase<Derived>& _matrix, int numCams, std::vector<Eigen::Triplet<number_t>>& coeffR) {
		// qr decomp
	    //Eigen::SparseMatrix<number_t >& matrix = static_cast<Eigen::SparseMatrix<number_t >&> (_matrix);
	    //const size_t dimRBlock = 6;
		number_t  timeSpentQR= 0;

	    //Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(currentBlock);
	    //Eigen::Matrix<number_t, 6, 6> inv;

	    long base = 0;
	    std::vector<number_t> coeffBlock;
#ifdef G2O_OPENMP
	    # pragma omp parallel default (shared) firstprivate(base,timeSpentQR, coeffBlock)
	    {
#endif
		    Eigen::Matrix<number_t, 6, 6> inv;
		    Eigen::SparseMatrix<number_t >& matrix = static_cast<Eigen::SparseMatrix<number_t >&> (_matrix);
		    coeffBlock.resize(6*2*_optimizer->maxDegree + 6 * 6);
#ifdef G2O_OPENMP
		#pragma omp for schedule(guided)
#endif
		    for (int k=0; k < numCams * 6; k += 6) {
			    Eigen::Index currentRow = 0;
			    size_t blocksize = 0;
			    //std::vector<number_t> coeffBlock;
			    Eigen::SparseMatrix<number_t>::InnerIterator itA(matrix,k);
			    Eigen::SparseMatrix<number_t>::InnerIterator itB(matrix,k + 1);
			    Eigen::SparseMatrix<number_t>::InnerIterator itC(matrix,k + 2);
			    Eigen::SparseMatrix<number_t>::InnerIterator itD(matrix,k + 3);
			    Eigen::SparseMatrix<number_t>::InnerIterator itE(matrix,k + 4);
			    Eigen::SparseMatrix<number_t>::InnerIterator itF(matrix,k + 5);
			    while (itA || itB || itC || itD ||itE || itF) {

				    currentRow = std::min({itA ? itA.row() : INT_MAX, itB ? itB.row() : INT_MAX,
				                              itC ? itC.row() : INT_MAX, itD ? itD.row() : INT_MAX,
				                              itE ? itE.row() : INT_MAX, itF ? itF.row() : INT_MAX});

				    if(currentRow == INT_MAX) break; // All iterators done TODO: break?

				    //Add coefficients of currentRow, 0 if not in column
				    coeffBlock[6*blocksize] = ((itA && itA.row() == currentRow) ? itA.value() : 0);
				    coeffBlock[6*blocksize + 1] = ((itB && itB.row() == currentRow) ? itB.value() : 0);
				    coeffBlock[6*blocksize + 2] = ((itC && itC.row() == currentRow) ? itC.value() : 0);
				    coeffBlock[6*blocksize + 3] = ((itD && itD.row() == currentRow) ? itD.value() : 0);
				    coeffBlock[6*blocksize + 4] = ((itE && itE.row() == currentRow) ? itE.value() : 0);
				    coeffBlock[6*blocksize + 5] = ((itF && itF.row() == currentRow) ? itF.value() : 0);


				    if (itA && itA.row() == currentRow) ++itA;
				    if (itB && itB.row() == currentRow) ++itB;
				    if (itC && itC.row() == currentRow) ++itC;
				    if (itD && itD.row() == currentRow) ++itD;
				    if (itE && itE.row() == currentRow) ++itE;
				    if (itF && itF.row() == currentRow) ++itF;
				    ++blocksize;
			    }
			    Eigen::Map<Eigen::Matrix<number_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> _currentBlock(coeffBlock.data(), blocksize, 6);
				Eigen::MatrixXd currentBlock = _currentBlock;

				Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(currentBlock);
			    qr.compute(currentBlock);
			    inv = static_cast<Eigen::MatrixXd>(qr.matrixQR().triangularView<Eigen::Upper>()).block<6,6>(0,0);
			    // get inverse
			    inv = inv.inverse().eval();
			    base = (k/6) * inv.cols() * inv.cols();

			    for (int j = 0; j < inv.cols();++j) {
				    for (int i = 0; i < inv.cols();++i) {
					    coeffR[base++] = Eigen::Triplet<number_t>(k + i, k + j, inv.coeff(i,j));
				    }
			    }
		    }
		    std::cout << "QR C only: " << timeSpentQR << std::endl;
#ifdef G2O_OPENMP
	    } //close parallel region
#endif
    }


    template <typename Derived>
    void computeRp_inverse(const Eigen::SparseMatrixBase<Derived>& matrix, int colDim, int numCams, std::vector<Eigen::Triplet<number_t>>& coeffR, number_t lambda) {

    	int rowDim = 3;

#ifdef G2O_OPENMP
# pragma omp parallel default (shared)
	    {
#endif
		    int rowOffset = 0;
		    int blockSize = 0;
		    const OptimizableGraph::Vertex *v;
		    long base = 0;
		    Eigen::Matrix<number_t, 3, 3> inv;

#ifdef G2O_OPENMP
#pragma omp for schedule(guided)
#endif
		    for (int i = 0; i < static_cast<int>(_optimizer->indexMapping().size() - numCams); ++i) {
			    v = static_cast<const OptimizableGraph::Vertex *>(_optimizer->indexMapping()[i + numCams]);
			    // current block size dependent of edges
			    blockSize = v->activeEdgeCount * 2;
			    rowOffset = v->acumulativeEdgeOffset * 2;
			    Eigen::MatrixXd currentBlock(blockSize + colDim, colDim);
			    // get current block. Row offset based on previous blocks, col offset on position in jacobian
			    currentBlock.topRows(blockSize) = matrix.block(rowOffset, i * colDim, blockSize, colDim);
			    // attach lambda scaling
			    currentBlock.bottomRows(colDim) = Eigen::MatrixXd::Identity(colDim, colDim) * lambda;
			    // initialize & compute qr in place
			    Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(currentBlock);
			    qr.compute(currentBlock);

			    inv = static_cast<Eigen::MatrixXd>(qr.matrixQR().triangularView<Eigen::Upper>()).block<3, 3>(0, 0);
			    //inv = tmp.topRows(tmp.cols());
			    // get inverse
			    inv = inv.inverse().eval();
			    base = numCams * 36 + 9 * i;
			    for (int j = 0; j < inv.cols(); ++j) {
				    for (int k = 0; k < inv.cols(); ++k) {
					    coeffR[base++] = Eigen::Triplet<number_t>(numCams * 6 + k + rowDim * i,
					                                              numCams * 6 + j + rowDim * i, inv.coeff(k, j));
				    }
			    }
		    }

#ifdef G2O_OPENMP
	    }
#endif

    }

	template <typename Derived>
	void printMatrix(const Eigen::MatrixBase<Derived>& matrix, std::string name, bool compact = false) {
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
	void printMatrix(const Eigen::SparseMatrixBase<Derived>& matrix, std::string name, bool compact = false) {
		Eigen::MatrixXd denseMat = matrix;
		printMatrix(denseMat, name, compact);
	}


        template <typename Derived>
    Eigen::SparseMatrix<number_t> computeR_inverse(const Eigen::SparseMatrixBase<Derived>& matrix) {

        Eigen::SparseQR<Eigen::SparseMatrix<number_t>, Eigen::COLAMDOrdering<int> > qr;

        qr.analyzePattern(matrix);
        qr.factorize(matrix);

        if(qr.info() != Eigen::Success)
            std::cout << "QR Decomposition failed." << std::endl;

        Eigen::SparseMatrix<number_t, Eigen::RowMajor> tmp = qr.matrixR();
        Eigen::SparseMatrix<number_t, Eigen::ColMajor> R = tmp.topRows(tmp.cols());

        Eigen::SparseLU<Eigen::SparseMatrix<number_t>> lu;
        lu.analyzePattern(R);
        lu.factorize(R);

        if(lu.info() != Eigen::Success)
            std::cout << "LU Decomposition failed." << std::endl;

        Eigen::SparseMatrix<number_t> I(R.cols(), R.cols());
        I.setIdentity();

        Eigen::SparseMatrix<number_t> R_inv =  lu.solve(I);

        return R_inv;
    }

};

} // end namespace

#endif

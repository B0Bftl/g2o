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
#include <Eigen/StdVector>
#include <g2o/core/batch_stats.h>

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
               int _rowDim, int _colDimCam, int _colDimPoint, number_t lambda, number_t _eta)
    {
      (void) _rowDim;
      // Get Refs to x and b
      VectorX::MapType xVec(x, J.cols());
      VectorX::ConstMapType bVec(b, J.cols());

      G2OBatchStatistics* globalStats = G2OBatchStatistics::globalStats();
	    //if(_writeDebug)
      //	printMatrix(J, "Jacobi");

      //number_t time_total = get_monotonic_time();
      // Compute Preconditioner R_inv, store in coeffR
      Eigen::Ref<Eigen::SparseMatrix<number_t>> Jc_tmp = J.leftCols(_numCams * _colDimCam);
      Eigen::Ref<Eigen::SparseMatrix<number_t>> Jp_tmp = J.rightCols(_numPoints * _colDimPoint);

      number_t timeQR = get_monotonic_time();
      std::vector<Eigen::Triplet<number_t >, Eigen::aligned_allocator<Eigen::Triplet<number_t >>> coeffR;
      coeffR.resize(static_cast<unsigned long>(_numCams * _colDimCam * _colDimCam + _numPoints * _colDimPoint * _colDimPoint));



      //std::cout << "Allocating coeffR: " << get_monotonic_time() - time_R << std::endl;

      //time_R = get_monotonic_time();
      computeRc_inverse(J, _numCams, coeffR);
      //std::cout << "Rc: " << get_monotonic_time() - time_R << std::endl;

      //time_R = get_monotonic_time();
      computeRp_inverse(J, _colDimPoint, _numCams, _numPoints, coeffR, lambda);
      //std::cout << "Rp: " << get_monotonic_time() - time_R << std::endl;

      Eigen::SparseMatrix<number_t> R_inv(J.cols(), J.cols());
      R_inv.setFromTriplets(coeffR.begin(), coeffR.end());

      if(globalStats)
        globalStats->timeQrDecomposition = get_monotonic_time() - timeQR;

      const Eigen::Ref<const Eigen::SparseMatrix<number_t>> Rc = R_inv.topLeftCorner(_numCams * 6,_numCams * 6);
      const Eigen::Ref<const Eigen::SparseMatrix<number_t>> Rp = R_inv.bottomRightCorner(_numPoints * 3,_numPoints * 3);

      //std::cout << "Time for QR total: " << get_monotonic_time() - time_total << std::endl;


      //time_R = get_monotonic_time();
      // apply Preconditioning with R_inv to J and b
      //Eigen::SparseMatrix<number_t> _precondJ = J * R_inv;
      Eigen::VectorXd _precond_b = R_inv.transpose() * bVec;

      // get Jc and Jp from preconditioned J
      const Eigen::Ref<const Eigen::SparseMatrix<number_t>> Jc = J.leftCols(_numCams * _colDimCam);
      const Eigen::Ref<const Eigen::SparseMatrix<number_t>> Jp = J.rightCols(_numPoints * _colDimPoint);

      //std::cout << "Time Applying QR: " << get_monotonic_time() - time_R << std::endl;
      //time_R = get_monotonic_time();

	    // Preconditioning is complete.
      // Initialize Algorithm.

      // Reference b
      const Eigen::Ref<const VectorX> bC = _precond_b.segment(0, _numCams * _colDimCam);
      const Eigen::Ref<const VectorX> bP = _precond_b.segment( _numCams * _colDimCam, _numPoints * _colDimPoint);

      // Map Vector x in Camera and Position Part. Writing to xC/xP writes to x
      VectorX::MapType xC(x, _numCams * _colDimCam);
      VectorX::MapType xP(x + _numCams * _colDimCam, _numPoints * _colDimPoint);

      VectorX s(xVec.rows());
      //VectorX tmp(xVec.rows());


      Eigen::Ref<VectorX> sC = s.segment(0, _numCams * _colDimCam);
      Eigen::Ref<VectorX> sP = s.segment(_numCams* _colDimCam , _numPoints * _colDimPoint);

      xC.setZero();

      xP = bP;
      //Eigen::VectorXd p = _precond_b - _precondJ.transpose() * (_precondJ * xVec);

      // to minimize memory reallocation, use standard tmp vectors for manual aliasing. Need 2 sizes
      Eigen::VectorXd tmpLong(J.rows());
      Eigen::VectorXd tmpShort(xVec.rows());


      //p =_precond_b - R_inv.transpose() *  (J.transpose() * (J * (R_inv * xVec)));
      Eigen::VectorXd p = _precond_b;

      tmpShort.noalias() = R_inv * xVec;
      tmpLong.noalias() = J * tmpShort;
      tmpShort.noalias() = J.transpose() * tmpLong;
      p.noalias() -= R_inv.transpose() * tmpShort;

      s = p;

      number_t beta;

      //Eigen::VectorXd q = J * (R_inv * p);
      tmpShort.noalias() = R_inv * p;
      Eigen::VectorXd q = J * tmpShort;


      long maxIter = J.rows();

      long iteration = 0;
      bool isEven = false;

	  //std::cout << "preptime " <<  get_monotonic_time() - time << std::endl;

	  number_t alpha;


	  number_t gamma = s.dot(s);
	  number_t gamma_old = gamma;


	  // compute Initial error
	  number_t scaledInitialError = _eta * s.dot(s);

	  //std::cout << "Initialisation: " << get_monotonic_time() - time_R << std::endl;
	  //time_R = get_monotonic_time();

	  for (iteration = 0; iteration < maxIter + maxIter%2; ++iteration) {

	  	if (gamma < scaledInitialError)
	  		break;

	  	isEven = !static_cast<bool>(iteration % 2);

	  	alpha = gamma/ q.dot(q);
	  	xVec.noalias() += alpha * p;
	  	if (!isEven) {
	  		//odd
			//sC.noalias() = (-1) * alpha * Rc.transpose() * (Jc.transpose() * q);

		    tmpLong.noalias() = Jc.transpose() * q;
		    sC.noalias() = (-1) * alpha * Rc.transpose() * tmpLong;
			sP.setZero();
	  	} else {
	  		//even
	  		// sP.noalias() = (-1) * alpha * Rp.transpose() * (Jp.transpose() * q);

	  		tmpLong.noalias() = Jp.transpose() * q;
		    sP.noalias() = (-1) * alpha * Rp.transpose() * tmpLong;
	  		sC.setZero();
	  	}
	  	//y = solver.solve(s);

		gamma = s.dot(s);
	  	//gamma = s.dot(y);
	  	beta = gamma/gamma_old;
	  	gamma_old = gamma;
	  	//p = s + beta * p;
	  	tmpShort.noalias() = beta * p;
		p.noalias() = s + tmpShort;

		if(!isEven) {
			//odd
			//q = beta * q + Jc * sC;
			// q.noalias() += Jc * (Rc * sC);
			tmpLong.noalias() = beta * q;
			tmpShort.noalias() = (Rc * sC);
			q.noalias() = tmpLong;
			q.noalias() +=  Jc * tmpShort;
		} else {
			// even
			//q = beta * q + Jp * sP;
			// q.noalias() += Jp * (Rp * sP);
			//q = beta * q;
			tmpLong.noalias() = beta * q;
			tmpShort.noalias() = (Rp * sP);
			q.noalias() = tmpLong;
			q.noalias() += Jp * tmpShort;
		}
      }
	  tmpShort.noalias() = R_inv * xVec;
	  xVec = tmpShort;
	  if (globalStats) {
	    globalStats->iterationsLinearSolver = iteration;
	  }
	  //std::cout << "Time Loop: " << get_monotonic_time() - time_R << " with " << iteration << " iterations." << std::endl;
	  //std::cout << "Time Total: " << get_monotonic_time() - time_total << std::endl;

      return true;
    }




    //! write a debug dump of the system matrix if it is not SPD in solve
    virtual bool writeDebug() const { return _writeDebug;}
    virtual void setWriteDebug(bool b) { _writeDebug = b;}

  protected:
    bool _init;
    bool _writeDebug = false;
    int _maxDegree = -1;

	template <typename Derived>
    void getMaxDegree(Eigen::SparseMatrixBase<Derived>& _matrix, int numCams) {

#ifdef G2O_OPENMP
#pragma omp parallel
    	{
#endif
			int max = -1;
#ifdef G2O_OPENMP
#pragma omp for schedule(static)
#endif
			for (int i = 0; i < numCams; ++i) {
				if (max < (_matrix.col(i*6).nonZeros() - 1) / 2)
					max = (_matrix.col(i*6).nonZeros() - 1) / 2;
			}
#ifdef G2O_OPENMP
#pragma omp critical
#endif
			{
				if (_maxDegree < max) {
					_maxDegree = max;
				}
			}

#ifdef G2O_OPENMP
	}
#endif
    }

    template <typename Derived>
    void computeRc_inverse(Eigen::SparseMatrixBase<Derived>& _matrix, int numCams,
    		std::vector<Eigen::Triplet<number_t>, Eigen::aligned_allocator<Eigen::Triplet<number_t >>>& coeffR) {
		// qr decomp

	    long base = 0;
	    std::vector<number_t, Eigen::aligned_allocator<number_t>> coeffBlock;

	    if(_maxDegree == -1)
	        getMaxDegree(_matrix, numCams);

#ifdef G2O_OPENMP
	    # pragma omp parallel default (shared) firstprivate(base, coeffBlock)
	    {
#endif
		    Eigen::Matrix<number_t, 6, 6> inv;
		    Eigen::SparseMatrix<number_t >& matrix = static_cast<Eigen::SparseMatrix<number_t >&> (_matrix);
		    coeffBlock.resize(6*2*_maxDegree + 6 * 6);
		    //number_t time;
#ifdef G2O_OPENMP
		#pragma omp for schedule(guided)
#endif
		    for (int k=0; k < numCams * 6; k += 6) {
		    	//time = get_monotonic_time();
			    size_t blocksize = 0;
			    //std::vector<number_t> coeffBlock;
			    Eigen::SparseMatrix<number_t>::InnerIterator itA(matrix,k);
			    Eigen::SparseMatrix<number_t>::InnerIterator itB(matrix,k + 1);
			    Eigen::SparseMatrix<number_t>::InnerIterator itC(matrix,k + 2);
			    Eigen::SparseMatrix<number_t>::InnerIterator itD(matrix,k + 3);
			    Eigen::SparseMatrix<number_t>::InnerIterator itE(matrix,k + 4);
			    Eigen::SparseMatrix<number_t>::InnerIterator itF(matrix,k + 5);
			    while (itA && itB && itC && itD && itE && itF && itA.row() == itB.row()) {

			    	assert(itA.row() == itB.row()
			    	&& itA.row() == itC.row()
			    	&& itA.row() == itD.row()
			    	&& itA.row() == itE.row()
			    	&& itA.row() == itF.row());

				    coeffBlock[6*blocksize] = itA.value();
				    coeffBlock[6*blocksize + 1] = itB.value();
				    coeffBlock[6*blocksize + 2] = itC.value();
				    coeffBlock[6*blocksize + 3] = itD.value();
				    coeffBlock[6*blocksize + 4] = itE.value();
				    coeffBlock[6*blocksize + 5] = itF.value();

					++itA;
					++itB;
					++itC;
					++itD;
					++itE;
					++itF;

				    ++blocksize;
			    }
			    // add lambda entries. All other entries are 0 by initialization
			    assert(6*(blocksize + 5) + 5 <= coeffBlock.size());
			    // clear from previous iteration
			    for(unsigned long j = 6* (blocksize); j < 6*(blocksize+5) +5; ++j) coeffBlock[j] = 0;

				coeffBlock[6 * (blocksize++)] = itA.value();
			    coeffBlock[6 * (blocksize++) + 1] = itB.value();
			    coeffBlock[6 * (blocksize++) + 2] = itC.value();
			    coeffBlock[6 * (blocksize++) + 3] = itD.value();
			    coeffBlock[6 * (blocksize++) + 4] = itE.value();
			    coeffBlock[6 * (blocksize++) + 5] = itF.value();

			    Eigen::Map<Eigen::Matrix<number_t, Eigen::Dynamic, 6, Eigen::ColMajor>, 0,Eigen::Stride<1,6> >
			        currentBlock(coeffBlock.data(), blocksize, 6, Eigen::Stride<1,6>(1,6));
				//std::cout << "Fetching Block: " << get_monotonic_time() - time << std::endl;
			    //time = get_monotonic_time();

				//Eigen::HouseholderQR<Eigen::Ref<Eigen::Map<Eigen::Matrix<number_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>>> qr(currentBlock);
			    //qr.compute(currentBlock);
			    inv = static_cast<Eigen::Matrix<number_t, Eigen::Dynamic, 6>>(currentBlock.householderQr().matrixQR().triangularView<Eigen::Upper>()).block<6,6>(0,0);
			    // get inverse

			    inv = (-1) * inv.inverse().eval();
			    base = (k/6) * inv.cols() * inv.cols();

			    for (int j = 0; j < inv.cols();++j) {
				    for (int i = 0; i < inv.cols();++i) {
					    coeffR[base++] = Eigen::Triplet<number_t>(k + i, k + j,inv.coeff(i,j));
				    }
			    }
			    //std::cout << "QR of block: " << get_monotonic_time() - time << std::endl;

		    }
		    //std::cout << "QR C only: " << timeSpentQR << std::endl;
#ifdef G2O_OPENMP
	    } //close parallel region
#endif
    }


    template <typename Derived>
    void computeRp_inverse(Eigen::SparseMatrixBase<Derived>& _matrix, int colDim, int numCams, int numPoints,
    		std::vector<Eigen::Triplet<number_t>, Eigen::aligned_allocator<Eigen::Triplet<number_t >>>& coeffR, number_t lambda) {

    	int rowDim = 3;
    	Eigen::SparseMatrix<number_t >& matrix = static_cast<Eigen::SparseMatrix<number_t >&> (_matrix);


#ifdef G2O_OPENMP
# pragma omp parallel default (shared)
	    {
#endif
		    int rowOffset = 0;
		    int blockSize = 0;
		    int colIndex = 0;
		    long base = 0;
		    Eigen::Matrix<number_t, 3, 3> inv;

#ifdef G2O_OPENMP
#pragma omp for schedule(guided)
#endif
		    for (int i = 0; i < numPoints; ++i) {
				colIndex = i*3 + numCams*6;

			    Eigen::SparseMatrix<number_t>::InnerIterator iterRow(matrix,colIndex);

			    // block starts at first non zero entry and ends at last (-1 because of lambda).
			    rowOffset = iterRow.row();
				blockSize = matrix.col(colIndex).nonZeros() - 1;


			    Eigen::Matrix<number_t, Eigen::Dynamic, 3> currentBlock(blockSize + colDim, colDim);
			    // get current block. Row offset based on previous blocks, col offset on position in jacobian
			    currentBlock.topRows(blockSize) = matrix.block(rowOffset, colIndex, blockSize, colDim);
			    // attach lambda scaling
			    currentBlock.bottomRows(colDim) = Eigen::Matrix<number_t,3,3>::Identity(colDim, colDim) * lambda;

			    // initialize & compute qr in place
			    //Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(currentBlock);
			    //qr.compute(currentBlock);

				Eigen::MatrixXd x;
			    inv = static_cast<Eigen::Matrix<number_t, Eigen::Dynamic, 3>>(currentBlock.householderQr().matrixQR().triangularView<Eigen::Upper>()).block<3, 3>(0, 0);

			    //inv = tmp.topRows(tmp.cols());
			    // get inverse
			    inv =  (-1) * inv.inverse().eval();
			    base = numCams * 36 + 9 * i;
			    for (int j = 0; j < inv.cols(); ++j) {
				    for (int k = 0; k < inv.cols(); ++k) {
					    coeffR[base++] = Eigen::Triplet<number_t>(numCams * 6 + k + rowDim * i,
					                                              numCams * 6 + j + rowDim * i,inv.coeff(k, j));
				    }
			    }
		    }

#ifdef G2O_OPENMP
	    }
#endif

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


	/*
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
	*/
};

} // end namespace

#endif

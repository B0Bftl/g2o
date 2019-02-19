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
    typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> PermutationMatrix;

    // I shouldnt need that
    /**
     * \brief Sub-classing Eigen's SimplicialLDLT to perform ordering with a given ordering
     */
    class CholeskyDecomposition : public Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper>
    {
      public:
        CholeskyDecomposition() : Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper>() {}
        using Eigen::SimplicialLDLT< SparseMatrix, Eigen::Upper>::analyzePattern_preordered;

        void analyzePatternWithPermutation(SparseMatrix& a, const PermutationMatrix& permutation)
        {
          m_Pinv = permutation;
          m_P = permutation.inverse();
          int size = a.cols();
          SparseMatrix ap(size, size);
          ap.selfadjointView<Eigen::Upper>() = a.selfadjointView<UpLo>().twistedBy(m_P);
          analyzePattern_preordered(ap, true);
        }
    };

  public:
    LinearSolverPCGEigen() :
      LinearSolver<MatrixType>(),
      _init(true), _blockOrdering(false), _writeDebug(false)
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
    bool solve(const Eigen::SparseMatrix<number_t>& J, number_t* x, number_t* b, int _numCams, int _numPoints,
               int _rowDim, int _colDimCam, int _colDimPoint, SparseOptimizer* optimizer, number_t lambda)
    {
	    (void) _rowDim;
      _optimizer = optimizer;
      // Get Refs to x and b
      VectorX::MapType xVec(x, J.cols());
      VectorX::ConstMapType bVec(b, J.cols());

        // Compute Preconditioner R_inv
      const Eigen::Ref<const Eigen::SparseMatrix<number_t>> Jc_tmp = J.leftCols(_numCams * _colDimCam);
      const Eigen::Ref<const Eigen::SparseMatrix<number_t>> Jp_tmp = J.rightCols(_numPoints * _colDimPoint);

      std::cout <<" computing Qr" << std::endl;
      std::vector<Eigen::Triplet<number_t >> coeffR;

      computeRc_inverse(Jc_tmp, _colDimCam, coeffR);
      computeRp_inverse(Jp_tmp, _colDimPoint, _numCams, coeffR, lambda);


      Eigen::SparseMatrix<number_t> R_inv(J.cols(), J.cols());
      R_inv.setFromTriplets(coeffR.begin(), coeffR.end());

      std::cout << "computing done" << std::endl;

        // apply Preconditioning with R_inv to J and b
      Eigen::SparseMatrix<number_t> _precondJ = J * R_inv;
      Eigen::VectorXd _precond_b = R_inv.transpose() * bVec;

      // get Jc and Jp from preconditioned J
      //TODO: Make this a REF or MAP
      Eigen::SparseMatrix<number_t> Jc = _precondJ.leftCols(_numCams * _colDimCam);
      Eigen::SparseMatrix<number_t> Jp = _precondJ.rightCols(_numPoints * _colDimPoint);

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
      rC = bC - (Jc.transpose() * (Jp * xP)); // TODO: sign correct?



      // Previous r
      VectorX r_1(xVec.rows());
      Eigen::Ref<VectorX> rC_1 = r_1.segment(0, _numCams * _colDimCam);
      Eigen::Ref<VectorX> rP_1 = r_1.segment(_numCams* _colDimCam , _numPoints * _colDimPoint);

      rC_1 = rC;
      rP_1 = rP;


      number_t q = 1;
      number_t q_1 = 1;
      number_t eta = 0.01;

      size_t maxIter = J.rows();

      //maxIter = 200;
      size_t iteration = 0;
      bool isEven = false;

      // compute Initial error

      number_t scaledInitialError = eta * rC.dot(rC) + bP.dot(bP);

      for (iteration = 0; iteration < maxIter + maxIter%2; ++iteration) {
        isEven = iteration % 2;
        if(isEven && (r.dot(r) + r_1.dot(r_1)) < scaledInitialError )
        	break;

        q_1 = q;
        q = 1 - e_1;
        if (!isEven) {
            // odd
            rP = (1/q) * ( (Jp.transpose()  * (Jc* rC)) - (e_1 * rP_1) );
	        rP_1 = rP;
	        rC.setZero();
        } else {
            // even
            rC = (1/q) * ( (Jc.transpose() * (Jp * rP)) - (e_1 * rC_1 ));
	        rP.setZero();
        }
        e_2 = e_1;
        e_1 = q *( r.dot(r) / r_1.dot(r_1));

        if(isEven) {
            //even
            xC_diff = (1/(q * q_1)) * (rC_1 + (e_1 * e_2 * xC_diff)); //TODO: rC_1 okay? oder verschieben?
	        rC_1 = rC;
	        xC = xC_diff + xC;
        }

      }

      xP = bP - (Jp.transpose() * (Jc * xC)) - rP;
      //retrieve xC, xP

      /*
      std::cout << "iter: " << iteration << std::endl;
        std::cout << "xVector: " << std::endl;
      for (int i = 0; i<J.cols();++i) {
          std::cout << xVec[i] << std::endl;
      }
		*/
      //Eigen::VectorXd approx =  R_inv.transpose() * J.transpose() * J  * R_inv * xVec;
	    Eigen::VectorXd approx =  _precondJ.transpose() * _precondJ * xVec;

      Eigen::VectorXd res = approx - ((-1) * R_inv.transpose() * bVec);

      number_t resAbs = res.dot(res);

      std::cout << "Residual: " << resAbs << std::endl;


      xVec = R_inv * xVec;



      return true;
    }

    //! do the AMD ordering on the blocks or on the scalar matrix
    bool blockOrdering() const { return _blockOrdering;}
    void setBlockOrdering(bool blockOrdering) { _blockOrdering = blockOrdering;}

    //! write a debug dump of the system matrix if it is not SPD in solve
    virtual bool writeDebug() const { return _writeDebug;}
    virtual void setWriteDebug(bool b) { _writeDebug = b;}

  protected:
    bool _init;
    bool _blockOrdering;
    bool _writeDebug;
    SparseOptimizer* _optimizer;
    CholeskyDecomposition _cholesky;

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
    void computeRc_inverse(const Eigen::SparseMatrixBase<Derived>& matrix, int colDim, std::vector<Eigen::Triplet<number_t>>& coeffR) {
		// qr decomp
	    Eigen::SparseQR<Eigen::SparseMatrix<number_t>, Eigen::NaturalOrdering<int> > qr;

	    for (int i = 0; i < matrix.cols(); i += colDim) {
			// retreive current block
		    Eigen::SparseMatrix<number_t> currentBlock = matrix.middleCols(i,colDim);
			// compute QR
		    qr.analyzePattern(currentBlock);
		    qr.factorize(currentBlock);
		    Eigen::MatrixXd R = qr.matrixR().topLeftCorner(qr.rank(), qr.rank());
		    //get reverse
		    R = R.inverse();
			// save in triplet vector
			for (int j = 0; j < R.cols(); ++j) {
				for (int k = 0; k < R.rows(); ++k) {
					coeffR.emplace_back(k + i, j + i, R.coeff(k,j));
				}
			}
    	}

    }

    template <typename Derived>
    void computeRp_inverse(const Eigen::SparseMatrixBase<Derived>& matrix, int colDim, int numCams, std::vector<Eigen::Triplet<number_t>>& coeffR, number_t lambda) {

    	int rowOffset = 0;
    	int rowDim = 3;
	    int blockSize = 0;
	    const OptimizableGraph::Vertex* v;

	    for (int i = 0; i < _optimizer->indexMapping().size() - numCams; ++i) {
            v = static_cast<const OptimizableGraph::Vertex*>(_optimizer->indexMapping()[i + numCams]);
		    // current block size dependent of edges

            blockSize = v->activeEdgeCount * 2;

		    Eigen::MatrixXd currentBlock(blockSize + colDim, colDim);
	        // get current block. Row offset based on previous blocks, col offset on position in jacobian
		    currentBlock.topRows(blockSize) = matrix.block(rowOffset, i * colDim, blockSize, colDim);
			// attach lambda scaling
		    currentBlock.bottomRows(colDim) = Eigen::MatrixXd::Identity(colDim, colDim) * lambda;
		    // initialize & compute qr in place
	        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(currentBlock);
            qr.compute(currentBlock);

            Eigen::MatrixXd tmp = qr.matrixQR().triangularView<Eigen::Upper>();
		    Eigen::MatrixXd inv = tmp.topRows(tmp.cols());
			// get inverse
		    inv = inv.inverse();
			for (int j = 0; j < tmp.cols();++j) {
				for (int k = 0; k < tmp.cols();++k) {
					coeffR.emplace_back(numCams*6 + k + rowDim * i,numCams*6 + j + rowDim * i, inv.coeff(k,j));
				}
			}

		    rowOffset += blockSize;
        }

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


    /**
     * compute the symbolic decompostion of the matrix only once.
     * Since A has the same pattern in all the iterations, we only
     * compute the fill-in reducing ordering once and re-use for all
     * the following iterations.
     */
    void computeSymbolicDecomposition(const SparseBlockMatrix<MatrixType>& A)
    {
      number_t t=get_monotonic_time();
      if (! _blockOrdering) {
        _cholesky.analyzePattern(A);
      } else {
        // block ordering with the Eigen Interface
        // This is really ugly currently, as it calls internal functions from Eigen
        // and modifies the SparseMatrix class
        Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic> blockP;
        {
          // prepare a block structure matrix for calling AMD
          std::vector<Triplet> triplets;
          for (size_t c = 0; c < A.blockCols().size(); ++c){
            const typename SparseBlockMatrix<MatrixType>::IntBlockMap& column = A.blockCols()[c];
            for (typename SparseBlockMatrix<MatrixType>::IntBlockMap::const_iterator it = column.begin(); it != column.end(); ++it) {
              const int& r = it->first;
              if (r > static_cast<int>(c)) // only upper triangle
                break;
              triplets.push_back(Triplet(r, c, 0.));
            }
          }

          // call the AMD ordering on the block matrix.
          // Relies on Eigen's internal stuff, probably bad idea
          SparseMatrix auxBlockMatrix(A.blockCols().size(), A.blockCols().size());
          auxBlockMatrix.setFromTriplets(triplets.begin(), triplets.end());
          typename CholeskyDecomposition::CholMatrixType C;
          C = auxBlockMatrix.selfadjointView<Eigen::Upper>();
          Eigen::internal::minimum_degree_ordering(C, blockP);
        }

        int rows = A.rows();
        assert(rows == A.cols() && "Matrix A is not square");

        // Adapt the block permutation to the scalar matrix
        PermutationMatrix scalarP;
        scalarP.resize(rows);
        int scalarIdx = 0;
        for (int i = 0; i < blockP.size(); ++i) {
          const int& p = blockP.indices()(i);
          int base  = A.colBaseOfBlock(p);
          int nCols = A.colsOfBlock(p);
          for (int j = 0; j < nCols; ++j)
            scalarP.indices()(scalarIdx++) = base++;
        }
        assert(scalarIdx == rows && "did not completely fill the permutation matrix");
        // analyze with the scalar permutation
        _cholesky.analyzePatternWithPermutation(A, scalarP);

      }
      G2OBatchStatistics* globalStats = G2OBatchStatistics::globalStats();
      if (globalStats)
        globalStats->timeSymbolicDecomposition = get_monotonic_time() - t;
    }

};

} // end namespace

#endif

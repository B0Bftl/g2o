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
    bool solve(const Eigen::SparseMatrix<number_t>& A, number_t* x, number_t* b, int _numCams, int _numPoints,
               int _rowDim, int _colDimCam, int _colDimPoint, SparseOptimizer* optimizer)
    {

	    testing();

      _optimizer = optimizer;
        // Get Matrix, x and b
      J =  A;
      VectorX::MapType xVec(x, J.cols());
      VectorX::ConstMapType bVec(b, J.cols());

        // Compute Preconditioner R_inv
      Eigen::SparseMatrix<number_t> Jc_tmp = J.leftCols(_numCams * _colDimCam);
      Eigen::SparseMatrix<number_t> Jp_tmp = J.rightCols(_numPoints * _colDimPoint);

      std::cout <<" computing Qr" << std::endl;


      Eigen::SparseMatrix<number_t> Rc_inv  = computeRc_inverse(Jc_tmp, _colDimCam);
      Eigen::SparseMatrix<number_t> Rp_inv2 = computeRp_inverse(Jp_tmp, _colDimPoint, _numCams);

      //Eigen::SparseMatrix<number_t> Rc_inv = computeR_inverse(Jc_tmp, 0, _numCams, _colDimCam);
      Eigen::SparseMatrix<number_t> Rp_inv = computeR_inverse(Jp_tmp, _numCams, -1, _colDimPoint);
        std::cout << "computing done" << std::endl;

        //TODO: DEBUG

        std::cout << "Matching Size: " << Rp_inv2.cols() << " " << Rp_inv.cols() << "x"<< Rp_inv2.rows() << " " << Rp_inv.rows() << std::endl;

        std::cout << "coeff: " << std::endl;

        for (int j = 0; j < Rp_inv.cols(); ++j) {
        	for (int k = 0; k < Rp_inv.rows(); ++k) {
				std::cout << k << "x" << j << ": " << Rp_inv.coeff(k,j) << " - " << Rp_inv2.coeff(k,j) << std::endl;
        	}
        }


	    Eigen::MatrixXd R_inv_tmp = Eigen::MatrixXd::Zero(Rc_inv.cols() + Rp_inv.cols(), Rc_inv.cols() + Rp_inv.cols());
      R_inv_tmp.setZero();
      R_inv_tmp.topLeftCorner(Rc_inv.cols(), Rc_inv.cols()) = Rc_inv;
      R_inv_tmp.bottomRightCorner(Rp_inv.cols(), Rp_inv.cols()) = Rp_inv;

      Eigen::SparseMatrix<number_t> R_inv = R_inv_tmp.sparseView();

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
    SparseMatrix J;
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

	    Eigen::SparseMatrix<number_t> mat (4,4);

	    mat.setFromTriplets(T.begin(), T.end());

	    Eigen::SparseMatrix<number_t> R1 = computeRp_inverse(mat,2,0);
	    Eigen::SparseMatrix<number_t> R2 = computeR_inverse(mat,0,0,0);

	    std::cout << "Matching Size: " << R1.cols() << " " << R2.cols() << "x"<< R1.rows() << " " << R2.rows() << std::endl;

	    std::cout << "coeff: " << std::endl;

	    for (int j = 0; j < R1.cols(); ++j) {
		    for (int k = 0; k < R1.rows(); ++k) {
			    std::cout << k << "x" << j << ": " << R1.coeff(k,j) << " | " << R2.coeff(k,j) << std::endl;
		    }
	    }


    }

    template <typename Derived>
    Eigen::SparseMatrix<number_t> computeRc_inverse(const Eigen::SparseMatrixBase<Derived>& matrix, int colDim) {

        for (int i = 0; i < _optimizer->indexMapping().size(); ++i) {
            const OptimizableGraph::Vertex* v = static_cast<const OptimizableGraph::Vertex*>(_optimizer->indexMapping()[i]);

            std::cout << "block: 2 x " << matrix.rows() << " starting at: 0 " <<  i * 2 << std::endl ;

        }

        return Eigen::SparseMatrix<number_t>(0,0);

    }

    template <typename Derived>
    Eigen::SparseMatrix<number_t> computeRp_inverse(const Eigen::SparseMatrixBase<Derived>& matrix, int colDim, int numCams) {

    	int rowOffset = 0;
        std::vector<Eigen::Triplet<number_t >> coeffR;

	    for (int i = 0; i < 2; ++i) {

	    //for (int i = 0; i < _optimizer->indexMapping().size() - numCams; ++i) {
            const OptimizableGraph::Vertex* v = static_cast<const OptimizableGraph::Vertex*>(_optimizer->indexMapping()[i + numCams]);
		    int offs = v->activeEdgeCount;
		    offs = 2;

	        Eigen::MatrixXd block(offs * 2 + colDim, colDim);
	        block.topRows(offs * 2) = matrix.block(rowOffset, i * colDim, offs * 2, colDim);
			block.bottomRows(colDim) = Eigen::MatrixXd::Identity(colDim, colDim) * 2;
            Eigen::ColPivHouseholderQR<Eigen::Ref<Eigen::MatrixXd>> qr(block);
            qr.compute(block);
	        Eigen::MatrixXd inv = qr.matrixR().topRows(qr.matrixR().cols());

			inv = inv.inverse();

	        for (int j = 0; j < inv.cols(); ++j) {
		        for (int k = 0; k < inv.rows(); ++k) {
			        coeffR.emplace_back(j+1 + i*2,k+1 + i*2, inv.coeff(k,j));
		        }
	        }

            rowOffset += offs * 2;
        }
	    Eigen::SparseMatrix<number_t> R_total(matrix.cols(), matrix.cols());
        R_total.setFromTriplets(coeffR.begin(), coeffR.end());
	    return  R_total;

    }


        template <typename Derived>
    Eigen::SparseMatrix<number_t> computeR_inverse(const Eigen::SparseMatrixBase<Derived>& matrix, int offset, int end, int colDim) {

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
        _cholesky.analyzePattern(J);
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
        _cholesky.analyzePatternWithPermutation(J, scalarP);

      }
      G2OBatchStatistics* globalStats = G2OBatchStatistics::globalStats();
      if (globalStats)
        globalStats->timeSymbolicDecomposition = get_monotonic_time() - t;
    }

    void fillSparseMatrix(const SparseBlockMatrix<MatrixType>& A, bool onlyValues)
    {
      if (onlyValues) {
        A.fillCCS(J.valuePtr(), true);
      } else {

        // create from triplet structure
        std::vector<Triplet> triplets;
        triplets.reserve(A.nonZeros());
        for (size_t c = 0; c < A.blockCols().size(); ++c) {
          int colBaseOfBlock = A.colBaseOfBlock(c);
          const typename SparseBlockMatrix<MatrixType>::IntBlockMap& column = A.blockCols()[c];
          for (typename SparseBlockMatrix<MatrixType>::IntBlockMap::const_iterator it = column.begin(); it != column.end(); ++it) {
            int rowBaseOfBlock = A.rowBaseOfBlock(it->first);
            const MatrixType& m = *(it->second);
            for (int cc = 0; cc < m.cols(); ++cc) {
              int aux_c = colBaseOfBlock + cc;
              for (int rr = 0; rr < m.rows(); ++rr) {
                int aux_r = rowBaseOfBlock + rr;
                if (aux_r > aux_c)
                  break;
                triplets.push_back(Triplet(aux_r, aux_c, m(rr, cc)));
              }
            }
          }
        }
        J.setFromTriplets(triplets.begin(), triplets.end());

      }
    }
};

} // end namespace

#endif

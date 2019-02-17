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
               int _rowDim, int _colDimCam, int _colDimPoint)
    {

      J =  A;

      VectorX::MapType xVec(x, J.cols());
      VectorX::ConstMapType errVec(b, J.rows());


      Eigen::SparseMatrix<number_t> Jc_tmp = J.leftCols(_numCams * _colDimCam);
      Eigen::SparseMatrix<number_t> Jp_tmp = J.rightCols(_numPoints * _colDimPoint);

	    saveMarket((J), "/home/lukas/Documents/eigenMatrices/j_orig.matx");
	    saveMarket((Jc_tmp), "/home/lukas/Documents/eigenMatrices/jC_tmp.matx");
        saveMarket((Jp_tmp), "/home/lukas/Documents/eigenMatrices/jP_tmp.matx");


	    Eigen::SparseMatrix<number_t> Rc_inv = computeR_inverse(Jc_tmp);
      Eigen::SparseMatrix<number_t> Rp_inv = computeR_inverse(Jp_tmp);

        saveMarket((Rc_inv), "/home/lukas/Documents/eigenMatrices/Rc_inv.matx");
        saveMarket((Rp_inv), "/home/lukas/Documents/eigenMatrices/Rp_inv.matx");

      //Jc_tmp.resize(0,0);
      //Jp_tmp.resize(0,0);

      Eigen::MatrixXd R_inv_tmp(Rc_inv.cols() + Rp_inv.cols(), Rc_inv.cols() + Rp_inv.cols());
      R_inv_tmp.topLeftCorner(Rc_inv.cols(), Rc_inv.cols()) = Rc_inv;
      R_inv_tmp.bottomRightCorner(Rp_inv.cols(), Rp_inv.cols()) = Rp_inv;

      Eigen::SparseMatrix<number_t> R_inv = R_inv_tmp.sparseView();

        saveMarket((R_inv_tmp), "/home/lukas/Documents/eigenMatrices/R_tmp_inv.matx");
        saveMarket((R_inv), "/home/lukas/Documents/eigenMatrices/R_inv.matx");


        //R_inv_tmp.resize(0,0);

      Eigen::SparseMatrix<number_t> _precondJ = J * R_inv;

      Eigen::SparseMatrix<number_t> Jc = _precondJ.leftCols(_numCams * _colDimCam);
      Eigen::SparseMatrix<number_t> Jp = _precondJ.rightCols(_numPoints * _colDimPoint);

        saveMarket((_precondJ), "/home/lukas/Documents/eigenMatrices/j_pre.matx");
        saveMarket((Jc), "/home/lukas/Documents/eigenMatrices/jC_pre.matx");
        saveMarket((Jp), "/home/lukas/Documents/eigenMatrices/jP_pre.matx");
        Eigen::SparseMatrix<number_t> hessian = _precondJ.transpose() * _precondJ;

        saveMarket(hessian,"/home/lukas/Documents/eigenMatrices/hessian_precond.matx");
      number_t eta = 0.1;
      VectorX::MapType xC(x, _numCams * _colDimCam);
      VectorX::MapType xP(x + _numCams * _colDimCam, _numPoints * _colDimPoint);

      xC.setZero();
      // We do not have r, but rather b. D
      xP = (-1) * Jp.transpose()*errVec;

        std::cout << "xVector Init: " << std::endl;
        for (int i = 0; i<J.cols();++i) {
            std::cout << xVec[i] << std::endl;
        }


      VectorX p = _precondJ.transpose() * ((-1) * errVec - (_precondJ * xVec));
      VectorX r = p;

      Eigen::Ref<VectorX> rC = r.segment(0, _numCams * _colDimCam);
      Eigen::Ref<VectorX> rP = r.segment(_numCams* _colDimCam , _numPoints * _colDimPoint);


      number_t gamma = xVec.dot(xVec);
      VectorX q = _precondJ * p;

      number_t err_start_rC = eta * rC.dot(rC);
      number_t err_start_rP = eta * rP.dot(rP);


      size_t maxIter = J.rows();

      maxIter = 3;
      number_t alpha;
      number_t beta;
      number_t gammaNew;
      size_t iteration;
      number_t otherErr;
      bool otherDone;
      for (iteration = 0; iteration < maxIter;++iteration) {
          //check if error small enough
          std::cout << "errC " << rC.dot(rC) << " start " <<  err_start_rC << std::endl;
          std::cout << "errP " << rP.dot(rP) << " start " <<  err_start_rP << std::endl;

          if (iteration % 2) {
            otherDone = otherErr < err_start_rC;
          } else {
            otherDone = otherErr < err_start_rP;
          }

          /*
          if (rC.dot(rC) < err_start_rC && rP.dot(rP) < err_start_rP && otherDone)
          {
              break;
          }
            */
          alpha = gamma / (q.dot(q));
          xVec = xVec + alpha * p;
          if (iteration % 2) {
            //odd
            rC = (-1) * alpha * Jc.transpose() * q;
            otherErr = rP.dot(rP);
            rP.setZero();

          } else {
            //even
            rP = (-1) * alpha * Jp.transpose() * q;
            otherErr = rC.dot(rC);
            rC.setZero();
          }
          gammaNew = r.dot(r);
          beta = gammaNew / gamma;
          gamma = gammaNew;
          p = r + beta*p;
          if (iteration % 2) {
            //odd
            q = beta * q + Jc * rC;
          } else {
            //even
            q = beta * q + Jp * rP;
          }

      }

      //retrieve xC, xP


      std::cout << "iter: " << iteration << std::endl;
        std::cout << "xVector: " << std::endl;
      for (int i = 0; i<J.cols();++i) {
          std::cout << xVec[i] << std::endl;
      }

      Eigen::VectorXd approx =  R_inv.transpose() * J.transpose() * J  * R_inv * xVec;

      Eigen::VectorXd res = approx - ((-1) * R_inv.transpose() * J.transpose() * errVec);

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
    SparseMatrix J;
    CholeskyDecomposition _cholesky;

    template <typename Derived>
    Eigen::SparseMatrix<number_t> computeR_inverse(const Eigen::SparseMatrixBase<Derived>& matrix) {

        Eigen::SparseQR<Eigen::SparseMatrix<number_t>, Eigen::NaturalOrdering<int> > qr;

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

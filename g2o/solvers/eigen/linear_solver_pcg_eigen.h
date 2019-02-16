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

      _sparseMatrix = A;

      VectorX::MapType xVec(x, _sparseMatrix.cols());
      VectorX::ConstMapType bVec(b, _sparseMatrix.cols());

      for (int i = 0; i<_numCams;++i) {
        // Go through all Cameras an compute their R


      }

      Eigen::SparseQR<Eigen::SparseMatrix<number_t>, Eigen::NaturalOrdering<int> > qr;
      qr.analyzePattern(_sparseMatrix);

      qr.factorize(_sparseMatrix);
      Eigen::SparseMatrix<number_t> R = qr.matrixR();
      // _sparseMatrix is now in EIGEN form, and can be used with its interface

      double eta = 0.1;
      VectorX::MapType xC(x, _numCams * 6);
      VectorX::MapType xP(x + _numCams * 6, _numPoints * 3);

      Eigen::SparseMatrix<number_t> Jc = _sparseMatrix.leftCols(_numCams * 6);
      Eigen::SparseMatrix<number_t> Jp = _sparseMatrix.rightCols(_numPoints * 3);
      

      if (_init) // compute the symbolic composition once
        _cholesky.analyzePattern(_sparseMatrix);
        //computeSymbolicDecomposition(A);
      _init = false;

      number_t t=get_monotonic_time();
      _cholesky.factorize(_sparseMatrix);
      if (_cholesky.info() != Eigen::Success) { // the matrix is not positive definite
        if (_writeDebug) {
          std::cerr << "Cholesky failure, writing debug.txt (Hessian loadable by Octave)" << std::endl;
          //_sparseMatrix.writeOctave("debug.txt");
        }
        return false;
      }

      // Solving the system
      VectorX::MapType xx(x, _sparseMatrix.cols());
      VectorX::ConstMapType bb(b, _sparseMatrix.cols());
      xx = _cholesky.solve(bb);

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
    SparseMatrix _sparseMatrix;
    CholeskyDecomposition _cholesky;

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
        _cholesky.analyzePattern(_sparseMatrix);
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
        _cholesky.analyzePatternWithPermutation(_sparseMatrix, scalarP);

      }
      G2OBatchStatistics* globalStats = G2OBatchStatistics::globalStats();
      if (globalStats)
        globalStats->timeSymbolicDecomposition = get_monotonic_time() - t;
    }

    void fillSparseMatrix(const SparseBlockMatrix<MatrixType>& A, bool onlyValues)
    {
      if (onlyValues) {
        A.fillCCS(_sparseMatrix.valuePtr(), true);
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
        _sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());

      }
    }
};

} // end namespace

#endif

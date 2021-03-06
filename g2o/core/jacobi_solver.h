//
// created on basis of block_solver.h by Lukas Schneider, 18.03.2019.
//
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

#ifndef G2O_JACOBI_SOLVER_H
#define G2O_JACOBI_SOLVER_H

#include <unsupported/Eigen/SparseExtra>
#include "solver.h"
#include "linear_solver.h"
#include "sparse_block_matrix.h"
#include "sparse_block_matrix_diagonal.h"
#include "openmp_mutex.h"
#include "g2o/config.h"
#include "dynamic_aligned_buffer.hpp"
#include "block_solver.h"
#include "Eigen/Core"

#include <memory>

namespace g2o {

  /**
   * \brief traits to summarize the properties of the fixed size optimization problem
   */
  template <int _PoseDim, int _LandmarkDim>
  struct JacobiSolverTraits
  {
    static const int PoseDim = _PoseDim;
    static const int LandmarkDim = _LandmarkDim;
    typedef Eigen::Matrix<number_t, PoseDim, PoseDim, Eigen::ColMajor> PoseMatrixType;
    typedef Eigen::Matrix<number_t, LandmarkDim, LandmarkDim, Eigen::ColMajor> LandmarkMatrixType;
    typedef Eigen::Matrix<number_t, PoseDim, LandmarkDim, Eigen::ColMajor> PoseLandmarkMatrixType;
    typedef Eigen::Matrix<number_t, PoseDim, 1, Eigen::ColMajor> PoseVectorType;
    typedef Eigen::Matrix<number_t, LandmarkDim, 1, Eigen::ColMajor> LandmarkVectorType;

    typedef SparseBlockMatrix<PoseMatrixType> PoseHessianType;
    typedef SparseBlockMatrix<LandmarkMatrixType> LandmarkHessianType;
    typedef SparseBlockMatrix<PoseLandmarkMatrixType> PoseLandmarkHessianType;
    typedef LinearSolver<PoseMatrixType> LinearSolverType;
  };

  /**
   * \brief traits to summarize the properties of the dynamic size optimization problem
   */
  template <>
  struct JacobiSolverTraits<Eigen::Dynamic, Eigen::Dynamic>
  {
    static const int PoseDim = Eigen::Dynamic;
    static const int LandmarkDim = Eigen::Dynamic;
    typedef MatrixX PoseMatrixType;
    typedef MatrixX LandmarkMatrixType;
    typedef MatrixX PoseLandmarkMatrixType;
    typedef VectorX PoseVectorType;
    typedef VectorX LandmarkVectorType;

    typedef SparseBlockMatrix<PoseMatrixType> PoseHessianType;
    typedef SparseBlockMatrix<LandmarkMatrixType> LandmarkHessianType;
    typedef SparseBlockMatrix<PoseLandmarkMatrixType> PoseLandmarkHessianType;
    typedef LinearSolver<PoseMatrixType> LinearSolverType;
  };

  /**
   * \brief Implementation of a solver operating on the blocks of the Jacobi
   */
  template <typename Traits>
  class JacobiSolver: public BlockSolverBase
  {
    public:
	  typedef Eigen::Triplet<number_t> T;
      static const int PoseDim = Traits::PoseDim;
      static const int LandmarkDim = Traits::LandmarkDim;
      typedef typename Traits::PoseMatrixType PoseMatrixType;
      typedef typename Traits::LandmarkMatrixType LandmarkMatrixType; 
      typedef typename Traits::PoseLandmarkMatrixType PoseLandmarkMatrixType;
      typedef typename Traits::PoseVectorType PoseVectorType;
      typedef typename Traits::LandmarkVectorType LandmarkVectorType;

      typedef typename Traits::PoseHessianType PoseHessianType;
      typedef typename Traits::LandmarkHessianType LandmarkHessianType;
      typedef typename Traits::PoseLandmarkHessianType PoseLandmarkHessianType;
      typedef typename Traits::LinearSolverType LinearSolverType;

    public:

      /**
       * allocate a block solver ontop of the underlying linear solver.
       * NOTE: The BlockSolver assumes exclusive access to the linear solver and will therefore free the pointer
       * in its destructor.
       */
      JacobiSolver(std::unique_ptr<LinearSolverType> linearSolver);
      ~JacobiSolver();

      virtual bool init(SparseOptimizer* optmizer, bool online = false);
      virtual bool buildStructure(bool zeroBlocks = false);
      virtual bool updateStructure(const std::vector<HyperGraph::Vertex*>& vset, const HyperGraph::EdgeSet& edges);
      virtual bool buildSystem();
      virtual bool solve();
      virtual bool computeMarginals(SparseBlockMatrix<MatrixX>& spinv, const std::vector<std::pair<int, int> >& blockIndices);
      virtual bool setLambda(number_t lambda, bool backup = false);
      virtual void restoreDiagonal();
      virtual bool supportsSchur() {return false;}
      virtual bool schur() { return _doSchur;}
      virtual void setSchur(bool s) { _doSchur = s;}

      LinearSolver<PoseMatrixType>& linearSolver() const { return *_linearSolver;}

      virtual void setWriteDebug(bool writeDebug);
      virtual bool writeDebug() const {return _linearSolver->writeDebug();}

      virtual bool saveHessian(const std::string& fileName) const;

      virtual void multiplyHessian(number_t* dest, const number_t* src) const { _Hpp->multiplySymmetricUpperTriangle(dest, src);}

      virtual Eigen::SparseMatrix<number_t>& getJacobi() {return *_jacobiFull;};

      virtual void setEta(number_t eta) {_eta = eta;};

  protected:
      void resize(int* blockPoseIndices, int numPoseBlocks, 
          int* blockLandmarkIndices, int numLandmarkBlocks, int totalDim);

      void deallocate();
      number_t _eta;
      number_t _lambda;
      //std::vector<number_t> _errVector;
      std::vector<std::reference_wrapper<number_t>> _scaleCoeff;
	  std::unique_ptr<Eigen::SparseMatrix<number_t>> _jacobiFull;

      std::unique_ptr<SparseBlockMatrix<PoseMatrixType>> _Hpp;
      std::unique_ptr<SparseBlockMatrix<LandmarkMatrixType>> _Hll;
      std::unique_ptr<SparseBlockMatrix<PoseLandmarkMatrixType>> _Hpl;

      std::unique_ptr<SparseBlockMatrix<PoseMatrixType>> _Hschur;
      std::unique_ptr<SparseBlockMatrixDiagonal<LandmarkMatrixType>> _DInvSchur;

      std::unique_ptr<SparseBlockMatrixCCS<PoseLandmarkMatrixType>> _HplCCS;
      std::unique_ptr<SparseBlockMatrixCCS<PoseMatrixType>> _HschurTransposedCCS;

      std::unique_ptr<LinearSolverType> _linearSolver;

      std::vector<PoseVectorType, Eigen::aligned_allocator<PoseVectorType> > _diagonalBackupPose;
      std::vector<LandmarkVectorType, Eigen::aligned_allocator<LandmarkVectorType> > _diagonalBackupLandmark;

#    ifdef G2O_OPENMP
      std::vector<OpenMPMutex> _coefficientsMutex;
#    endif

      bool _doSchur;

      std::unique_ptr<number_t[], aligned_deleter<number_t>> _coefficients;
      std::unique_ptr<number_t[], aligned_deleter<number_t>> _bschur;

      int _numPoses, _numLandmarks;
      int _sizePoses, _sizeLandmarks;
  };


  template<int p, int l>
  using JacobiSolverPL = JacobiSolver< JacobiSolverTraits<p, l> >;

  // solver for BA/3D SLAM
  using JacobiSolver_6_3 = JacobiSolverPL<6, 3>;

} // end namespace

#include "jacobi_solver.hpp"


#endif

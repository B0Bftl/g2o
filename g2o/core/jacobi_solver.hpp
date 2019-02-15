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

#include "sparse_optimizer.h"

#include <Eigen/LU>
#include <fstream>
#include <iomanip>

#include "g2o/stuff/timeutil.h"
#include "g2o/stuff/macros.h"
#include "g2o/stuff/misc.h"

#include "jacobi_solver.h"

namespace g2o {

template <typename Traits>
JacobiSolver<Traits>::JacobiSolver(std::unique_ptr<LinearSolverType> linearSolver)
    :   BlockSolverBase(),
        _linearSolver(std::move(linearSolver))
{
  // workspace
  _xSize=0;
  _numPoses=0;
  _numLandmarks=0;
  _sizePoses=0;
  _sizeLandmarks=0;
  _doSchur=false;
}

template <typename Traits>
void JacobiSolver<Traits>::resize(int* blockPoseIndices, int numPoseBlocks,
              int* blockLandmarkIndices, int numLandmarkBlocks,
              int s)
{
  deallocate();

  resizeVector(s);

  if (_doSchur) {
    // the following two are only used in schur
    assert(_sizePoses > 0 && "allocating with wrong size");
    _coefficients.reset(allocate_aligned<number_t>(s));
    _bschur.reset(allocate_aligned<number_t>(_sizePoses));
  }
  _JacobiC = g2o::make_unique<Eigen::SparseMatrix<number_t>>(0,0);
  _JacobiP = g2o::make_unique<Eigen::SparseMatrix<number_t>>(0,0);

  _Hpp= g2o::make_unique<PoseHessianType>(blockPoseIndices, blockPoseIndices, numPoseBlocks, numPoseBlocks);
  _Hll = g2o::make_unique<LandmarkHessianType>(blockLandmarkIndices, blockLandmarkIndices, numLandmarkBlocks, numLandmarkBlocks);
  _Hpl = g2o::make_unique<PoseLandmarkHessianType>(blockPoseIndices, blockLandmarkIndices, numPoseBlocks, numLandmarkBlocks);
  _HplCCS = g2o::make_unique<SparseBlockMatrixCCS<PoseLandmarkMatrixType>>(_Hpl->rowBlockIndices(), _Hpl->colBlockIndices());

	if (_doSchur) {
    _Hschur = g2o::make_unique<PoseHessianType>(blockPoseIndices, blockPoseIndices, numPoseBlocks, numPoseBlocks);
    _DInvSchur = g2o::make_unique<SparseBlockMatrixDiagonal<LandmarkMatrixType>>(_Hll->colBlockIndices());
    _HschurTransposedCCS = g2o::make_unique<SparseBlockMatrixCCS<PoseMatrixType>>(_Hschur->colBlockIndices(), _Hschur->rowBlockIndices());
#ifdef G2O_OPENMP
    _coefficientsMutex.resize(numPoseBlocks);
#endif
  }
}

template <typename Traits>
void JacobiSolver<Traits>::deallocate()
{
    //_JacobiP->resize(0,0);
    //_JacobiC->resize(0,0);
    _Hpp.reset();
    _Hll.reset();
    _Hpl.reset();
    _Hschur.reset();
    _DInvSchur.reset();
    _coefficients.reset();
    _bschur.reset();
    
    _HplCCS.reset();
    _HschurTransposedCCS.reset();
}

template <typename Traits>
JacobiSolver<Traits>::~JacobiSolver()
{}

template <typename Traits>
bool JacobiSolver<Traits>::buildStructure(bool zeroBlocks)
{
  assert(_optimizer);

  size_t sparseDim = 0;
  _numPoses=0;
  _numLandmarks=0;
  _sizePoses=0;
  _sizeLandmarks=0;
  int* blockPoseIndices = new int[_optimizer->indexMapping().size()];
  int* blockLandmarkIndices = new int[_optimizer->indexMapping().size()];

  for (size_t i = 0; i < _optimizer->indexMapping().size(); ++i) {
    OptimizableGraph::Vertex* v = _optimizer->indexMapping()[i];
    int dim = v->dimension();
    if (! v->marginalized()){
      v->setColInHessian(_sizePoses);
      _sizePoses+=dim;
      blockPoseIndices[_numPoses]=_sizePoses;
      ++_numPoses;
    } else {
      v->setColInHessian(_sizeLandmarks);
      _sizeLandmarks+=dim;
      blockLandmarkIndices[_numLandmarks]=_sizeLandmarks;
      ++_numLandmarks;
    }
    sparseDim += dim;
  }
  resize(blockPoseIndices, _numPoses, blockLandmarkIndices, _numLandmarks, sparseDim);
  delete[] blockLandmarkIndices;
  delete[] blockPoseIndices;

  // allocate the diagonal on Hpp and Hll
  int poseIdx = 0;
  int landmarkIdx = 0;
  for (size_t i = 0; i < _optimizer->indexMapping().size(); ++i) {
    OptimizableGraph::Vertex* v = _optimizer->indexMapping()[i];
    if (! v->marginalized()){
      //assert(poseIdx == v->hessianIndex());
      PoseMatrixType* m = _Hpp->block(poseIdx, poseIdx, true);
      if (zeroBlocks)
        m->setZero();
      v->mapHessianMemory(m->data());
      ++poseIdx;
    } else {
      LandmarkMatrixType* m = _Hll->block(landmarkIdx, landmarkIdx, true);
      if (zeroBlocks)
        m->setZero();
      v->mapHessianMemory(m->data());
      ++landmarkIdx;
    }
  }
  assert(poseIdx == _numPoses && landmarkIdx == _numLandmarks);

  // temporary structures for building the pattern of the Schur complement
  SparseBlockMatrixHashMap<PoseMatrixType>* schurMatrixLookup = 0;
  if (_doSchur) {
    schurMatrixLookup = new SparseBlockMatrixHashMap<PoseMatrixType>(_Hschur->rowBlockIndices(), _Hschur->colBlockIndices());
    schurMatrixLookup->blockCols().resize(_Hschur->blockCols().size());
  }

  // here we assume that the landmark indices start after the pose ones
  // create the structure in Hpp, Hll and in Hpl
  for (SparseOptimizer::EdgeContainer::const_iterator it=_optimizer->activeEdges().begin(); it!=_optimizer->activeEdges().end(); ++it){
    OptimizableGraph::Edge* e = *it;

    for (size_t viIdx = 0; viIdx < e->vertices().size(); ++viIdx) {
      OptimizableGraph::Vertex* v1 = (OptimizableGraph::Vertex*) e->vertex(viIdx);
      int ind1 = v1->hessianIndex();
      if (ind1 == -1)
        continue;
      int indexV1Bak = ind1;
      for (size_t vjIdx = viIdx + 1; vjIdx < e->vertices().size(); ++vjIdx) {
        OptimizableGraph::Vertex* v2 = (OptimizableGraph::Vertex*) e->vertex(vjIdx);
        int ind2 = v2->hessianIndex();
        if (ind2 == -1)
          continue;
        ind1 = indexV1Bak;
        bool transposedBlock = ind1 > ind2;
        if (transposedBlock){ // make sure, we allocate the upper triangle block
          std::swap(ind1, ind2);
        }
        if (! v1->marginalized() && !v2->marginalized()){
          PoseMatrixType* m = _Hpp->block(ind1, ind2, true);
          if (zeroBlocks)
            m->setZero();
          e->mapHessianMemory(m->data(), viIdx, vjIdx, transposedBlock);
          if (_Hschur) {// assume this is only needed in case we solve with the schur complement
            schurMatrixLookup->addBlock(ind1, ind2);
          }
        } else if (v1->marginalized() && v2->marginalized()){
          // RAINER hmm.... should we ever reach this here????
          LandmarkMatrixType* m = _Hll->block(ind1-_numPoses, ind2-_numPoses, true);
          if (zeroBlocks)
            m->setZero();
          e->mapHessianMemory(m->data(), viIdx, vjIdx, false);
        } else { 
          if (v1->marginalized()){ 
            PoseLandmarkMatrixType* m = _Hpl->block(v2->hessianIndex(),v1->hessianIndex()-_numPoses, true);
            if (zeroBlocks)
              m->setZero();
            e->mapHessianMemory(m->data(), viIdx, vjIdx, true); // transpose the block before writing to it
          } else {
            PoseLandmarkMatrixType* m = _Hpl->block(v1->hessianIndex(),v2->hessianIndex()-_numPoses, true);
            if (zeroBlocks)
              m->setZero();
            e->mapHessianMemory(m->data(), viIdx, vjIdx, false); // directly the block
          }
        }
      }
    }
  }

  if (! _doSchur) {
    delete schurMatrixLookup;
    return true;
  }

  _DInvSchur->diagonal().resize(landmarkIdx);
  _Hpl->fillSparseBlockMatrixCCS(*_HplCCS);

  for (OptimizableGraph::Vertex* v : _optimizer->indexMapping()) {
    if (v->marginalized()){
      const HyperGraph::EdgeSet& vedges=v->edges();
      for (HyperGraph::EdgeSet::const_iterator it1=vedges.begin(); it1!=vedges.end(); ++it1){
        for (size_t i=0; i<(*it1)->vertices().size(); ++i)
        {
          OptimizableGraph::Vertex* v1= (OptimizableGraph::Vertex*) (*it1)->vertex(i);
          if (v1->hessianIndex()==-1 || v1==v)
            continue;
          for  (HyperGraph::EdgeSet::const_iterator it2=vedges.begin(); it2!=vedges.end(); ++it2){
            for (size_t j=0; j<(*it2)->vertices().size(); ++j)
            {
              OptimizableGraph::Vertex* v2= (OptimizableGraph::Vertex*) (*it2)->vertex(j);
              if (v2->hessianIndex()==-1 || v2==v)
                continue;
              int i1=v1->hessianIndex();
              int i2=v2->hessianIndex();
              if (i1<=i2) {
                schurMatrixLookup->addBlock(i1, i2);
              }
            }
          }
        }
      }
    }
  }

  _Hschur->takePatternFromHash(*schurMatrixLookup);
  delete schurMatrixLookup;
  _Hschur->fillSparseBlockMatrixCCSTransposed(*_HschurTransposedCCS);

  return true;
}

template <typename Traits>
bool JacobiSolver<Traits>::updateStructure(const std::vector<HyperGraph::Vertex*>& vset, const HyperGraph::EdgeSet& edges)
{
  for (std::vector<HyperGraph::Vertex*>::const_iterator vit = vset.begin(); vit != vset.end(); ++vit) {
    OptimizableGraph::Vertex* v = static_cast<OptimizableGraph::Vertex*>(*vit);
    int dim = v->dimension();
    if (! v->marginalized()){
      v->setColInHessian(_sizePoses);
      _sizePoses+=dim;
      _Hpp->rowBlockIndices().push_back(_sizePoses);
      _Hpp->colBlockIndices().push_back(_sizePoses);
      _Hpp->blockCols().push_back(typename SparseBlockMatrix<PoseMatrixType>::IntBlockMap());
      ++_numPoses;
      int ind = v->hessianIndex();
      PoseMatrixType* m = _Hpp->block(ind, ind, true);
      v->mapHessianMemory(m->data());
    } else {
      std::cerr << "updateStructure(): Schur not supported" << std::endl;
      abort();
    }
  }
  resizeVector(_sizePoses + _sizeLandmarks);

  for (HyperGraph::EdgeSet::const_iterator it = edges.begin(); it != edges.end(); ++it) {
    OptimizableGraph::Edge* e = static_cast<OptimizableGraph::Edge*>(*it);

    for (size_t viIdx = 0; viIdx < e->vertices().size(); ++viIdx) {
      OptimizableGraph::Vertex* v1 = (OptimizableGraph::Vertex*) e->vertex(viIdx);
      int ind1 = v1->hessianIndex();
      int indexV1Bak = ind1;
      if (ind1 == -1)
        continue;
      for (size_t vjIdx = viIdx + 1; vjIdx < e->vertices().size(); ++vjIdx) {
        OptimizableGraph::Vertex* v2 = (OptimizableGraph::Vertex*) e->vertex(vjIdx);
        int ind2 = v2->hessianIndex();
        if (ind2 == -1)
          continue;
        ind1 = indexV1Bak;
        bool transposedBlock = ind1 > ind2;
        if (transposedBlock) // make sure, we allocate the upper triangular block
          std::swap(ind1, ind2);

        if (! v1->marginalized() && !v2->marginalized()) {
          PoseMatrixType* m = _Hpp->block(ind1, ind2, true);
          e->mapHessianMemory(m->data(), viIdx, vjIdx, transposedBlock);
        } else { 
          std::cerr << __PRETTY_FUNCTION__ << ": not supported" << std::endl;
        }
      }
    }

  }

  return true;
}

template <typename Traits>
bool JacobiSolver<Traits>::solve(){
  //cerr << __PRETTY_FUNCTION__ << endl;
  if (! _doSchur){
    number_t t=get_monotonic_time();
    bool ok = _linearSolver->solve(*_Hpp, _x, _b);
    G2OBatchStatistics* globalStats = G2OBatchStatistics::globalStats();
    if (globalStats) {
      globalStats->timeLinearSolver = get_monotonic_time() - t;
      globalStats->hessianDimension = globalStats->hessianPoseDimension = _Hpp->cols();
    }
    return ok;
  }

  // schur thing

  // backup the coefficient matrix
  number_t t=get_monotonic_time();

  // _Hschur = _Hpp, but keeping the pattern of _Hschur
  _Hschur->clear();
  _Hpp->add(*_Hschur);

  //_DInvSchur->clear();
  memset(_coefficients.get(), 0, _sizePoses*sizeof(number_t));
# ifdef G2O_OPENMP
# pragma omp parallel for default (shared) schedule(dynamic, 10)
# endif
  for (int landmarkIndex = 0; landmarkIndex < static_cast<int>(_Hll->blockCols().size()); ++landmarkIndex) {
    const typename SparseBlockMatrix<LandmarkMatrixType>::IntBlockMap& marginalizeColumn = _Hll->blockCols()[landmarkIndex];
    assert(marginalizeColumn.size() == 1 && "more than one block in _Hll column");

    // calculate inverse block for the landmark
    const LandmarkMatrixType * D = marginalizeColumn.begin()->second;
    assert (D && D->rows()==D->cols() && "Error in landmark matrix");
    LandmarkMatrixType& Dinv = _DInvSchur->diagonal()[landmarkIndex];
    Dinv = D->inverse();

    LandmarkVectorType  db(D->rows());
    for (int j=0; j<D->rows(); ++j) {
      db[j]=_b[_Hll->rowBaseOfBlock(landmarkIndex) + _sizePoses + j];
    }
    db=Dinv*db;

    assert((size_t)landmarkIndex < _HplCCS->blockCols().size() && "Index out of bounds");
    const typename SparseBlockMatrixCCS<PoseLandmarkMatrixType>::SparseColumn& landmarkColumn = _HplCCS->blockCols()[landmarkIndex];

    for (typename SparseBlockMatrixCCS<PoseLandmarkMatrixType>::SparseColumn::const_iterator it_outer = landmarkColumn.begin();
        it_outer != landmarkColumn.end(); ++it_outer) {
      int i1 = it_outer->row;

      const PoseLandmarkMatrixType* Bi = it_outer->block;
      assert(Bi);

      PoseLandmarkMatrixType BDinv = (*Bi)*(Dinv);
      assert(_HplCCS->rowBaseOfBlock(i1) < _sizePoses && "Index out of bounds");
      typename PoseVectorType::MapType Bb(&_coefficients[_HplCCS->rowBaseOfBlock(i1)], Bi->rows());
#    ifdef G2O_OPENMP
      ScopedOpenMPMutex mutexLock(&_coefficientsMutex[i1]);
#    endif
      Bb.noalias() += (*Bi)*db;

      assert(i1 >= 0 && i1 < static_cast<int>(_HschurTransposedCCS->blockCols().size()) && "Index out of bounds");
      typename SparseBlockMatrixCCS<PoseMatrixType>::SparseColumn::iterator targetColumnIt = _HschurTransposedCCS->blockCols()[i1].begin();

      typename SparseBlockMatrixCCS<PoseLandmarkMatrixType>::RowBlock aux(i1, 0);
      typename SparseBlockMatrixCCS<PoseLandmarkMatrixType>::SparseColumn::const_iterator it_inner = lower_bound(landmarkColumn.begin(), landmarkColumn.end(), aux);
      for (; it_inner != landmarkColumn.end(); ++it_inner) {
        int i2 = it_inner->row;
        const PoseLandmarkMatrixType* Bj = it_inner->block;
        assert(Bj); 
        while (targetColumnIt->row < i2 /*&& targetColumnIt != _HschurTransposedCCS->blockCols()[i1].end()*/)
          ++targetColumnIt;
        assert(targetColumnIt != _HschurTransposedCCS->blockCols()[i1].end() && targetColumnIt->row == i2 && "invalid iterator, something wrong with the matrix structure");
        PoseMatrixType* Hi1i2 = targetColumnIt->block;//_Hschur->block(i1,i2);
        assert(Hi1i2);
        (*Hi1i2).noalias() -= BDinv*Bj->transpose();
      }
    }
  }
  //cerr << "Solve [marginalize] = " <<  get_monotonic_time()-t << endl;

  // _bschur = _b for calling solver, and not touching _b
  memcpy(_bschur.get(), _b, _sizePoses * sizeof(number_t));
  for (int i=0; i<_sizePoses; ++i){
    _bschur[i]-=_coefficients[i];
  }

  G2OBatchStatistics* globalStats = G2OBatchStatistics::globalStats();
  if (globalStats){
    globalStats->timeSchurComplement = get_monotonic_time() - t;
  }

  t=get_monotonic_time();
  bool solvedPoses = _linearSolver->solve(*_Hschur, _x, _bschur.get());
  if (globalStats) {
    globalStats->timeLinearSolver = get_monotonic_time() - t;
    globalStats->hessianPoseDimension = _Hpp->cols();
    globalStats->hessianLandmarkDimension = _Hll->cols();
    globalStats->hessianDimension = globalStats->hessianPoseDimension + globalStats->hessianLandmarkDimension;
  }
  //cerr << "Solve [decompose and solve] = " <<  get_monotonic_time()-t << endl;

  if (! solvedPoses)
    return false;

  // _x contains the solution for the poses, now applying it to the landmarks to get the new part of the
  // solution;
  number_t* xp = _x;
  number_t* cp = _coefficients.get();

  number_t* xl=_x+_sizePoses;
  number_t* cl=_coefficients.get() + _sizePoses;
  number_t* bl=_b+_sizePoses;

  // cp = -xp
  for (int i=0; i<_sizePoses; ++i)
    cp[i]=-xp[i];

  // cl = bl
  memcpy(cl,bl,_sizeLandmarks*sizeof(number_t));

  // cl = bl - Bt * xp
  //Bt->multiply(cl, cp);
  _HplCCS->rightMultiply(cl, cp);

  // xl = Dinv * cl
  memset(xl,0, _sizeLandmarks*sizeof(number_t));
  _DInvSchur->multiply(xl,cl);
  //_DInvSchur->rightMultiply(xl,cl);
  //cerr << "Solve [landmark delta] = " <<  get_monotonic_time()-t << endl;

  return true;
}


template <typename Traits>
bool JacobiSolver<Traits>::computeMarginals(SparseBlockMatrix<MatrixX>& spinv, const std::vector<std::pair<int, int> >& blockIndices)
{
  number_t t = get_monotonic_time();
  bool ok = _linearSolver->solvePattern(spinv, blockIndices, *_Hpp);
  G2OBatchStatistics* globalStats = G2OBatchStatistics::globalStats();
  if (globalStats) {
    globalStats->timeMarginals = get_monotonic_time() - t;
  }
  return ok;
}

template <typename Traits>
bool JacobiSolver<Traits>::buildSystem()
{
  // clear b vector
# ifdef G2O_OPENMP
# pragma omp parallel for default (shared) if (_optimizer->indexMapping().size() > 1000)
# endif
  for (int i = 0; i < static_cast<int>(_optimizer->indexMapping().size()); ++i) {
    OptimizableGraph::Vertex* v=_optimizer->indexMapping()[i];
    assert(v);
    v->clearQuadraticForm();
  }
  _Hpp->clear();
  //TODO: Clear Jacobi
  //if (_doSchur) {
    _Hll->clear();
    _Hpl->clear();
  //}

  // resetting the terms for the pairwise constraints
  // built up the current system by storing the Hessian blocks in the edges and vertices
# ifndef G2O_OPENMP
  // no threading, we do not need to copy the workspace
  JacobianWorkspace& jacobianWorkspace = _optimizer->jacobianWorkspace();
# else
  // if running with threads need to produce copies of the workspace for each thread
  JacobianWorkspace jacobianWorkspace = _optimizer->jacobianWorkspace();
# pragma omp parallel for default (shared) firstprivate(jacobianWorkspace) if (_optimizer->activeEdges().size() > 100)
# endif
  std::vector<Eigen::Triplet<number_t>> jacobiDataP;
  std::vector<Eigen::Triplet<number_t>> jacobiDataC;

  //TODO: Find correct values here
  int sizeEdges = static_cast<int>(_optimizer->activeEdges().size());
  jacobiDataP.reserve(sizeEdges * 6);
  jacobiDataC.reserve(sizeEdges * 12);

  int rowsP = 2;
  int colsP = 3;

  int rowsC = 2;
  int colsC = 6;

  int offsetCol;
  int offsetRow;

  int rowCount = -1;

  std::set<int> setFixedPoints;
  std::set<int> setFixedCameras;


  number_t* data;


  for (int k = 0;k < static_cast<int>(_optimizer->indexMapping().size()); ++k) {
    const OptimizableGraph::Vertex* v = static_cast<const OptimizableGraph::Vertex*>(_optimizer->indexMapping()[k]);
    // only iterate over point vertices
    if(!v->marginalized()) continue;

    for (OptimizableGraph::EdgeSet::const_iterator it=v->edges().begin(); it!=v->edges().end(); ++it){
      OptimizableGraph::Edge* e = reinterpret_cast<OptimizableGraph::Edge*>(*it);
      if (!e->active()) continue;
      // get all active vertices and linearize their constraint

      const OptimizableGraph::Vertex* vi = static_cast<const OptimizableGraph::Vertex*>(e->vertex(0));
      const OptimizableGraph::Vertex* vj = static_cast<const OptimizableGraph::Vertex*>(e->vertex(1));


      if(vi->fixed() && vi->marginalized()) setFixedPoints.insert(vi->id());
      if(vi->fixed() && !vi->marginalized()) setFixedCameras.insert(vi->id());
      if(vj->fixed() && vj->marginalized()) setFixedPoints.insert(vj->id());
      if(vj->fixed() && !vj->marginalized()) setFixedCameras.insert(vj->id());


      //for (int k = 0; k < sizeEdges; ++k) {
      //OptimizableGraph::Edge* e = _optimizer->activeEdges()[k];
      e->linearizeOplus(jacobianWorkspace); // jacobian of the nodes' oplus (manifold)
      e->constructQuadraticForm();

      //if(vi->fixed() || vj->fixed()) continue;
      ++rowCount;
      // We only

      data = jacobianWorkspace.workspaceForVertex(0);
      if(vi->hessianIndex() >= 0){
        if(vi->marginalized()) {
          // Point
          // We know that we are sorted
          e->
          offsetRow = rowCount * rowsP;
          offsetCol = ((_numPoses) * colsC) + ((vi->hessianIndex() - _numPoses) * colsP);
          jacobiDataP.emplace_back(offsetRow + 0, offsetCol + 0,data[0]);
          jacobiDataP.emplace_back(offsetRow + 1, offsetCol + 0,data[1]);
          jacobiDataP.emplace_back(offsetRow + 0, offsetCol + 1,data[2]);
          jacobiDataP.emplace_back(offsetRow + 1, offsetCol + 1,data[3]);
          jacobiDataP.emplace_back(offsetRow + 0, offsetCol + 2,data[4]);
          jacobiDataP.emplace_back(offsetRow + 1, offsetCol + 2,data[5]);
        } else {
          // Camera
          offsetRow = rowCount * rowsC;
          offsetCol = (vi->hessianIndex()) * colsC;
          jacobiDataC.emplace_back(offsetRow + 0, offsetCol + 0,data[0]);
          jacobiDataC.emplace_back(offsetRow + 1, offsetCol + 0,data[1]);
          jacobiDataC.emplace_back(offsetRow + 0, offsetCol + 1,data[2]);
          jacobiDataC.emplace_back(offsetRow + 1, offsetCol + 1,data[3]);
          jacobiDataC.emplace_back(offsetRow + 0, offsetCol + 2,data[4]);
          jacobiDataC.emplace_back(offsetRow + 1, offsetCol + 2,data[5]);
          jacobiDataC.emplace_back(offsetRow + 0, offsetCol + 3,data[6]);
          jacobiDataC.emplace_back(offsetRow + 1, offsetCol + 3,data[7]);
          jacobiDataC.emplace_back(offsetRow + 0, offsetCol + 4,data[8]);
          jacobiDataC.emplace_back(offsetRow + 1, offsetCol + 4,data[9]);
          jacobiDataC.emplace_back(offsetRow + 0, offsetCol + 5,data[10]);
          jacobiDataC.emplace_back(offsetRow + 1, offsetCol + 5,data[11]);
        }
      }


      data = jacobianWorkspace.workspaceForVertex(1);
      if(vj->hessianIndex() >= 0){
        if(vj->marginalized()) {
          // Point
          // We know that we are sorted
            offsetRow = rowCount * rowsP;
            offsetCol = ((_numPoses) * colsC) + ((vj->hessianIndex() - _numPoses) * colsP);
            jacobiDataP.emplace_back(offsetRow + 0, offsetCol + 0,data[0]);
            jacobiDataP.emplace_back(offsetRow + 1, offsetCol + 0,data[1]);
            jacobiDataP.emplace_back(offsetRow + 0, offsetCol + 1,data[2]);
            jacobiDataP.emplace_back(offsetRow + 1, offsetCol + 1,data[3]);
            jacobiDataP.emplace_back(offsetRow + 0, offsetCol + 2,data[4]);
            jacobiDataP.emplace_back(offsetRow + 1, offsetCol + 2,data[5]);
        } else {
          // Camera
            offsetRow = rowCount * rowsC;
            offsetCol = (vj->hessianIndex()) * colsC;
            jacobiDataC.emplace_back(offsetRow + 0, offsetCol + 0,data[0]);
            jacobiDataC.emplace_back(offsetRow + 1, offsetCol + 0,data[1]);
            jacobiDataC.emplace_back(offsetRow + 0, offsetCol + 1,data[2]);
            jacobiDataC.emplace_back(offsetRow + 1, offsetCol + 1,data[3]);
            jacobiDataC.emplace_back(offsetRow + 0, offsetCol + 2,data[4]);
            jacobiDataC.emplace_back(offsetRow + 1, offsetCol + 2,data[5]);
            jacobiDataC.emplace_back(offsetRow + 0, offsetCol + 3,data[6]);
            jacobiDataC.emplace_back(offsetRow + 1, offsetCol + 3,data[7]);
            jacobiDataC.emplace_back(offsetRow + 0, offsetCol + 4,data[8]);
            jacobiDataC.emplace_back(offsetRow + 1, offsetCol + 4,data[9]);
            jacobiDataC.emplace_back(offsetRow + 0, offsetCol + 5,data[10]);
            jacobiDataC.emplace_back(offsetRow + 1, offsetCol + 5,data[11]);
        }
      }



      # ifndef NDEBUG
      for (size_t i = 0; i < e->vertices().size(); ++i) {
        const OptimizableGraph::Vertex* v = static_cast<const OptimizableGraph::Vertex*>(e->vertex(i));
        if (! v->fixed()) {
          bool hasANan = arrayHasNaN(jacobianWorkspace.workspaceForVertex(i), e->dimension() * v->dimension());
          if (hasANan) {
            std::cerr << "buildSystem(): NaN within Jacobian for edge " << e << " for vertex " << i << std::endl;
            break;
          }
        }
      }
      #  endif
  //}
    }
  }

  // flush the current system in a sparse block matrix
# ifdef G2O_OPENMP
# pragma omp parallel for default (shared) if (_optimizer->indexMapping().size() > 1000)
# endif

  int dimCam = (_numPoses) * 6;
  int dimPoints = (_numLandmarks) * 3;
  int rowDim = (rowCount + 1) * 2;


  std::cout << "P" << std::endl;
  for(auto const& t : jacobiDataP) {
    std::cout << t.row() << "," << t.col() << ":" << std::endl;
   }

  std::cout << std::endl << "C" << std::endl;
  for(auto const& t : jacobiDataC) {
    std::cout << t.row() << "," << t.col() << ":" << std::endl;
  }

  _JacobiP->resize(rowDim, dimCam + dimPoints);
  _JacobiP->setFromTriplets(jacobiDataP.begin(), jacobiDataP.end());
  _JacobiC->resize(rowDim, dimCam);
  _JacobiC->setFromTriplets(jacobiDataC.begin(), jacobiDataC.end());
  std::move(jacobiDataC.begin(), jacobiDataC.end(), std::back_inserter(jacobiDataP));

  Eigen::SparseMatrix<number_t> _jacobiFull(rowDim, dimCam + dimPoints);
  _jacobiFull.setFromTriplets(jacobiDataP.begin(), jacobiDataP.end());

  Eigen::SparseMatrix<number_t > _hessian(dimCam + dimPoints, dimCam + dimPoints);
  _hessian = (_jacobiFull.transpose()) * _jacobiFull;

  for (int i = 0; i < static_cast<int>(_optimizer->indexMapping().size()); ++i) {
    OptimizableGraph::Vertex* v=_optimizer->indexMapping()[i];
    int iBase = v->colInHessian();
    if (v->marginalized())
      iBase+=_sizePoses;
    v->copyB(_b+iBase);
  }

  saveMarket((*_JacobiP), "/home/lukas/Documents/eigenMatrices/jP.matx");
  saveMarket((*_JacobiC), "/home/lukas/Documents/eigenMatrices/jC.matx");
  saveMarket((_jacobiFull), "/home/lukas/Documents/eigenMatrices/jFull.matx");
  saveMarket((_hessian), "/home/lukas/Documents/eigenMatrices/hessian.matx");
  saveHessian("/home/lukas/Documents/eigenMatrices/hessian.txt");

  return false;
}


template <typename Traits>
bool JacobiSolver<Traits>::setLambda(number_t lambda, bool backup)
{
  if (backup) {
    _diagonalBackupPose.resize(_numPoses);
    _diagonalBackupLandmark.resize(_numLandmarks);
  }
# ifdef G2O_OPENMP
# pragma omp parallel for default (shared) if (_numPoses > 100)
# endif
  for (int i = 0; i < _numPoses; ++i) {
    PoseMatrixType *b=_Hpp->block(i,i);
    if (backup)
      _diagonalBackupPose[i] = b->diagonal();
    b->diagonal().array() += lambda;
  }
# ifdef G2O_OPENMP
# pragma omp parallel for default (shared) if (_numLandmarks > 100)
# endif
  for (int i = 0; i < _numLandmarks; ++i) {
    LandmarkMatrixType *b=_Hll->block(i,i);
    if (backup)
      _diagonalBackupLandmark[i] = b->diagonal();
    b->diagonal().array() += lambda;
  }
  return true;
}

template <typename Traits>
void JacobiSolver<Traits>::restoreDiagonal()
{
  assert((int) _diagonalBackupPose.size() == _numPoses && "Mismatch in dimensions");
  assert((int) _diagonalBackupLandmark.size() == _numLandmarks && "Mismatch in dimensions");
  for (int i = 0; i < _numPoses; ++i) {
    PoseMatrixType *b=_Hpp->block(i,i);
    b->diagonal() = _diagonalBackupPose[i];
  }
  for (int i = 0; i < _numLandmarks; ++i) {
    LandmarkMatrixType *b=_Hll->block(i,i);
    b->diagonal() = _diagonalBackupLandmark[i];
  }
}

template <typename Traits>
bool JacobiSolver<Traits>::init(SparseOptimizer* optimizer, bool online)
{
  _optimizer = optimizer;
  if (! online) {
    if (_Hpp)
      _Hpp->clear();
    if (_Hpl)
      _Hpl->clear();
    if (_Hll)
      _Hll->clear();
  }
  _linearSolver->init();
  return true;
}

template <typename Traits>
void JacobiSolver<Traits>::setWriteDebug(bool writeDebug)
{
  _linearSolver->setWriteDebug(writeDebug);
}

template <typename Traits>
bool JacobiSolver<Traits>::saveHessian(const std::string& fileName) const
{
	std::string pp = "pp";
	std::string ll = "ll";
	std::string pl = "pl";

    return _Hpp->writeOctave((fileName + pp).c_str(), true)
	  && _Hpl->writeOctave((fileName + pl).c_str(), true)
	  && _Hll->writeOctave((fileName + ll).c_str(), true);
}

} // end namespace

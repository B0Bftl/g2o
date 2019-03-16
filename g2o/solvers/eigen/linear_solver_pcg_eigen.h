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
#include <g2o/core/sparse_block_matrix.h>

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
	template<typename MatrixType>
	class LinearSolverPCGEigen : public LinearSolver<MatrixType> {
	public:
		typedef Eigen::SparseMatrix<number_t, Eigen::ColMajor> SparseMatrix;
		typedef Eigen::Triplet<number_t> Triplet;

	public:
		LinearSolverPCGEigen() :
				LinearSolver<MatrixType>(),
				_init(true), _writeDebug(false) {
		}

		virtual ~LinearSolverPCGEigen() {
		}

		virtual bool init() {
			_indices.clear();
			_sparseMat.clear();
			_init = true;
			return true;

		}


		bool solve(const SparseBlockMatrix <MatrixType> &A, number_t *x, number_t *b) {
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
		bool solve(Eigen::SparseMatrix<number_t> &J, number_t *x, number_t *b, int numCams, int numPoints,
		           int rowDim, int colDimCam, int colDimPoint, number_t lambda, number_t eta) {
			(void) rowDim;
			_numCams = numCams;
			_numPoints = numPoints;
			_colDimCam = colDimCam;
			_colDimPoint = colDimPoint;

			// Get Refs to x and b
			VectorX::MapType xVec(x, J.cols());
			VectorX::MapType bVec(b, J.cols());
			VectorX::MapType bCVec(b, _numCams* colDimCam);
			VectorX::MapType bPVec(b + _numCams *  colDimCam, _numPoints *  colDimPoint );


			G2OBatchStatistics *globalStats = G2OBatchStatistics::globalStats();
			number_t timeQR = get_monotonic_time();

			// Blocks of preconditioner R^-1.
			_Rc_Array = new Eigen::Matrix<number_t, 6, 6>[_numCams];
			_Rp_Array = new Eigen::Matrix<number_t, 3, 3>[_numPoints];


			// Compute preconditioner and store it in matrix array
			computeRc_inverse(J, _numCams, _Rc_Array);
			computeRp_inverse(J, _colDimPoint, _numCams, _numPoints, _Rp_Array, lambda);

			if (globalStats)
				globalStats->timeQrDecomposition = get_monotonic_time() - timeQR;

			//Computation of Preconditioner complete
			// Initialize Algorithm.

			// Reference b
			Eigen::VectorXd _precond_b(bVec.rows());
			// Map Vector b in Camera and Position Part.
			VectorX::MapType bC(_precond_b.data(), _numCams*_colDimCam);
			VectorX::MapType bP(_precond_b.data() + _numCams * _colDimCam, _numPoints * _colDimPoint );

			// apply Preconditioning with R_inv to b
			// _precond_b = R_inv.transpose() * bVec
			mult_RcT_Vec(bCVec, bC);
			mult_RpT_Vec(bPVec, bP);

			// get Jc and Jp from J
			const Eigen::Ref<const Eigen::SparseMatrix<number_t>> Jc = J.leftCols(_numCams * _colDimCam);
			const Eigen::Ref<const Eigen::SparseMatrix<number_t>> Jp = J.rightCols(_numPoints * _colDimPoint);
.

			// Map Vector x in Camera and Position Part. Writing to xC/xP writes to x
			VectorX::MapType xC(x, _numCams * _colDimCam);
			VectorX::MapType xP(x + _numCams * _colDimCam, _numPoints * _colDimPoint);

			VectorX s(xVec.rows());

			Eigen::Map<Eigen::VectorXd> sC(s.data(), _numCams * _colDimCam);
			Eigen::Map<Eigen::VectorXd> sP(s.data() + _numCams * _colDimCam, _numPoints * _colDimPoint);



			xC.setZero();
			xP = bP;
			//Eigen::VectorXd p = _precond_b - _precondJ.transpose() * (_precondJ * xVec);

			// to minimize memory reallocation, use standard tmp vectors for manual aliasing. Need 2 sizes
			Eigen::VectorXd tmpLong(J.rows());
			Eigen::VectorXd tmpShort(xVec.rows());
			Eigen::Map<Eigen::VectorXd> tmpShortC(tmpShort.data(), _numCams * _colDimCam);
			Eigen::Map<Eigen::VectorXd> tmpShortP(tmpShort.data() + _numCams * _colDimCam, _numPoints * _colDimPoint);

			Eigen::VectorXd tmpShort_2(xVec.rows());
			Eigen::Map<Eigen::VectorXd> tmpShortC_2(tmpShort_2.data(), _numCams * _colDimCam);
			Eigen::Map<Eigen::VectorXd> tmpShortP_2(tmpShort_2.data() + _numCams * _colDimCam, _numPoints * _colDimPoint);


			//p =_precond_b - R_inv.transpose() *  (J.transpose() * (J * (R_inv * xVec)));
			Eigen::VectorXd p = _precond_b;
			Eigen::Map<Eigen::VectorXd> pC(p.data(), _numCams * _colDimCam);
			Eigen::Map<Eigen::VectorXd> pP(p.data() + _numCams * _colDimCam, _numPoints * _colDimPoint);




			// tmpShort.noalias() = R_inv * xVec;
			mult_Rc_Vec(xC, tmpShortC);
			mult_Rp_Vec(xP, tmpShortP);
			tmpLong.noalias() = J * tmpShort;
			tmpShort.noalias() = J.transpose() * tmpLong;
			mult_RcT_Vec(tmpShortC, tmpShortC_2);
			mult_RpT_Vec(tmpShortP, tmpShortP_2);
			p.noalias() -= tmpShort_2;

			s = p;

			number_t beta;

			//Eigen::VectorXd q = J * (R_inv * p);

			//tmpShort.noalias() = R_inv * p;
			mult_Rc_Vec(pC, tmpShortC);
			mult_Rp_Vec(pP, tmpShortP);
			Eigen::VectorXd q = J * tmpShort;


			long maxIter = J.rows();

			long iteration = 0;
			bool isEven = false;

			number_t alpha;

			number_t gamma = s.dot(s);
			number_t gamma_old = gamma;

			// compute Initial error
			number_t scaledInitialError = eta * s.dot(s);

			// Initialization complete, start iterating
			for (iteration = 0; iteration < maxIter + maxIter % 2; ++iteration) {
				// check if error small enough according to eta
				if (gamma < scaledInitialError)
					break;

				isEven = !static_cast<bool>(iteration % 2);

				alpha = gamma / q.dot(q);
				xVec.noalias() += alpha * p;
				if (!isEven) {
					//odd
					//sC.noalias() = (-1) * alpha * Rc.transpose() * (Jc.transpose() * q);
					tmpShortC.noalias() = (-1) * alpha * Jc.transpose() * q;
					mult_RcT_Vec(tmpShortC, sC);
					sP.setZero();
				} else {
					//even
					// sP.noalias() = (-1) * alpha * Rp.transpose() * (Jp.transpose() * q);
					tmpShortP.noalias() = (-1) * alpha * Jp.transpose() * q;
					mult_RpT_Vec(tmpShortP, sP);
					sC.setZero();
				}

				gamma = s.dot(s);
				beta = gamma / gamma_old;
				gamma_old = gamma;
				//p = s + beta * p;
				tmpShort.noalias() = beta * p;
				p.noalias() = s + tmpShort;

				if (!isEven) {
					//odd
					//q = beta * q + Jc * sC;
					// q.noalias() += Jc * (Rc * sC);
					tmpLong.noalias() = beta * q;
					mult_Rc_Vec(sC, tmpShortC);
					q.noalias() = tmpLong;
					q.noalias() += Jc * tmpShortC;
				} else {
					// even
					//q = beta * q + Jp * sP;
					// q.noalias() += Jp * (Rp * sP);
					//q = beta * q;
					tmpLong.noalias() = beta * q;
					mult_Rp_Vec(sP, tmpShortP);
					q.noalias() = tmpLong;
					q.noalias() += Jp * tmpShortP;
				}
			}
			// Retreive correct x from transformed x
			//	tmpShort.noalias() = R_inv * xVec;
			mult_Rc_Vec(xC,tmpShortC);
			mult_Rp_Vec(xP,tmpShortP);
			xVec = tmpShort;
			if (globalStats) {
				globalStats->iterationsLinearSolver = iteration;
			}

			delete[] _Rc_Array;
			delete[] _Rp_Array;
			return true;
		}


		//! write a debug dump of the system matrix if it is not SPD in solve
		virtual bool writeDebug() const { return _writeDebug; }

		virtual void setWriteDebug(bool b) { _writeDebug = b; }

	protected:
		bool _init;
		bool _writeDebug = false;
		int _maxDegree = -1;
		std::vector<std::pair<int, int> > _indices;
		typedef std::vector<const MatrixType *> MatrixPtrVector;
		MatrixPtrVector _sparseMat;

		int _numCams;
		int _numPoints;
		int _colDimCam;
		int _colDimPoint;
		Eigen::Matrix<number_t, 6, 6> *_Rc_Array;
		Eigen::Matrix<number_t, 3, 3> *_Rp_Array;

		/**
		 * Computes maximum degree to know required memory size during computation of Rc
		 * @tparam Derived
		 * @param _matrix
		 * @param numCams
		 */
		template<typename Derived>
		inline void getMaxDegree(Eigen::SparseMatrixBase<Derived> &_matrix, int numCams) {

			#ifdef G2O_OPENMP
			#pragma omp parallel
			{
			#endif
			int max = -1;
			#ifdef G2O_OPENMP
			#pragma omp for schedule(runtime)
			#endif
			for (int i = 0; i < numCams; ++i) {
				if (max < (_matrix.col(i * 6).nonZeros() - 1) / 2)
					max = (_matrix.col(i * 6).nonZeros() - 1) / 2;
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

		/**
		 *  Carries out the multiplication dest = Rc * src
		 * 	Rc must have been computed beforehand
		 *  IMPORTANT: UNDEFINED BEHAVIOUR IF src AND dest POINT TO SAME MEMORY
		 * @param src	Vector to be multiplied with
		 * @param dest	Vector to store the result
		 */
		inline void mult_Rc_Vec(Eigen::Map<VectorX>& src, Eigen::Map<VectorX>& dest) {
#ifdef G2O_OPENMP
#pragma omp parallel for schedule(runtime)
#endif
			for (int i = 0; i < _numCams; ++i) {
				dest.segment<6>(i * 6).noalias() = _Rc_Array[i] * src.segment<6>(i * 6);
			}
		}

		/**
		 *  Carries out the multiplication dest = Rp * src
		 * 	Rp must have been computed beforehand
		 *  IMPORTANT: UNDEFINED BEHAVIOUR IF src AND dest POINT TO SAME MEMORY
		 * @param src	Vector to be multiplied with
		 * @param dest	Vector to store the result
		 */
		inline void mult_Rp_Vec(Eigen::Map<VectorX>& src, Eigen::Map<VectorX>& dest) {
#ifdef G2O_OPENMP
#pragma omp parallel for schedule(runtime)
#endif
			for (int i = 0; i < _numPoints; ++i) {
				dest.segment<3>(i * 3).noalias() = _Rp_Array[i] * src.segment<3>(i * 3);
			}
		}

		/**
		 *  Carries out the multiplication dest = Rc^T * src
		 * 	Rc must have been computed beforehand
		 *  IMPORTANT: UNDEFINED BEHAVIOUR IF src AND dest POINT TO SAME MEMORY
		 * @param src	Vector to be multiplied with
		 * @param dest	Vector to store the result
		 */
		inline void mult_RcT_Vec(Eigen::Map<VectorX>& src, Eigen::Map<VectorX>& dest) {
#ifdef G2O_OPENMP
#pragma omp parallel for schedule(runtime)
#endif
			for (int i = 0; i < _numCams; ++i) {
				dest.segment<6>(i * 6).noalias() = _Rc_Array[i].transpose() * src.segment<6>(i * 6);
			}
		}

		/**
		 *  Carries out the multiplication dest = Rp^T * src
		 * 	Rp must have been computed beforehand
		 *  IMPORTANT: UNDEFINED BEHAVIOUR IF src AND dest POINT TO SAME MEMORY
		 * @param src	Vector to be multiplied with
		 * @param dest	Vector to store the result
		 */
		inline void mult_RpT_Vec(Eigen::Map<VectorX>& src, Eigen::Map<VectorX>& dest) {
#ifdef G2O_OPENMP
#pragma omp parallel for schedule(runtime)
#endif
			for (int i = 0; i < _numPoints; ++i) {
				dest.segment<3>(i * 3).noalias() = _Rp_Array[i].transpose() * src.segment<3>(i * 3);
			}
		}

		/**
		 * Computes preconditioner Rc and stores it in Rc_Array
		 * @tparam Derived	matrix type
		 * @param _matrix	Jacobi J
		 * @param numCams	number of cameras
		 * @param Rc_Array	Pointer to initialized array to store result
		 */
		template<typename Derived>
		inline void computeRc_inverse(Eigen::SparseMatrixBase<Derived> &_matrix, int numCams,
		                              Eigen::Matrix<number_t, 6, 6> *Rc_Array) {
			//std::vector<Eigen::Triplet<number_t>, Eigen::aligned_allocator<Eigen::Triplet<number_t >>>& coeffR) {
			// qr decomp

			//long base = 0;
			std::vector<number_t, Eigen::aligned_allocator<number_t>> coeffBlock;

			if (_maxDegree == -1)
				getMaxDegree(_matrix, numCams);

			#ifdef G2O_OPENMP
			# pragma omp parallel default (shared) firstprivate(coeffBlock)
						{
			#endif
			Eigen::SparseMatrix<number_t> &matrix = static_cast<Eigen::SparseMatrix<number_t> &> (_matrix);
			coeffBlock.resize(6 * 2 * _maxDegree + 6 * 6);
			#ifdef G2O_OPENMP
			#pragma omp for schedule(runtime)
			#endif
			for (int k = 0; k < numCams * 6; k += 6) {
				// Fetch current block. store coefficients
				size_t blocksize = 0;
				Eigen::SparseMatrix<number_t>::InnerIterator itA(matrix, k);
				Eigen::SparseMatrix<number_t>::InnerIterator itB(matrix, k + 1);
				Eigen::SparseMatrix<number_t>::InnerIterator itC(matrix, k + 2);
				Eigen::SparseMatrix<number_t>::InnerIterator itD(matrix, k + 3);
				Eigen::SparseMatrix<number_t>::InnerIterator itE(matrix, k + 4);
				Eigen::SparseMatrix<number_t>::InnerIterator itF(matrix, k + 5);
				while (itA && itB && itC && itD && itE && itF && itA.row() == itB.row()) {

					assert(itA.row() == itB.row()
					       && itA.row() == itC.row()
					       && itA.row() == itD.row()
					       && itA.row() == itE.row()
					       && itA.row() == itF.row());

					coeffBlock[6 * blocksize] = itA.value();
					coeffBlock[6 * blocksize + 1] = itB.value();
					coeffBlock[6 * blocksize + 2] = itC.value();
					coeffBlock[6 * blocksize + 3] = itD.value();
					coeffBlock[6 * blocksize + 4] = itE.value();
					coeffBlock[6 * blocksize + 5] = itF.value();

					++itA;
					++itB;
					++itC;
					++itD;
					++itE;
					++itF;

					++blocksize;
				}
				// add lambda entries. All other entries are 0 by initialization
				assert(6 * (blocksize + 5) + 5 <= coeffBlock.size());
				// clear from previous iteration
				for (unsigned long j = 6 * (blocksize); j < 6 * (blocksize + 5) + 5; ++j) coeffBlock[j] = 0;

				coeffBlock[6 * (blocksize++)] = itA.value();
				coeffBlock[6 * (blocksize++) + 1] = itB.value();
				coeffBlock[6 * (blocksize++) + 2] = itC.value();
				coeffBlock[6 * (blocksize++) + 3] = itD.value();
				coeffBlock[6 * (blocksize++) + 4] = itE.value();
				coeffBlock[6 * (blocksize++) + 5] = itF.value();

				// map coefficients to dense matrix
				Eigen::Map<Eigen::Matrix<number_t, Eigen::Dynamic, 6, Eigen::ColMajor>, 0, Eigen::Stride<1, 6> >
						currentBlock(coeffBlock.data(), blocksize, 6, Eigen::Stride<1, 6>(1, 6));

				// compute qr of dense matrix and retreive R, write into Array
				Rc_Array[k/6] = static_cast<Eigen::Matrix<number_t, Eigen::Dynamic, 6>>
				(currentBlock.householderQr().matrixQR().triangularView<Eigen::Upper>()).block<6, 6>(0, 0);

				// get inverse
				Rc_Array[k / 6] = (Rc_Array[k / 6]).inverse().eval();

			}
		#ifdef G2O_OPENMP
					} //close parallel region
		#endif
		}

		/**
		 *
		 * Computes preconditioner Rp and stores it in Rp_Array
		 *
		 * @tparam Derived matrix type
		 * @param _matrix	Jacobi
		 * @param colDim	colDimension of p
		 * @param numCams	number of cams
		 * @param numPoints	number of points
		 * @param Rp_Array	pointer to initialized array to store Rp
		 * @param lambda	Lambda of LM algorithm
		 */
		template<typename Derived>
		inline void computeRp_inverse(Eigen::SparseMatrixBase<Derived> &_matrix, int colDim, int numCams, int numPoints,
		                              Eigen::Matrix<number_t, 3, 3> *Rp_Array, number_t lambda) {

			Eigen::SparseMatrix<number_t> &matrix = static_cast<Eigen::SparseMatrix<number_t> &> (_matrix);

		#ifdef G2O_OPENMP
		# pragma omp parallel default (shared)
					{
		#endif
			int rowOffset = 0;
			int blockSize = 0;
			int colIndex = 0;

		#ifdef G2O_OPENMP
		#pragma omp for schedule(runtime)
		#endif
			for (int i = 0; i < numPoints; ++i) {
				colIndex = i * 3 + numCams * 6;

				Eigen::SparseMatrix<number_t>::InnerIterator iterRow(matrix, colIndex);

				// block starts at first non zero entry and ends at last (-1 because of lambda).
				rowOffset = iterRow.row();
				blockSize = matrix.col(colIndex).nonZeros() - 1;

				Eigen::Matrix<number_t, Eigen::Dynamic, 3> currentBlock(blockSize + colDim, colDim);
				// get current block. Row offset based on previous blocks, col offset on position in jacobian
				currentBlock.topRows(blockSize) = matrix.block(rowOffset, colIndex, blockSize, colDim);
				// attach lambda scaling
				currentBlock.bottomRows(colDim) = Eigen::Matrix<number_t, 3, 3>::Identity(colDim, colDim) * lambda;

				// initialize & compute qr in place, store in array
				Rp_Array[i] = static_cast<Eigen::Matrix<number_t, Eigen::Dynamic, 3>>(currentBlock.householderQr().matrixQR().triangularView<Eigen::Upper>()).block<3, 3>(
						0, 0);

				// get inverse
				Rp_Array[i] = (Rp_Array[i]).inverse().eval();

			}

#ifdef G2O_OPENMP
			}
#endif

		}

		/**
		 * Prints given dense matrix. For debug purposes
		 * @tparam Derived
		 * @param matrix
		 * @param name
		 * @param compact
		 */
		template<typename Derived>
		void printMatrix(const Eigen::MatrixBase<Derived> &matrix, std::string name, bool compact = true) {
			std::cout << "printing " << name << std::endl << "----------------------------" << std::endl;
			if (compact) {
				for (int j = 0; j < matrix.rows(); ++j) {
					for (int k = 0; k < matrix.cols(); ++k) {
						std::cout << "   " << matrix.coeff(j, k);
					}
					std::cout << std::endl;
				}
			} else {
				for (int j = 0; j < matrix.cols(); ++j) {
					for (int k = 0; k < matrix.rows(); ++k) {
						std::cout << k << " x " << j << ": " << matrix.coeff(k, j) << std::endl;
					}
				}
			}
			std::cout << "----------------------------" << std::endl << std::endl;
		}

		/**
		 * prints given sparse Matrix. For debug purposes
		 * @tparam Derived
		 * @param matrix
		 * @param name
		 * @param compact
		 */
		template<typename Derived>
		void printMatrix(const Eigen::SparseMatrixBase<Derived> &matrix, std::string name, bool compact = true) {
			Eigen::MatrixXd denseMat = matrix;
			printMatrix(denseMat, name, compact);
		}

	};
}
#endif

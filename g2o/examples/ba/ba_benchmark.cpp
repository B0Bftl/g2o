// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
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

#include <Eigen/StdVector>
#include <iostream>
#include <stdint.h>

#include <unordered_set>

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/jacobi_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/eigen/linear_solver_pcg_eigen.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

#include "g2o/solvers/structure_only/structure_only_solver.h"
#include <g2o/core/batch_stats.h>
#include <g2o/apps/g2o_cli/dl_wrapper.h>
#include <g2o/apps/g2o_cli/g2o_common.h>

#include "g2o/stuff/command_args.h"
#include "g2o/stuff/filesys_tools.h"
#include "g2o/stuff/string_tools.h"


using namespace Eigen;
using namespace std;



int main(int argc, char** argv){
  if (argc<6)
  {
    cout << endl;
    cout << "Please type: " << endl;
    cout << "ba_benchmark [FILENAME] [ITERATIONS_PCG] [ITERATIONS_CHOL] [ROUNDS] [STATFILE] [ETA]" << endl;
    cout << endl;
    cout << "FILENAME: File to load graph from" << endl;
    cout << "ITERATIONS_PCG: number of iterations for pcg solver" << endl;
    cout << "ITERATIONS_CHOL: number of iterations for cholesky solver" << endl;
    cout << "ROUNDS: number of runs" << endl;
    cout << "STATFILE: File to save stats to" << endl;
    cout << "ETA: adjust accuracy of pcg solver" << endl;
    cout << endl;
    cout << endl;
    exit(0);
  }

  std::string graphFile = "";
  int iterations_pcg = 6;
  int iterations_chol = 5;
  int rounds = 100;
  std::string statsFile = "";
  number_t eta = 0.1;


  graphFile = (argv[1]);
  iterations_pcg = atoi(argv[2]);
  iterations_chol = atoi(argv[3]);
  rounds = atoi(argv[4]);
  statsFile = argv[5];
  if (argc > 6)
  	eta = atof(argv[6]);

  cout << "Graphfile: " <<  graphFile << endl;
  cout << "Iterations PCG: "<<  iterations_pcg << endl;
  cout << "Iterations Chol: "<<  iterations_chol << endl;
  cout << "Rounds: " << rounds << endl;
  cout << "Stats File: "<<  statsFile << endl;
  cout << "Eta : "<<  eta << endl;


	g2o::DlWrapper dlTypesWrapper;
  g2o::loadStandardTypes(dlTypesWrapper, argc, argv);
  // register all the solvers
  g2o::DlWrapper dlSolverWrapper;
  g2o::loadStandardSolver(dlSolverWrapper, argc, argv);

  //Setup 2 solver

  for (int i = 0; i < rounds; ++i) {

    g2o::SparseOptimizer optimizerPCG;

    optimizerPCG.setVerbose(false);

    std::unique_ptr<g2o::JacobiSolver_6_3::LinearSolverType> linearSolverPCG = g2o::make_unique<g2o::LinearSolverPCGEigen<g2o::JacobiSolver_6_3::PoseMatrixType>>();

    linearSolverPCG->eta = eta;

    g2o::OptimizationAlgorithmLevenberg* solverPCG = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<g2o::JacobiSolver_6_3>(std::move(linearSolverPCG)));

    optimizerPCG.setAlgorithm(solverPCG);


    if(statsFile.length() > 0) {
      optimizerPCG.setComputeBatchStatistics(true);
    }

    //load graph & optimize
    if(!optimizerPCG.load(graphFile.c_str(), true)) {
      std::cerr << "failed to load file for PCG";
      return -1;
    }


    for (g2o::HyperGraph::VertexIDMap::iterator it=optimizerPCG.vertices().begin(); it!=optimizerPCG.vertices().end(); it++){
      g2o::OptimizableGraph::Vertex* v=static_cast<g2o::OptimizableGraph::Vertex*>(it->second);
      if (v->dimension() != 6) {
        v->setMarginalized(true);
      }
    }

	solverPCG->getLinSolver()->setEta(eta);
    optimizerPCG.initializeOptimization(0);
    optimizerPCG.computeActiveErrors();
    std::cout << "Initial chi: " << optimizerPCG.chi2() << std::endl;
    optimizerPCG.optimize(iterations_pcg);


    if (statsFile!=""){
      std::string pcgFile = statsFile + "_pcg.txt";

      ofstream foutPCG(pcgFile.c_str(), ios_base::app);
      const g2o::BatchStatisticsContainer& bscPCG = optimizerPCG.batchStatistics();


      for (size_t i=0; i<bscPCG.size(); i++)
        foutPCG << bscPCG[i] << endl;

    }
  }

  for (int i = 0; i < rounds; ++i) {

		g2o::SparseOptimizer optimizerChol;

		optimizerChol.setVerbose(false);

		std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolverCholesky = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3 ::PoseMatrixType>>();

		g2o::OptimizationAlgorithmLevenberg* solverChol = new g2o::OptimizationAlgorithmLevenberg(
				g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolverCholesky)));


		optimizerChol.setAlgorithm(solverChol);

		if(statsFile.length() > 0) {
			optimizerChol.setComputeBatchStatistics(true);
		}

		//load graph & optimize
		if(!optimizerChol.load(graphFile.c_str(), true)) {
			std::cerr << "failed to load file for PCG";
			return -1;
		}

		for (g2o::HyperGraph::VertexIDMap::iterator it=optimizerChol.vertices().begin(); it!=optimizerChol.vertices().end(); it++){
			g2o::OptimizableGraph::Vertex* v=static_cast<g2o::OptimizableGraph::Vertex*>(it->second);
			if (v->dimension() != 6) {
				v->setMarginalized(true);
			}
		}

		optimizerChol.initializeOptimization(0);
	    optimizerChol.computeActiveErrors();
	    std::cout << "Initial chi: " << optimizerChol.chi2() << std::endl;
		optimizerChol.optimize(iterations_chol);

		if (statsFile!=""){
			std::string cholFile = statsFile + "_chol.txt";

			ofstream foutChol(cholFile.c_str(), ios_base::app);
			const g2o::BatchStatisticsContainer& bscChol = optimizerChol.batchStatistics();

			for (size_t i=0; i<bscChol.size(); i++)
				foutChol << bscChol[i] << endl;
		}
	}

	//variable block size
	for (int i = 0; i < rounds; ++i) {

		g2o::SparseOptimizer optimizerChol;

		optimizerChol.setVerbose(false);

		std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolverCholesky = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();

		g2o::OptimizationAlgorithmLevenberg* solverChol = new g2o::OptimizationAlgorithmLevenberg(
				g2o::make_unique<g2o::BlockSolverX>(std::move(linearSolverCholesky)));


		optimizerChol.setAlgorithm(solverChol);

		if(statsFile.length() > 0) {
			optimizerChol.setComputeBatchStatistics(true);
		}

		//load graph & optimize
		if(!optimizerChol.load(graphFile.c_str(), true)) {
			std::cerr << "failed to load file for PCG";
			return -1;
		}

		for (g2o::HyperGraph::VertexIDMap::iterator it=optimizerChol.vertices().begin(); it!=optimizerChol.vertices().end(); it++){
			g2o::OptimizableGraph::Vertex* v=static_cast<g2o::OptimizableGraph::Vertex*>(it->second);
			if (v->dimension() != 6) {
				v->setMarginalized(true);
			}
		}

		optimizerChol.initializeOptimization(0);
		optimizerChol.computeActiveErrors();
		std::cout << "Initial chi: " << optimizerChol.chi2() << std::endl;
		optimizerChol.optimize(iterations_chol);

		if (statsFile!=""){
			std::string cholFile = statsFile + "_chol_var.txt";

			ofstream foutChol(cholFile.c_str(), ios_base::app);
			const g2o::BatchStatisticsContainer& bscChol = optimizerChol.batchStatistics();

			for (size_t i=0; i<bscChol.size(); i++)
				foutChol << bscChol[i] << endl;
		}
	}


}

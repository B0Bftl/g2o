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
//#include "g2o/math_groups/se3quat.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include <g2o/core/batch_stats.h>

using namespace Eigen;
using namespace std;


class Sample {
public:
  static int uniform(int from, int to);
  static double uniform();
  static double gaussian(double sigma);
};

static double uniform_rand(double lowerBndr, double upperBndr){
  return lowerBndr + ((double) std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}

static double gauss_rand(double mean, double sigma){
  double x, y, r2;
  do {
    x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    r2 = x * x + y * y;
  } while (r2 > 1.0 || r2 == 0.0);
  return mean + sigma * y * std::sqrt(-2.0 * log(r2) / r2);
}

int Sample::uniform(int from, int to){
  return static_cast<int>(uniform_rand(from, to));
}

double Sample::uniform(){
  return uniform_rand(0., 1.);
}

double Sample::gaussian(double sigma){
  return gauss_rand(0., sigma);
}

int main(int argc, const char* argv[]){
  if (argc<2)
  {
    cout << endl;
    cout << "Please type: " << endl;
    cout << "ba_demo [PIXEL_NOISE] [OUTLIER RATIO] [ROBUST_KERNEL] [STRUCTURE_ONLY] [DENSE]" << endl;
    cout << endl;
    cout << "PIXEL_NOISE: noise in image space (E.g.: 1)" << endl;
    cout << "OUTLIER_RATIO: probability of spuroius observation  (default: 0.0)" << endl;
    cout << "ROBUST_KERNEL: use robust kernel (0 or 1; default: 0==false)" << endl;
    cout << "STRUCTURE_ONLY: performe structure-only BA to get better point initializations (0 or 1; default: 0==false)" << endl;
    cout << "Linsolver: 0-3" << endl;
    cout << "numCameras" << endl;
    cout << "numPoints" << endl;
    cout << endl;
    cout << "Note, if OUTLIER_RATIO is above 0, ROBUST_KERNEL should be set to 1==true." << endl;
    cout << endl;
    exit(0);
  }

  double PIXEL_NOISE = atof(argv[1]);
  double OUTLIER_RATIO = 0.0;

  if (argc>2)  {
    OUTLIER_RATIO = atof(argv[2]);
  }

  bool ROBUST_KERNEL = false;
  if (argc>3){
    ROBUST_KERNEL = atoi(argv[3]) != 0;
  }
  bool STRUCTURE_ONLY = false;
  if (argc>4){
    STRUCTURE_ONLY = atoi(argv[4]) != 0;
  }

  int linSolver = 0;
  if (argc>5){
    linSolver = atoi(argv[5]);
  }

  int numCameras = 15;
	if (argc>6){
		numCameras = atoi(argv[6]);
	}

	int numPoints = 300;
	if (argc>7){
		numPoints = atoi(argv[7]);
	}
  int iterations = 10;
  if (argc>8){
    iterations = atoi(argv[8]);
  }
  std::string statsFilename = "";
  if (argc>8){
    statsFilename = (argv[9]);
  }


  cout << "PIXEL_NOISE: " <<  PIXEL_NOISE << endl;
  cout << "OUTLIER_RATIO: " << OUTLIER_RATIO<<  endl;
  cout << "ROBUST_KERNEL: " << ROBUST_KERNEL << endl;
  cout << "STRUCTURE_ONLY: " << STRUCTURE_ONLY<< endl;
  cout << "LinSolver: "<<  linSolver << endl;
  cout << "Cameras: "<<  numCameras << endl;
  cout << "Points: "<<  numPoints << endl;
  cout << "Iterations: "<<  iterations << endl;
  cout << "Stats File: "<<  statsFilename << endl;





  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);
  std::unique_ptr<g2o::JacobiSolver_6_3 ::LinearSolverType> linearSolver;
  if (linSolver == 0) {
    linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::JacobiSolver_6_3::PoseMatrixType>>();
  } else if(linSolver == 1) {
    linearSolver = g2o::make_unique<g2o::LinearSolverEigen<g2o::JacobiSolver_6_3::PoseMatrixType>>();
  } else if(linSolver == 2) {
    linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::JacobiSolver_6_3::PoseMatrixType>>();
  } else if(linSolver == 3) {
    linearSolver = g2o::make_unique<g2o::LinearSolverPCG<g2o::JacobiSolver_6_3::PoseMatrixType>>();
  } else {
    linearSolver = g2o::make_unique<g2o::LinearSolverPCGEigen<g2o::JacobiSolver_6_3::PoseMatrixType>>();
  }


  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<g2o::JacobiSolver_6_3>(std::move(linearSolver))
  );
  optimizer.setAlgorithm(solver);
  if(statsFilename.length() > 0)
    optimizer.setComputeBatchStatistics(true);

  vector<Vector3d> true_points;
  for (int i=0;i<numPoints; ++i)
  {
    true_points.push_back(Vector3d((Sample::uniform()-0.5)*3,
                                   Sample::uniform()-0.5,
                                   Sample::uniform()+3));
  }

  double focal_length= 1000.;
  double cx = 320.;
  double cy = 240.;
  Vector2d principal_point(320., 240.);

  vector<g2o::SE3Quat,
      aligned_allocator<g2o::SE3Quat> > true_poses;


  g2o::CameraParameters * cam_params
      = new g2o::CameraParameters (focal_length, principal_point, 0.);
  cam_params->setId(0);

  if (!optimizer.addParameter(cam_params)) {
    assert(false);
  }
  // Cameras
  int vertex_id = 0;
  for (int i=0; i<numCameras; ++i) {
    Vector3d trans(i*0.04-1.,0,0);

    Eigen:: Quaterniond q;
    q.setIdentity();
    g2o::SE3Quat pose(q,trans);
    g2o::VertexSE3Expmap * v_se3
        = new g2o::VertexSE3Expmap();
    v_se3->setId(vertex_id);
    if (i<1){
      v_se3->setFixed(true);
    }
    v_se3->setEstimate(pose);

    optimizer.addVertex(v_se3);
    true_poses.push_back(pose);
    vertex_id++;
  }
  int point_id=vertex_id;
  int point_num = 0;
  double sum_diff2 = 0;

  cout << endl;
  unordered_map<int,int> pointid_2_trueid;
  unordered_set<int> inliers;

  // Points
  for (size_t i=0; i<true_points.size(); ++i){
    g2o::VertexSBAPointXYZ * v_p
        = new g2o::VertexSBAPointXYZ();
    v_p->setId(point_id);
    v_p->setMarginalized(true);
    v_p->setEstimate(true_points.at(i)
                     + Vector3d(Sample::gaussian(1),
                                Sample::gaussian(1),
                                Sample::gaussian(1)));
    int num_obs = 0;
    for (size_t j=0; j<true_poses.size(); ++j){
      Vector2d z = cam_params->cam_map(true_poses.at(j).map(true_points.at(i)));
      if (z[0]>=0 && z[1]>=0 && z[0]<640 && z[1]<480){
        ++num_obs;
      }
    }
    if (num_obs>=2){
      optimizer.addVertex(v_p);
      bool inlier = true;
      for (size_t j=0; j<true_poses.size(); ++j){
        Vector2d z
            = cam_params->cam_map(true_poses.at(j).map(true_points.at(i)));

        if (z[0]>=0 && z[1]>=0 && z[0]<640 && z[1]<480){
          double sam = Sample::uniform();
          if (sam<OUTLIER_RATIO){
            z = Vector2d(Sample::uniform(0,640),
                         Sample::uniform(0,480));
            inlier= false;
          }
          z += Vector2d(Sample::gaussian(PIXEL_NOISE),
                        Sample::gaussian(PIXEL_NOISE));
          g2o::EdgeSE3ProjectXYZ * e
              = new g2o::EdgeSE3ProjectXYZ();
          e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p));
          e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>
                       (optimizer.vertices().find(j)->second));
          e->setMeasurement(z);
          e->information() = Matrix2d::Identity();
          if (ROBUST_KERNEL) {
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            e->setRobustKernel(rk);
          }
          e->setParameterId(0, 0);
          e->fx = focal_length;
          e->fy = focal_length;
          e->cx = cx;
          e->cy = cy;

          optimizer.addEdge(e);
        }
      }

      if (inlier){
        inliers.insert(point_id);
        Vector3d diff = v_p->estimate() - true_points[i];

        sum_diff2 += diff.dot(diff);
      }
      pointid_2_trueid.insert(make_pair(point_id,i));
      ++point_id;
      ++point_num;
    }
  }
  cout << endl;
  optimizer.initializeOptimization();
  optimizer.setVerbose(true);
  if (STRUCTURE_ONLY){
    g2o::StructureOnlySolver<3> structure_only_ba;
    cout << "Performing structure-only BA:"   << endl;
    g2o::OptimizableGraph::VertexContainer points;
    for (g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it) {
      g2o::OptimizableGraph::Vertex* v = static_cast<g2o::OptimizableGraph::Vertex*>(it->second);
      if (v->dimension() == 3)
        points.push_back(v);
    }
    structure_only_ba.calc(points, 10);
  }
  //optimizer.save("test.g2o");
  cout << endl;
  cout << "Performing full BA:" << endl;
  optimizer.optimize(iterations);
  cout << endl;
  cout << "Point error before optimisation (inliers only): " << sqrt(sum_diff2/inliers.size()) << endl;
  point_num = 0;
  sum_diff2 = 0;
  for (unordered_map<int,int>::iterator it=pointid_2_trueid.begin();
       it!=pointid_2_trueid.end(); ++it){
    g2o::HyperGraph::VertexIDMap::iterator v_it
        = optimizer.vertices().find(it->first);
    if (v_it==optimizer.vertices().end()){
      cerr << "Vertex " << it->first << " not in graph!" << endl;
      exit(-1);
    }
    g2o::VertexSBAPointXYZ * v_p
        = dynamic_cast< g2o::VertexSBAPointXYZ * > (v_it->second);
    if (v_p==0){
      cerr << "Vertex " << it->first << "is not a PointXYZ!" << endl;
      exit(-1);
    }
    Vector3d diff = v_p->estimate()-true_points[it->second];
    if (inliers.find(it->first)==inliers.end())
      continue;

    sum_diff2 += diff.dot(diff);
    ++point_num;
  }
  cout << "Point error after optimisation (inliers only): " << sqrt(sum_diff2/inliers.size()) << endl;
  cout << endl;

  if (statsFilename!=""){
    cerr << "writing stats to file \"" << statsFilename << "\" ... ";
    ofstream fout(statsFilename.c_str());
    const g2o::BatchStatisticsContainer& bsc = optimizer.batchStatistics();
    for (size_t i=0; i<bsc.size(); i++)
      fout << bsc[i] << endl;
    cerr << "done." << endl;
  }

}

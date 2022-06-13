//
// Created by Will on 13/06/2022.
//

#include <iostream>
#include <Eigen/Dense>
#include "PreProccessingUtils.h"

using namespace std;
using namespace Eigen;

tuple<VectorXd, MatrixXd> splitLabelsFromFeatures(const MatrixXd& A) {
    VectorXd labels = A(all, 0);
    MatrixXd features = A(all, seqN(1, last));

    return make_tuple(labels, features.transpose());
}

MatrixXd oneHotEncode(const VectorXd& labels) {
    // (m x 10) matrix of zeros
    MatrixXd one_hot_y = MatrixXd::Zero(labels.rows(), (labels.maxCoeff() + 1));

    for(int i = 0; i < labels.rows(); i++) {
        one_hot_y(i, (int) labels(i)) = 1;
    }

    // return a (10 x m) matrix
    return one_hot_y.transpose();
}

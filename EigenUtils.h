//
// Created by Will on 01/06/2022.
//

#ifndef MNIST_NEURAL_NET_EIGENUTILS_H
#define MNIST_NEURAL_NET_EIGENUTILS_H

#include <Eigen/Dense>
#include <iostream>
#include <sstream>

using namespace std;

/*
 * Credit:
 * https://stackoverflow.com/questions/68877737/how-to-get-shape-dimensions-of-an-eigen-matrix
 */
template <typename Derived>
string get_shape(const Eigen::EigenBase<Derived>& x) {
    ostringstream oss;
    oss << "(" << x.rows() << ", " << x.cols() << ")";
    return oss.str();
}

double eigenExp(double x);

double eigenLog(double x);

#endif //MNIST_NEURAL_NET_EIGENUTILS_H

//
// Created by Will on 01/06/2022.
//

#ifndef MNIST_NEURAL_NET_EIGENUTILS_H
#define MNIST_NEURAL_NET_EIGENUTILS_H

#include <Eigen/Dense>
#include <iostream>
#include <sstream>

using namespace Eigen;
using namespace std;

/*
 * Credit:
 * https://stackoverflow.com/questions/68877737/how-to-get-shape-dimensions-of-an-eigen-matrix
 */
template <typename Derived>
string get_shape(const EigenBase<Derived>& x) {
    ostringstream oss;
    oss << "(" << x.rows() << ", " << x.cols() << ")";
    return oss.str();
}

/*
 * Exp() helper function for element-wise exponential function
 */
double EigenExp(double x) {
    return exp(x);
}

#endif //MNIST_NEURAL_NET_EIGENUTILS_H

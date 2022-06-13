//
// Created by Will on 13/06/2022.
//

#include <Eigen/Dense>
#include <iostream>
#include <sstream>
#include "EigenUtils.h"

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

double eigenExp(double x) {
    return std::exp(x);
}
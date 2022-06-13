//
// Created by Will on 01/06/2022.
//

#ifndef MNIST_NEURAL_NET_EIGENUTILS_H
#define MNIST_NEURAL_NET_EIGENUTILS_H

#include <Eigen/Dense>
#include <iostream>
#include <sstream>

using namespace std;

template <typename Derived>
string get_shape(const Eigen::EigenBase<Derived>& x);

double eigenExp(double x);

#endif //MNIST_NEURAL_NET_EIGENUTILS_H

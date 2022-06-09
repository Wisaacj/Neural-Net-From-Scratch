//
// Created by Will on 10/06/2022.
//

#include "MLP.h"

MLP::MLP() {
    this->initialiseWeightsAndBiases();
}

MLP::~MLP() = default;

void MLP::initialiseWeightsAndBiases() {
    this->w_1 = MatrixXd::Random(10, 784);
    this->b_1 = MatrixXd::Random(10, 1);
    this->w_2 = MatrixXd::Random(10, 10);
    this->b_2 = MatrixXd::Random(10, 1);
}

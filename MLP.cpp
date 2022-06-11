//
// Created by Will on 10/06/2022.
//

#include "MLP.h"
#include "ActivationFunctions.h"

MLP::MLP() {
    this->initialiseParameters();
}

MLP::~MLP() = default;

void MLP::initialiseParameters() {
    this->w_1 = MatrixXd::Random(10, 784);
    this->b_1 = MatrixXd::Random(10, 1);
    this->w_2 = MatrixXd::Random(10, 10);
    this->b_2 = MatrixXd::Random(10, 1);
}

void MLP::train(const MatrixXd &X, const MatrixXd &y, double learningRate, int epochs) {

}

tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> MLP::forwardPropagation(const MatrixXd &X) {
    // Calculate first hidden-layer (unactivated)
    MatrixXd Z_1 = (this->w_1 * X) + this->b_1;
    // Apply ReLU activation function (activated)
    MatrixXd A_1 = Z_1.unaryExpr(&ReLU);
    // Calculate output-layer (unactivated)
    MatrixXd Z_2 = (this->w_2 * A_1) + this->b_1;
    // Apply softmax activation function (activated)
    MatrixXd A_2 = Softmax(Z_2);

    return make_tuple(Z_1, A_1, Z_2, A_2);
}

tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd>
MLP::backwardPropagation(const MatrixXd &X, const MatrixXd &y, const MatrixXd &A_2, const MatrixXd &A_1) {
    return tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd>();
}

void MLP::updateParameters(const MatrixXd &dW_1, const MatrixXd &db_1, const MatrixXd &dW_2, const MatrixXd &db_2, double learningRate) {

}

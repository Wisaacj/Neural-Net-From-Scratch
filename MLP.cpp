//
// Created by Will on 10/06/2022.
//

#include <Eigen/Dense>
#include <iostream>
#include "MLP.h"
#include "ActivationFunctions.h"
#include "Metrics.h"

using namespace std;
using namespace Eigen;

MLP::MLP() {
    this->initialiseParameters();
}

MLP::~MLP() = default;

void MLP::initialiseParameters() {
    this->w_1 = MatrixXd::Random(10, 784);
    this->b_1 = VectorXd::Random(10, 1);
    this->w_2 = MatrixXd::Random(10, 10);
    this->b_2 = VectorXd::Random(10, 1);
}

void MLP::train(const MatrixXd &X, const MatrixXd &y, double learningRate, int epochs) {
    for (int i=0; i<epochs; i++) {
        MatrixXd Z_1, A_1, Z_2, A_2;
        tie(Z_1, A_1, Z_2, A_2) = forwardPropagation(X);

        MatrixXd dW_1, db_1, dW_2, db_2;
        tie(dW_1, db_1, dW_2, db_2) = backwardPropagation(X, y, A_2, A_1, Z_1);

        updateParameters(dW_1, db_1, dW_2, db_2, learningRate);

        if (i % 2 == 0) {
            cout << "Iteration: " << i << endl;
            cout << "Cross-entropy loss: " << crossEntropyLoss(y, A_2) << endl;
        }
    }
}

tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> MLP::forwardPropagation(const MatrixXd &X) {
    // Calculate first hidden-layer (unactivated)
    MatrixXd Z_1 = (this->w_1 * X).colwise() + this->b_1;
    // Apply ReLU activation function (activated)
    MatrixXd A_1 = Z_1.unaryExpr(&ReLU);
    // Calculate output-layer (unactivated)
    MatrixXd Z_2 = (this->w_2 * A_1).colwise() + this->b_2;
    // Apply softmax activation function (activated)
    MatrixXd A_2 = Softmax(Z_2);

    return make_tuple(Z_1, A_1, Z_2, A_2);
}

tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd>
MLP::backwardPropagation(const MatrixXd &X, const MatrixXd &y, const MatrixXd &A_2, const MatrixXd &A_1, const MatrixXd &Z_1) {
    MatrixXd dZ_2 = A_2 - y;
    MatrixXd dW_2 = (dZ_2 * A_1.transpose()) * 1/X.cols(); // X.cols() is the number of instances
    MatrixXd db_2 = dZ_2.rowwise().sum() * 1/X.cols(); // Average (row-wise) error

    MatrixXd dZ_1 = (this->w_2.transpose() * dZ_2).cwiseProduct(Z_1.unaryExpr(&ReLUDerivative)); // cwiseProduct() is coefficient-wise
    MatrixXd dW_1 = (dZ_1 * X.transpose()) * (1/X.cols());
    MatrixXd db_1 = dZ_1.rowwise().sum() * 1/X.cols();

    return make_tuple(dW_1, db_1, dW_2, db_2);
}

// Gradient descent update function
void MLP::updateParameters(const MatrixXd &dW_1, const MatrixXd &db_1, const MatrixXd &dW_2, const MatrixXd &db_2, double learningRate) {
    this->w_1 = this->w_1 - learningRate * dW_1;
    this->b_1 = this->b_1 - learningRate * db_1;
    this->w_2 = this->w_2 - learningRate * dW_2;
    this->b_2 = this->b_2 - learningRate * db_2;
}

/*
 * See page: https://eigen.tuxfamily.org/dox/group__TutorialReductionsVisitorsBroadcasting.html
 * [ There's almost certainly a more elegant way to do this ]
 */
VectorXd MLP::predict(const MatrixXd &X) {
    MatrixXd _1, _2, _3, probaPredictions;
    tie(_1, _2, _3, probaPredictions) = forwardPropagation(X);

    VectorXd predictions(1, X.cols());
    for (int i = 0; i < X.cols(); i++) {
        Index maxRow;
        probaPredictions.col(i).maxCoeff(&maxRow);
        predictions(0, i) = (double) maxRow;
    }

    return predictions;
}

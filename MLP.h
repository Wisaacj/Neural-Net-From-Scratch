//
// Created by Will on 10/06/2022.
//

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

#ifndef MNIST_NEURAL_NET_MLP_H
#define MNIST_NEURAL_NET_MLP_H

class MLP {
public:
    MLP();
    virtual ~MLP();
    void initialiseParameters();
    void train(const MatrixXd& X, const MatrixXd& y, double learningRate, int epochs);
    MatrixXd predict(const MatrixXd& X);
private:
    MatrixXd w_1, w_2;
    VectorXd b_1, b_2;
    tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> forwardPropagation(const MatrixXd &X);
    tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> backwardPropagation(const MatrixXd &X, const MatrixXd &y, const MatrixXd &A_2, const MatrixXd &A_1, const MatrixXd &Z_1);
    void updateParameters(const MatrixXd &dW_1, const MatrixXd &db_1, const MatrixXd &dW_2, const MatrixXd &db_2, double learningRate);
};


#endif //MNIST_NEURAL_NET_MLP_H

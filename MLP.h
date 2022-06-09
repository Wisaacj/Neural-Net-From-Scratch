//
// Created by Will on 10/06/2022.
//

#ifndef MNIST_NEURAL_NET_MLP_H
#define MNIST_NEURAL_NET_MLP_H

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class MLP {
public:
    MLP();
    virtual ~MLP();
    void initialiseWeightsAndBiases();
    void train();
    MatrixXd predict(const MatrixXd& X);
private:
    MatrixXd w_1, b_1, w_2, b_2;
    MatrixXd relu(const MatrixXd& Z);
    MatrixXd softmax(const MatrixXd& Z);
};


#endif //MNIST_NEURAL_NET_MLP_H

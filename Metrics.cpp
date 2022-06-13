//
// Created by Will on 13/06/2022.
//

#include <Eigen/Dense>
#include "Metrics.h"
#include "EigenUtils.h"

using namespace Eigen;
using namespace std;

double crossEntropyLoss(const MatrixXd &groundTruth, const MatrixXd &predictedProba) {
    double totalLoss = -1 * groundTruth.cwiseProduct(predictedProba.unaryExpr(&eigenLog)).sum();
    return totalLoss * 1/(groundTruth.cols()); // Averaging total loss
}

double accuracy(const MatrixXd &groundTruth, const MatrixXd &predictions) {
    return 0;
}
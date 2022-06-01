#include <iostream>
#include <Eigen/Dense>
#include "CSVReader.h"

using namespace std;
using namespace Eigen;

tuple<VectorXd, MatrixXd> splitLabelsFromFeatures(MatrixXd A) {
    VectorXd labels = A(all, 0);
    MatrixXd features = A(all, seqN(1, last));

    return make_tuple(labels, features);
}

int main() {
    // Loading csv datasets into Eigen matrices
    MatrixXd test = load_csv<MatrixXd>(R"(C:\Users\Will\OneDrive\Projects\C++\mnist-neural-net\data\mnist_test.csv)");
    MatrixXd train = load_csv<MatrixXd>(R"(C:\Users\Will\OneDrive\Projects\C++\mnist-neural-net\data\mnist_train.csv)");

    VectorXd test_labels, train_labels;
    MatrixXd test_features, train_features;

    // tie() unpacks the tuple values into separate variables
    tie(test_labels, test_features) = splitLabelsFromFeatures(test);
    tie(train_labels, train_features) = splitLabelsFromFeatures(train);
    
    cout << "10th training instance: " << train_labels(10) << "\n" << train_features(10, all).reshaped(28, 28) << endl;

    return 0;
}

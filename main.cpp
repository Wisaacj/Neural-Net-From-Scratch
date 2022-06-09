#include <iostream>
#include <Eigen/Dense>
#include "CSVReader.h"
#include "EigenUtils.h"
#include "PreprocessingUtils.h"

using namespace std;
using namespace Eigen;

int main() {
    // Loading csv datasets into Eigen matrices
    MatrixXd test = load_csv<MatrixXd>(R"(C:\Users\Will\OneDrive\Projects\C++\mnist-neural-net\data\mnist_test.csv)");
    MatrixXd train = load_csv<MatrixXd>(R"(C:\Users\Will\OneDrive\Projects\C++\mnist-neural-net\data\mnist_train.csv)");

    VectorXd test_labels, train_labels;
    MatrixXd test_features, train_features;

    // tie() unpacks the tuple values into separate variables
    tie(test_labels, test_features) = splitLabelsFromFeatures(test);
    tie(train_labels, train_features) = splitLabelsFromFeatures(train);

    // One-hot encode the labels (y_test, y_train)
    MatrixXd test_labels_encoded = oneHotEncode(test_labels);
    MatrixXd train_labels_encoded = oneHotEncode(train_labels);

    cout << "Training labels (10 x m) 1-Hot encoded: \n" << train_labels_encoded(all, seqN(0, 25)) << "\n" << endl;
    cout << "10th training instance: " << train_labels(10) << "\n" << train_features(10, all).reshaped(28, 28) << endl;
    cout << "Shape of training set: " << get_shape(train_features) << endl;
    cout << "Shape of test set: " << get_shape(test_features) << endl;

    return 0;
}

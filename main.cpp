#include <iostream>
#include <Eigen/Dense>
#include "CSVReader.h"
#include "PreProccessingUtils.h"
#include "MLP.h"
#include "EigenUtils.h"

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

    cout << "Training labels 1-hot encoded: \n" << train_labels_encoded(all, seqN(0, 25)) << "\n" << endl;
    cout << "10th training instance: " << train_labels(10) << "\n" << train_features(all, 10).reshaped(28, 28) << endl;

    cout << "\nShape of training feature set: " << get_shape(train_features) << endl;
    cout << "Shape of training label set: " << get_shape(train_labels_encoded) << endl;
    cout << "Shape of test feature set: " << get_shape(test_features) << endl;
    cout << "Shape of test label set: " << get_shape(test_labels_encoded) << endl;

    // Initialising neural network (multi-layered perceptron)
    MLP* neural_network = new MLP();
    neural_network->train(train_features, train_labels_encoded, 0.1, 200);

    return 0;
}

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include "CSVReader.h"
#include "EigenUtils.h"
#include "PreProccessingUtils.h"
#include "MLP.h"

using namespace std;
using namespace Eigen;

int main() {
//    // Loading csv datasets into Eigen matrices
//    MatrixXd test = load_csv<MatrixXd>(R"(C:\Users\Will\OneDrive\Projects\C++\mnist-neural-net\data\mnist_test.csv)");
//    MatrixXd train = load_csv<MatrixXd>(R"(C:\Users\Will\OneDrive\Projects\C++\mnist-neural-net\data\mnist_train.csv)");
//
//    VectorXd test_labels, train_labels;
//    MatrixXd test_features, train_features;
//
//    // tie() unpacks the tuple values into separate variables
//    tie(test_labels, test_features) = splitLabelsFromFeatures(test);
//    tie(train_labels, train_features) = splitLabelsFromFeatures(train);
//
//    // One-hot encode the labels (y_test, y_train)
//    MatrixXd test_labels_encoded = oneHotEncode(test_labels);
//    MatrixXd train_labels_encoded = oneHotEncode(train_labels);
//
//    cout << "Training labels 1-Hot encoded: \n" << train_labels_encoded(all, seqN(0, 25)) << "\n" << endl;
//    cout << "10th training instance: " << train_labels(10) << "\n" << train_features(all, 10).reshaped(28, 28) << endl;
//    cout << "\nShape of training set: " << get_shape(train_features) << endl;
//    cout << "Shape of test set: " << get_shape(test_features) << endl;
//
//    // Initialising neural network (multi-layered perceptron)
//    MLP* neural_network = new MLP();

    MatrixXd m(2, 2);
    m << -1.4, 22, 3, 400;

    MatrixXd a(2, 1);
    a << 4, 8;
    cout << a << endl << a.array() / a.array() << endl;

    cout << a.unaryExpr(&EigenExp).sum() << endl;

    const auto mColwise = m.colwise();
    for_each(mColwise.begin(), mColwise.end(), [](const auto &column){
        cout << "Column: " << column << "\n" << "Softmax column: " << "\n" << column.unaryExpr(&EigenExp) / column.unaryExpr(&EigenExp).sum() << endl;
       return column.unaryExpr(&EigenExp) / column.unaryExpr(&EigenExp).sum();
    });

    cout << endl << m;

    return 0;
}

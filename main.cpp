#include <iostream>
#include <Eigen/Dense>
#include "CSVReader.h"

using namespace std;
using namespace Eigen;

int main() {
    // Testing load_csv method
    MatrixXd test = load_csv<MatrixXd>(R"(C:\Users\Will\OneDrive\Projects\C++\mnist-neural-net\data\mnist_test.csv)");
    // Printing out the first 10 rows of the test dataset
    cout << test(Eigen::seqN(0,1,1), Eigen::all) << endl;
    
    return 0;
}

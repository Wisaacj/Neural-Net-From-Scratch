//
// Created by Will on 31/05/2022.
//

#ifndef MNIST_NEURAL_NET_CSVREADER_H
#define MNIST_NEURAL_NET_CSVREADER_H
#include <iostream>

class CSVReader {
public:
    CSVReader(); // Default constructor
    virtual ~CSVReader(); // De-constructor

    template<typename M>
    M load_csv(const std::string & path);
};


#endif //MNIST_NEURAL_NET_CSVREADER_H

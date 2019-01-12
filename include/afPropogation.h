#ifndef AFPROPOGATION_H
#define AFPROPOGATION_H

#include <Eigen/Sparse>
using SpMatU = Eigen::SparseMatrix<uint8_t>;
using SpMatS = Eigen::SparseMatrix<long>;
using Vector = Eigen::Matrix<long, -1, 1>;

class AfPropogation
{
private:
    SpMatS R;
    SpMatS A;
    void update_R(const SpMatU& S);
    void update_A(const SpMatU& S);
public:
    Vector run(const SpMatU& S, size_t max_iteration);
};

#endif

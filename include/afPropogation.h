#ifndef AFPROPOGATION_H
#define AFPROPOGATION_H

#include <Eigen/Sparse>
using SpMatU = Eigen::SparseMatrix<int16_t>;
using SpMatL = Eigen::SparseMatrix<long>;
using VectorL = Eigen::Matrix<long, -1, 1>;
using VectorS = Eigen::Matrix<size_t, -1, 1>;
class AfPropogation
{
private:
    SpMatL R;
    SpMatL A;
    VectorS argMax(const SpMatL& C);
    void update_R(const SpMatU& S);
    void update_A(const SpMatU& S);
public:
    VectorS run(const SpMatU& S, size_t max_iteration);
};

#endif

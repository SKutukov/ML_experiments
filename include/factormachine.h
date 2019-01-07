#ifndef FACTORMACHINE_H
#define FACTORMACHINE_H

#include <Eigen/Sparse>
#include <Eigen/Dense>

using SpMat = Eigen::SparseMatrix<uint16_t>;
using Vector = Eigen::VectorXf;
using Matrix = Eigen::MatrixXf;
class FactorMachine
{
private:
    int n = 0;
    int rank = 0;
    float base;
    Vector weigths;
    Matrix V;
    Matrix common_sum;

public:
    FactorMachine(int n, int rank);
    auto RMSE(const Vector& y, const Vector& y_pred);
    void batch_step(const SpMat& X_mini, const Vector& Y_mini, float step);
    float fit(const SpMat& X, const Vector& Y, float step);

    Vector predict(const SpMat &X);
};

#endif // FACTORMACHINE_H

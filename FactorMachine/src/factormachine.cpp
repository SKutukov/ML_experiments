#include "factormachine.h"
#include <iostream>
#include <random>
#include <algorithm>

FactorMachine::FactorMachine(int n, int rank)
{
    base = 0;
    weigths = Vector::Random(n,1);
    V = Matrix::Random(n,rank);
}

auto FactorMachine::RMSE(const Vector& y, const Vector& y_pred)
{
    // mean of (y - y_pred)^2
    return sqrtf((y - y_pred).array().square().mean());
}



Vector FactorMachine::predict(const SpMat& X)
{
    common_sum = (X.cast<float>() * V).array().square().matrix();

    Matrix sqSum = X.cast<float>() * (V.array().square().matrix());
    auto v_part = (common_sum - sqSum) * Vector::Ones(sqSum.cols(), 1);
    auto res = base + (X.cast<float>() * weigths).array() - 0.5 * v_part.array();
    return res;
}

void FactorMachine::batch_step(const SpMat& X, const Vector& Y, float step)
{
    auto y_pred = predict(X);
    auto dy = (y_pred - Y);
    base -= dy.mean() * step;
    auto t = (dy.transpose() * X.cast<float>())/ Y.rows();
    weigths -= 2* t.transpose() * step;

    for(int f = 0; f< rank; f++)
    {
        Vector common_part = X.cast<float>().transpose() * common_sum.col(f);
        Eigen::SparseMatrix<float> Vf_diagonalize = X.cast<float>() * (V.col(f).asDiagonal());
        Vector second_part = Vf_diagonalize.transpose() * Vector::Ones(X.rows(), 1);
        V.col(f) = (common_part - second_part) * step / X.rows();
    }

}

float FactorMachine::fit(const SpMat& X, const Vector& Y, float step)
{
     size_t k = 0;
     float cost1 = 0;
     float cost = 0;

     batch_step(X, Y, step);
     auto y_pred = predict(X);
     return (Y - y_pred).array().square().sum();
}

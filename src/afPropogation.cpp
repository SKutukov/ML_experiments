#include "afPropogation.h"
#include <iostream>

Vector AfPropogation::run(const SpMatU& S, size_t max_iteration)
{
    R = SpMatS(S.rows(), S.cols());
    A = SpMatS(S.rows(), S.cols());
    for (size_t i = 1; i <= max_iteration; i++) {
        std::cout<< i <<std::endl;
//        update_R(S);
        update_A(S);
    }

    //get maximum in each column
    return (A + R).toDense().colwise().maxCoeff();
}

void AfPropogation::update_R(const SpMatU &S)
{
    for (int row = 0; row < S.outerSize(); ++row){
      for (int col = row; col < S.outerSize(); ++col)
      {
        std::cout<< row << ' ' << col << std::endl;
        // get column
        auto S_column = S.col(col);
        auto A_column = A.col(col);
        auto dense_column = (A.col(col) + S.col(col).cast<long>() );
        long max1 = 0;
        if ( row != 0 )
        {
            // get maximum of first row elem
            max1 = dense_column.head( row ).toDense().maxCoeff();
        }
        long max2 = 0;
        if ( row !=  (S.outerSize() - 1))
        {
            // get maximum of last S.outerSize() - row - 1  elem
            max2 = dense_column.tail( S.outerSize() - row - 1 ).toDense().maxCoeff();
        }
        R.coeffRef(row, col) = S.coeff(row, col) - std::max(max1, max2);

      }
    }
}

void AfPropogation::update_A(const SpMatU &S)
{
    for (int row = 0; row < S.outerSize(); ++row){
          for (int col = row; col < S.outerSize(); ++col)
          {
              std::cout<< row << ' ' << col << std::endl;
              auto min = std::min(row , col);
              auto max = std::max(row , col);
              auto R_column = R.col(col).toDense().cwiseMax(0);
              std::cout<< row << ' ' << col << std::endl;
              // get maximum of first min - 1 elem
              long sum1 = 0;
              if(min != 0) {
                  sum1 = R_column.head( min ).sum();
              }
              std::cout<< row << ' ' << col << std::endl;
              // get maximum of last S.outerSize() - max - 1  elem
              long sum2 = 0;
              if(max != (S.outerSize() - 1)){
                  std::cout<< S.outerSize() - max - 1 << std::endl;
                  auto t = R_column.tail( S.outerSize() - max - 1 );
                  std::cout<< t.rows() << std::endl;
                  sum2 = t.sum();
              }
              std::cout<< row << ' ' << col << std::endl;
              auto positive_sum =  sum1 + sum2;
              std::cout<< row << ' ' << col << std::endl;
              if(row != col){
                long sum3 = 0;
                if(max - min - 1 > 0) {
                    sum3 = R_column.block(min + 1, 0, max - min - 1, 1).sum();
                }
                A.coeffRef(row, col) = std::min(long(0),  R.coeff(col, col) + positive_sum + sum3);
              } else {
                A.coeffRef(row, col) = std::min(long(0),  positive_sum);
              }

          }
    }
}

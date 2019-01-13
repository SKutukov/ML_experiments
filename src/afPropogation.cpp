#include "afPropogation.h"
#include <iostream>


VectorS AfPropogation::run(const SpMatU& S, size_t max_iteration)
{
    std::cout<< S << std::endl;
    R = SpMatL(S.rows(), S.cols());
    A = SpMatL(S.rows(), S.cols());
    for (size_t i = 0; i < max_iteration; i++) {
        update_R(S);
        update_A(S);
        std::cout<<R<<std::endl;
        std::cout<<A<<std::endl;
    }

    //get maximum in each column
    return argMax(A + R);
}

VectorS AfPropogation::argMax(const SpMatL &C)
{
    std::cout<< C << std::endl;
    VectorS maxIndexs(C.cols(),1);
    SpMatL::Index tempIndex;
    std::cout<<"argmax" << std::endl;
    for(long i=0; i<C.cols(); ++i){
        long maxIndex = 0;
        long maxValue = C.coeff(0, i);
        for(long j=1; j<C.rows(); ++j){
            if(maxValue < C.coeff(j, i)){
                maxValue = C.coeff(j, i);
                maxIndex = j;
            }
        }
        maxIndexs.coeffRef(i) = static_cast<size_t>(maxIndex);
    }
//    std::cout<< maxIndex << std::endl;
    return  maxIndexs;
}

void AfPropogation::update_R(const SpMatU &S)
{
    for (int row = 0; row < S.outerSize(); ++row){
      for (int col = 0; col < S.outerSize(); ++col)
      {
        // get row
        auto S_column = S.row(row);
        auto A_column = A.row(row);
        auto dense_column = (A_column + S_column.cast<long>() );
        long max1 = std::numeric_limits<long>::min();
        if ( row != 0 )
        {
            // get maximum of first row elem
            max1 = dense_column.head( row ).toDense().maxCoeff();
        }
        long max2 = std::numeric_limits<long>::min();
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
          for (int col = 0; col < S.outerSize(); ++col)
          {
              auto min = std::min(row , col);
              auto max = std::max(row , col);
              auto R_column = R.col(col).toDense().cwiseMax(0);
              // get maximum of first min - 1 elem
//              std::cout<< row << ' '<< col << std::endl;
              // get maximum of last S.outerSize() - max - 1  elem
              long sum2 = 0;
              if(max != (S.outerSize() - 1)){
                  // todo
                  // undestand whats wrong happend on release if next two rows are missing
                  std::stringstream ss;
                  ss << R_column.block(row, 0, S.outerSize() - max - 1, 1);

                  sum2 = R_column.block(row, 0, S.outerSize() - max - 1, 1).eval().sum();
              }

              long sum1 = 0;
              if(min != 0) {
                  sum1 = R_column.head( min ).eval().sum();
              }

//              std::cout<< row << ' '<< col << std::endl;
              auto positive_sum =  sum1 + sum2;
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

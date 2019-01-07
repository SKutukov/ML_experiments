
#include <iostream>
#include <Eigen/Sparse>
#include "factormachine.h"

#include <iostream>
#include <fstream>
#include <string>


#include <algorithm>
#include <random>

typedef Eigen::Triplet<uint16_t> T;
using namespace std;

struct triplet
{
    uint16_t MovieID;
    uint16_t CustomersID;
    uint16_t r;
};

std::vector<triplet> read_data(std::string filename ){
    std::vector<triplet> triplets;
    ifstream file(filename);
    std::cout<< "Start reading ... \n";
    size_t k = 0;
    std::string line;
    if (file.is_open())
    {
      while ( std::getline(file, line) )
      {
        std::istringstream iss(line);
        uint16_t MovieID;
        uint16_t CustomerID;
        uint16_t r;
        iss >> MovieID >> CustomerID >> r;
//        std::cout<< MovieID << ' ' << CustomerID << ' ' << uint16_t(r) << std::endl;
        triplets.push_back({MovieID, CustomerID, r});
        k++;

      }

      file.close();
      std::cout<<k<<std::endl;
      std::cout<< "Stop reading ... \n";
      return triplets;
    }
}

struct dataset{
    std::vector<triplet> triplets_train;
    std::vector<triplet> triplets_test;
};

dataset cross_validation(const std::vector<triplet>& triplets, size_t part)
{
    assert(part<5);
    auto first_test = triplets.cbegin() + size_t((double(part)/5) * triplets.size());
    auto last_test = triplets.cbegin() + size_t(((double(part + 1)/5) * triplets.size()));
    std::vector<triplet> test(first_test, last_test);

    auto first_train1 = triplets.cbegin();
    auto last_train1 = triplets.cbegin() + size_t((double(part)/5) * triplets.size());
    std::vector<triplet> train(first_test, last_test);

    auto first_train2 = triplets.cbegin() + size_t((double(part + 1)/5) * triplets.size());
    auto last_train2 = triplets.cbegin() + triplets.size();

    train.insert(
          train.end(),
          std::make_move_iterator(first_train2),
          std::make_move_iterator(last_train2)
        );

    return {train, test};

}
int main(int, char *[])
{
    const size_t user_count = 17770;
    const size_t movie_count = 480189;
    const size_t sum_count = user_count + movie_count;
//    const size_t user_count = 10;
//    const size_t movie_count =1000;
//    const size_t sum_count = 1000 + 10;
    const uint8_t k = 5;
    const size_t batch_size = 10000;
    std::string filename = "/home/skutukov/work/FactorMachine/full_norm_set.txt";
    std::vector<triplet> triplets = read_data(filename);

    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(triplets), std::end(triplets), rng);


    float step = 0.05f;
    size_t max_iter = 5;
    float e = 0.1f;

    for(size_t p = 0; p<5; p++)
    {
        FactorMachine f(sum_count, k);

        size_t iter = 0;
        float cost1 =0.;
        dataset data = cross_validation(triplets, p);
        // shuffle data
        auto rng = std::default_random_engine {};
        std::shuffle(std::begin(data.triplets_train), std::end(data.triplets_train), rng);
        std::shuffle(std::begin(data.triplets_test), std::end(data.triplets_test), rng);
        // train
        while( true )
        {
            float cost = 0;
            for(size_t i = 0; i< data.triplets_train.size()/batch_size; i++){
                // create sparce matrix
                SpMat spMat(batch_size, sum_count);
                Vector Y(batch_size);
                size_t right_border = std::min((i + 1) * batch_size, data.triplets_train.size());
                std::vector<T> vec;
                std::vector<T> Y_vec;
                for(size_t j = 0; j < right_border - i * batch_size; j++){
                    triplet a = data.triplets_train[j];
                    vec.push_back(T(j, a.MovieID, 1));
                    vec.push_back(T(j, movie_count + a.CustomersID, 1));
                    Y[j] = float(a.r);
                }

                spMat.setFromTriplets(vec.begin(), vec.end());
                spMat.makeCompressed();

                // fit model
                cost += f.fit(spMat, Y, step);
//                std::cout<< "progress: "<< float(i)/(data.triplets_train.size()/batch_size) << " cost: " << cost/(i*batch_size) << "\r";

            }
//            std::cout<<std::endl;
            cost /= data.triplets_train.size();
            cost = sqrtf(cost);

            iter++;
            std::cout<< "k: "<< iter << " cost: " << cost << std::endl;
            if(std::fabs(cost - cost1)<e || iter>=max_iter){
                break;
            }
            cost1 = cost;
         }
        // test
        float cost = 0;
        for(size_t i = 0; i< data.triplets_test.size()/batch_size; i++){
            // create sparce matrix
            SpMat spMat(batch_size, sum_count);
            Vector Y(batch_size);
            size_t right_border = std::min((i + 1) * batch_size, data.triplets_test.size());
            std::vector<T> vec;
            std::vector<T> Y_vec;
            for(size_t j = 0; j < right_border - i * batch_size; j++){
                triplet a = data.triplets_test[j];
                vec.push_back(T(j, a.MovieID, 1));
                vec.push_back(T(j, movie_count + a.CustomersID, 1));
                Y[j] = float(a.r);
            }

            spMat.setFromTriplets(vec.begin(), vec.end());
            spMat.makeCompressed();

            // fit model
            auto y_pred = f.predict(spMat);
            cost += (Y - y_pred).array().square().sum();
        }
        cost /= data.triplets_test.size();
        cost = sqrtf(cost);
        std::cout<< "test cost: " << cost << std::endl;


    }
}



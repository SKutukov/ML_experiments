#include <iostream>
#include <vector>
#include <Eigen/Sparse>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include "afPropogation.h"

using namespace std;
typedef Eigen::Triplet<uint8_t> T;

struct edge
{
    size_t beg;
    size_t end;
};

std::vector<edge> read_data(std::string filename ){
    std::vector<edge> edges;
    ifstream file(filename);

    std::cout<< "Start reading ... \n";
    size_t k = 0;
    std::string line;
    if (file.is_open())
    {
      while ( std::getline(file, line) )
      {
        std::istringstream iss(line);
        size_t beg;
        size_t end;
        iss >> beg >> end;
        edges.push_back({beg, end});
        k++;

      }

      file.close();
      std::cout<<k<<std::endl;
      std::cout<< "Stop reading ... \n";
      return edges;
    }
}

struct dataset{
    std::vector<edge> edges_train;
    std::vector<edge> edges_test;
};

dataset cross_validation(const std::vector<edge>& edges, size_t part)
{
    assert(part<5);
    auto first_test = edges.cbegin() + size_t((double(part)/5) * edges.size());
    auto last_test = edges.cbegin() + size_t(((double(part + 1)/5) * edges.size()));
    std::vector<edge> test(first_test, last_test);

    auto first_train1 = edges.cbegin();
    auto last_train1 = edges.cbegin() + size_t((double(part)/5) * edges.size());
    std::vector<edge> train(first_test, last_test);

    auto first_train2 = edges.cbegin() + size_t((double(part + 1)/5) * edges.size());
    auto last_train2 = edges.cbegin() + edges.size();

    train.insert(
          train.end(),
          std::make_move_iterator(first_train2),
          std::make_move_iterator(last_train2)
        );

    return {train, test};

}


int main(int, char *[])
{
    const size_t nodes_count = 196591;
    const size_t batch_size = 10000;
    std::string filename = "/home/skutukov/work/AffinitiPropagation/Gowalla_edges.txt";
    std::vector<edge> edges = read_data(filename);

    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(edges), std::end(edges), rng);


    size_t max_iter = 500;

    for(size_t p = 0; p<5; p++)
    {

        AfPropogation AF;
        dataset data = cross_validation(edges, p);
        // shuffle data
        auto rng = std::default_random_engine {};
        std::shuffle(std::begin(data.edges_train), std::end(data.edges_train), rng);
        std::shuffle(std::begin(data.edges_test), std::end(data.edges_test), rng);
        // train

        SpMatU spMat(nodes_count, nodes_count) ;
        std::vector<T> vec;
        for(size_t j = 0; j < data.edges_train.size(); j++){
            edge a = data.edges_train[j];
            vec.push_back(T(a.beg, a.end, 1));
        }
        spMat.setFromTriplets(vec.begin(), vec.end());
        spMat.makeCompressed();

        auto C = AF.run(spMat, max_iter);

        std::cout<< C << std::endl;

    }
}



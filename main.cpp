#include <iostream>
#include <vector>
#include <Eigen/Sparse>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include "afPropogation.h"
#include <fstream>

using namespace std;
typedef Eigen::Triplet<int16_t> T;

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


int main(int, char *[])
{
//    const size_t nodes_count = 196591;
    const size_t nodes_count = 14;
//    std::string filename = "/home/skutukov/work/AffinitiPropagation/Gowalla_edges.txt";
    std::string filename = "/home/skutukov/work/AffinitiPropagation/test.txt";
    std::vector<edge> edges = read_data(filename);

    size_t max_iter = 10;


    AfPropogation AF;
    // shuffle data
    SpMatU spMat(nodes_count, nodes_count) ;
    std::vector<T> vec;
    for(size_t j = 0; j < edges.size(); j++){
        edge a = edges[j];
        vec.push_back(T(a.beg, a.end, 1));
        vec.push_back(T(a.end, a.beg, 1));
    }

    spMat.setFromTriplets(vec.begin(), vec.end());
    spMat.makeCompressed();

    auto C = AF.run(spMat, max_iter);

    ofstream log;
    std::cout<< C << std::endl;
    log.open ("log.txt");
    log << C;
    log.close();

}



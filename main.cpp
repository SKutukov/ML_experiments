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
typedef Eigen::Triplet<double> T;

std::vector<edge> read_data(std::string filename, graph& graph_in, graph& graph_out){
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
        std::shared_ptr<edge> sp_e = std::make_shared<edge>(edge(beg, end, 1., 0, 0.));
        graph_in.lists[beg].push_back(sp_e);
        graph_out.lists[end].push_back(sp_e);
        k++;

      }

      file.close();

      // init diag edges
      for (size_t i=0; i < graph_in.lists.size(); ++i) {
          std::shared_ptr<edge> sp_e = std::make_shared<edge>(edge(i, i, -1., 0, 0.));
          graph_in.lists[i].push_back(sp_e);
          graph_out.lists[i].push_back(sp_e);
      }

      std::cout<<k<<std::endl;
      std::cout<< "Stop reading ... \n";
      return edges;
    }
}


int main(int, char *[])
{
//    const size_t nodes_count = 10;
//    std::string filename = "/home/skutukov/work/AffinitiPropagation/test.txt";

    const size_t nodes_count = 196591;
    std::string filename = "/home/skutukov/work/AffinitiPropagation/loc-gowalla_edges.txt";

    graph graph_in(nodes_count), graph_out(nodes_count);
    std::vector<edge> edges = read_data(filename, graph_in, graph_out);
    size_t max_iter = 20;

    AfPropogation AF;
    auto C = AF.run(max_iter, graph_in, graph_out);

    ofstream log;
    log.open ("res.txt");
    size_t i = 0;
    for(auto c: C){
        log<< i << "\t" << c << std::endl;
    }
    log<< std::endl;
    log.close();
}



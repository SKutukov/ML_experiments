#include "afPropogation.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <set>

void AfPropogation::update_R(double lambda, graph& graph){
    for (size_t i = 0; i < graph.lists.size(); ++i) {
        auto& edges = graph.lists[i];

        for (size_t j = 0; j < edges.size(); ++j) {
            auto& edge = edges[j];

            double max_value = -std::numeric_limits<double>::infinity();
            int max_index = -1;
            for (size_t k = 0; k < edges.size(); ++k) {
                auto& temp_edge = edges[k];
                if(max_value< (temp_edge->a + temp_edge->s) && j != k){
                    max_value = temp_edge->a + temp_edge->s;
                    max_index = k;
                }
            }
            // set max value as 0 if row have only one elem
            if( max_index == -1){
                max_value = 0;
            }
            double new_value = new_value = edge->s - max_value;
            edge->r =  edge->r * lambda - new_value* (1. - lambda);
        }
    }
}
void AfPropogation::update_A(double lambda, graph& graph){
    for (size_t i = 0; i < graph.lists.size(); ++i) {
        auto& edges = graph.lists[i];
        double sum = 0;

        for (size_t j = 0; j < edges.size(); ++j){
            sum += std::max<double>(0., edges[j]->r);
        }

        double self_responsibility = edges.back()->r;

        for (size_t j = 0; j < edges.size() - 1; ++j){
            double new_value = self_responsibility + sum - std::max<double>(0., edges[j]->r);
            edges[j]->a = edges[j]->a * lambda + new_value * (1. - lambda);
        }

        double new_value = sum;
        edges.back()->a= edges.back()->a * lambda + new_value * (1. - lambda);
    }
}

std::vector<size_t> AfPropogation::argmax(const graph& graph)
{
    std::vector<size_t> C(graph.lists.size());
    for(size_t i = 0; i< graph.lists.size(); ++i){
        auto& p_edgesRow = graph.lists[i];
        double max = -std::numeric_limits<double>::infinity();
        size_t max_index = -1;
        double temp_value;

        for (size_t j = 0; j < p_edgesRow.size(); ++j) {
            auto& edge = p_edgesRow[j];
            temp_value = edge->s + edge->a;

            if (temp_value > max) {
                max = temp_value;
                max_index = edge->end;
            }
        }
        C[i] = max_index;
    }
    return C;
}
std::vector<size_t> AfPropogation::run(size_t max_iteration, graph& graph_in, graph& graph_out){
        double lambda = 0.7;
        for (size_t i = 0; i < max_iteration; i++) {
            std::cout<< i << "\n";
            update_R(lambda, graph_in);
            update_A(lambda, graph_out);
            auto temp = argmax(graph_in);
            std::cout << "Number of unique elements is "
                          << std::set<size_t>( temp.begin(), temp.end() ).size()
                          << "\r";
        }
        return argmax(graph_in);
}

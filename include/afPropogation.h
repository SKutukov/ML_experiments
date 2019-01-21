#ifndef AFPROPOGATION_H
#define AFPROPOGATION_H

#include <memory>
#include <vector>

struct edge
{
    size_t beg;
    size_t end;
    double s, a, r;
    edge( size_t beg, size_t end, double s, double a, double r):
    beg(beg),end(end),s(s),a(a),r(r)
    {}
};

struct graph
{
    std::vector<std::vector<std::shared_ptr<edge>>> lists;
    graph(size_t nodes_count){
        //init list with empty
        lists.resize(nodes_count);
        lists.assign(nodes_count, std::vector<std::shared_ptr<edge>>());
    }
};

class AfPropogation
{
private:
    void update_R(double lambda, graph& graph_out);
    void update_A(double lambda, graph& graph_in);
    std::vector<size_t> argmax(const graph& graph_out);
public:
    std::vector<size_t> run(size_t max_iteration, graph& graph_in, graph& graph_out);
};

#endif

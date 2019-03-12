
#include <iostream>
#include <Eigen/Sparse>
#include "factormachine.h"
#include "boost_serialization_eigen.h"
#include <iostream>
#include <fstream>
#include <string>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <algorithm>
#include <random>

typedef Eigen::Triplet<uint8_t> T;
using namespace std;

struct triplet
{
    uint16_t user_id;
    uint16_t metadata_ownerId;
    uint16_t object_id;
    std::vector<int> themes;
    float r;
};

#include <boost/regex.hpp>
 
// used to split the file in lines
const boost::regex linesregx("\\r\\n|\\n\\r|\\n|\\r");
 
// used to split each line to tokens, assuming ',' as column separator
const boost::regex fieldsregx("\t(?=(?:[^\"]*\"[^\"]*\")*(?![^\"]*\"))");
 
typedef std::vector<std::string> Row;
std::vector<int> parse_onehot(std::string str){
    std::vector<int> themes;

    std::string s = str.substr(1,str.size() - 2);
    std::string delimiter = " ";
    size_t pos = 0;
    int count = 0;
    float value;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        value = std::stof(token);
        
        s.erase(0, pos + delimiter.length());
        if (value){
            themes.push_back(count);
        }
        count++;
    }
    value = std::stof(s);
    if (value == 1.){
            themes.push_back(count);
    }
    return themes;
} 
std::vector<Row> parse(const char* data, unsigned int length)
{
    std::vector<Row> result;
 
    // iterator splits data to lines
    boost::cregex_token_iterator li(data, data + length, linesregx, -1);
    boost::cregex_token_iterator end;
 
    while (li != end) {
        std::string line = li->str();
        ++li;
 
        // Split line to tokens
        boost::sregex_token_iterator ti(line.begin(), line.end(), fieldsregx, -1);
        boost::sregex_token_iterator end2;
 
        std::vector<std::string> row;
        while (ti != end2) {
            std::string token = ti->str();
            ++ti;
            row.push_back(token);
        }
        if (line.back() == ',') {
            // last character was a separator
            row.push_back("");
        }
        result.push_back(row);
    }
    return result;
}

std::vector<triplet> read_data(std::string filename, bool isTrain=true){
    std::vector<triplet> triplets;
    std::cout<< "Start reading ... \n";
    std::ifstream infile;
    infile.open(filename);
    std::string line;
    std::getline(infile, line);
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::vector<Row> result  = parse(line.c_str(), line.size());
        for(size_t r=0; r < result.size(); r++) {
            triplet tr;
            Row& row = result[r];
            tr.user_id = std::stoi(row[1]); 
            tr.object_id = std::stoi(row[2]);
            tr.metadata_ownerId = std::stoi(row[3]);
            
            tr.themes = parse_onehot(row[4]);
            if(isTrain){
            //    std::cout<< row[5] << std::endl;
                tr.r = std::stof(row[5]);
            }
            triplets.push_back(tr);
        }
    }

    // parse file
 
      std::cout<< "Stop reading ... \n";
      return triplets;
    
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
    const size_t user_count = 15710942;
    const size_t autor_count = 85735;
    const size_t themes_count = 19;
    const size_t sum_count = user_count + autor_count + themes_count;


    const uint8_t k = 5;
    const size_t batch_size = 10000;
    dataset data;
    std::string filename = "/home/skutukov/Documents/SNA/data/train_data.csv";
    data.triplets_train = read_data(filename);
    filename = "/home/skutukov/Documents/SNA/data/test_onehots.csv";
    data.triplets_test = read_data(filename, false);
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(data.triplets_train), std::end(data.triplets_train), rng);


    float step = 0.05f;
    size_t max_iter = 10;
    float e = 0.01f;

    // for(size_t p = 0; p<5; p++){
        FactorMachine f(sum_count, k);

        size_t iter = 0;
        float cost1 =0.;
        // dataset data = cross_validation(triplets, p);
        // shuffle data
        // train
        std::cout<< "begin train" << std::endl;
        while( true )
        {
            std::shuffle(std::begin(data.triplets_train), std::end(data.triplets_train), rng);
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
                    vec.push_back(T(j, a.user_id, 1));
                    vec.push_back(T(j, user_count + a.metadata_ownerId, 1));
                    for( size_t i=0; i < a.themes.size(); i++){
                        vec.push_back(T(j, user_count + autor_count + a.themes[i], 1));
                    }
                    Y[j] = float(a.r);
                }

                spMat.setFromTriplets(vec.begin(), vec.end());
                spMat.makeCompressed();

                // fit model
                cost += f.fit(spMat, Y, step);
                std::cout<< "progress: "<< float(i)/(data.triplets_train.size()/batch_size) << " cost: " << cost/(i*batch_size) << "\n";

            }
            std::cout<< (data.triplets_train.size()) << std::endl;
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
        
        // std::cout<< "write weight" << std::endl;
        // std::ofstream ofs("weight");
        // boost::archive::text_oarchive oa(ofs);
        // boost::serialization::save(oa, f.weigths, 1);
        // std::cout<< "write factors" << std::endl;
        // std::ofstream ofs2("factors");
        // boost::archive::text_oarchive oa2(ofs2);
        // boost::serialization::save(oa2, f.V, 1);

        // FactorMachine f_test(sum_count, k);
        // std::cout<< "load weight" << std::endl;
        // std::ifstream ifs("weight");
        // boost::archive::text_iarchive ia(ifs);
        // boost::serialization::load(ia, f_test.weigths, 1);
        // std::cout<< "load factors" << std::endl;
        // std::ifstream ifs2("factors");
        // boost::archive::text_iarchive ia2(ifs2);
        // boost::serialization::load(ia2, f_test.V, 1);

        // test
        ofstream myfile;
        myfile.open ("result.csv");
        for(size_t i = 0; i< data.triplets_test.size()/batch_size; i++){
            // create sparce matrix
            SpMat spMat(batch_size, sum_count);
            Vector Y(batch_size);
            size_t right_border = std::min((i + 1) * batch_size, data.triplets_test.size());
            std::vector<T> vec;
            std::vector<T> Y_vec;
            for(size_t j = 0; j < right_border - i * batch_size; j++){
                triplet a = data.triplets_test[j];
                vec.push_back(T(j, a.user_id, 1));
                vec.push_back(T(j, user_count + a.metadata_ownerId, 1));
                for( size_t i=0; i < a.themes.size(); i++){
                    vec.push_back(T(j, user_count + autor_count + a.themes[i], 1));
                }
                // Y[j] = float(a.r);
            }

            spMat.setFromTriplets(vec.begin(), vec.end());
            spMat.makeCompressed();

            // fit model
            auto y_pred = f.predict(spMat);
            for(int i=0; i < y_pred.size(); i++){
                myfile << data.triplets_test[i].user_id << "\t"  << data.triplets_test[i].object_id
                       << "\t" << y_pred[i] << "\n";
            }
            
        }
        myfile.close();
    // }
}



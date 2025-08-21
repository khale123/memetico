
/**
 * @file
 * @author andy@impv.au
 * @version 1.0
 * @brief Entry point and harness for the memetic algorithm
 * 
 * @bug Need to add .devcontainer and .vscode configurations files to the list of files 
 * @bug Need to change all clone() functions to copy constructors
 * 
 */

// Local
#include "doctest.h"
#include <memetico/data/data_set.h>
#include <memetico/helpers/distance.h>

// Std
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

inline vector<vector<double>> get_test_data(string file_name) {
    ifstream file(file_name);
    string line;
    vector<vector<double>> data;

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;
        while (getline(ss, cell, ',')) {
            row.push_back(stod(cell));
        }
        data.push_back(row);
    }
    return data;
}

inline vector<vector<vector<double>>> uncollapse_hesses(vector<vector<double>> data_set, size_t dim) {
    vector<vector<vector<double>>> hesses;
    for (size_t i = 0; i < data_set.size(); i++) {
        vector<vector<double>> hess;
        for (size_t j = 0; j < dim; j++) {
            vector<double> row;
            for (size_t k = 0; k < dim; k++) {
                row.push_back(data_set[i][j * dim + k]);
            }
            hess.push_back(row);
        }
        hesses.push_back(hess);
    }
    return hesses;
}

inline double MSE_hess_test(DataSet ds, string hess_data_file) {
    double mse = 0.0;
    vector<vector<double>> hess_data = get_test_data(hess_data_file);
    vector<vector<vector<double>>> hesses = uncollapse_hesses(hess_data, ds.samples[0].size());
    vector<vector<double>> true_hess = hesses[0];
    for (size_t i = 0; i < ds.get_count(); ++i) {
        vector<vector<double>> true_hess = hesses[i];
        vector<vector<double>> app_hess = ds.app_hesses[i];
        double err = frobenius_metric_matrix(true_hess, app_hess);
        mse += err * err; // Squared error
    }
    mse /= ds.get_count();
    return mse;
}

inline double MSE_grad_test(DataSet ds, string grad_data_file) {
    double mse = 0.0;
    vector<vector<double>> grad_data = get_test_data(grad_data_file);
    for (size_t i = 0; i < ds.get_count(); ++i) {
        vector<double> true_grad = grad_data[i];
        vector<double> app_grad = ds.app_grads[i];
        double err = euclidean_distance(true_grad, app_grad);
        mse += err * err; // Squared error
    }
    mse /= ds.get_count();
    return mse;        
}

TEST_CASE("Dataset: compute_app_der_multiV") {
    cout << "BRUH"  << endl;
    const vector<string> func_names = {"f1", "f2", "f3", "f4"};
    const vector<size_t> num_samples = {128, 256, 512, 1024, 2048, 4096, 8192};
    const string rel_path = "multiV_test_data/";
    for (const auto& func_name : func_names) {
        cout << "Function: " << func_name << endl;
        // Loop through different sample sizes for each function
        for (const auto& num_sample : num_samples) {
            string file_name = rel_path + func_name + "_data_" + to_string(num_sample) + ".csv";
            DataSet ds(file_name);
            cout << "Loading dataset: " << file_name << endl;
            ds.load();
            cout << "Dataset loaded." << endl;
            string grad_file_name = rel_path + func_name + "_grads_" + to_string(num_sample) + ".csv";
            double mse = MSE_grad_test(ds, grad_file_name);
            cout << "Sample Size: " << num_sample << ", Grad App MSE: " << mse << endl;
            string hess_file_name = rel_path + func_name + "_hesses_" + to_string(num_sample) + ".csv";
            double hess_mse = MSE_hess_test(ds, hess_file_name);
            cout << "Sample Size: " << num_sample << ", Hess App MSE: " << hess_mse << endl;
        }
    }
}

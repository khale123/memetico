
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
#include <memetico/args.h>
#include <memetico/globals.h>
#include <memetico/helpers/rng.h>
#include <memetico/data/data_set.h>
#include <memetico/optimise/objective.h>
#include <memetico/models/regression.h>
#include <memetico/models/cont_frac_dd.h>
#include <memetico/models/branch_cont_frac_dd.h>
#include <memetico/population/pop.h>
#include <memetico/global_types.h>
#include <memetico/helpers/distance.h>
#include <mpi.h>

// Std
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

// Logic Globals
bool            meme::GPU = false;
uint_fast32_t   meme::SEED = 42;
size_t          meme::GENERATIONS = 200;
double          meme::MUTATE_RATE = 0.2;
size_t          meme::LOCAL_SEARCH_INTERVAL = 1;
size_t          meme::STALE_RESET = 5;
bool            meme::INT_ONLY = false;
size_t          meme::LOCAL_SEARCH_RUNS = 4;
size_t          meme::NELDER_MEAD_STALE = 10;
size_t          meme::NELDER_MEAD_MOVES = 1500;
double          meme::LOCAL_SEARCH_DATA_PCT = 0;
double          meme::PENALTY = 0;
size_t          meme::GEN = 0;
long int        meme::MAX_TIME = 10*60;
long int        meme::RUN_TIME = 0;
double          meme::EPSILON = 0;

size_t          meme::DEPTH = 4;
size_t          meme::POCKET_DEPTH = 1;
size_t          meme::DIVERSITY_COUNT = 3;

// Derivative Globals
size_t          meme::IFR = 0;
string          meme::IN_DER = "exact";
size_t          meme::MAX_DER_ORD = 3;
// Least Squares Globals
// To-do: Implement a way to set this in args
size_t          meme::NUM_NEIGHBORS = 16;

DynamicDepthType meme::DYNAMIC_DEPTH_TYPE = DynamicNone;

// File Globals
string          meme::TRAIN_FILE = "sinx.csv";
string          meme::TEST_FILE = "sinx.csv";
string          meme::LOG_DIR = "out/";
ofstream        meme::master_log;

// Technical Globals
size_t          meme::PREC = 18;
bool            meme::DEBUG = false;

// Global Heplers
RandReal        meme::RANDREAL;
RandInt         meme::RANDINT;

// Local Helpers
FILE*           meme::STD_OUT;
FILE*           meme::STD_ERR;



vector<vector<double>> get_test_data(string file_name) {
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

vector<vector<vector<double>>> uncollapse_hesses(vector<vector<double>> data_set, size_t dim) {
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

double MSE_hess_test(DataSet ds, string hess_data_file) {
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

double MSE_grad_test(DataSet ds, string grad_data_file) {
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

int main() {
    const vector<string> func_names = {"f1", "f2", "f3", "f4"};
    const vector<size_t> num_samples = {128, 256, 512, 1024, 2048, 4096, 8192};
    const string rel_path = "multiV_test_data/";
    for (const auto& func_name : func_names) {
        cout << "Function: " << func_name << endl;
        // Loop through different sample sizes for each function
        for (const auto& num_sample : num_samples) {
            string file_name = rel_path + func_name + "_data_" + to_string(num_sample) + ".csv";
            DataSet ds(file_name);
            ds.load();
            ds.compute_app_der_multiV();
            string grad_file_name = rel_path + func_name + "_grads_" + to_string(num_sample) + ".csv";
            double mse = MSE_grad_test(ds, grad_file_name);
            cout << "Sample Size: " << num_sample << ", Grad App MSE: " << mse << endl;
            string hess_file_name = rel_path + func_name + "_hesses_" + to_string(num_sample) + ".csv";
            double hess_mse = MSE_hess_test(ds, hess_file_name);
            cout << "Sample Size: " << num_sample << ", Hess App MSE: " << hess_mse << endl;
        }
    }
    return 0;
}

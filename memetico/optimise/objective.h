
/**
 * @file
 * @author andy@impv.au
 * @version 1.0
 * @brief Collection of objective functions
*/

#ifndef MEMETICO_OBJECTIVE_H
#define MEMETICO_OBJECTIVE_H

#include <memetico/helpers/safe_ops.h>
#include <memetico/data/data_set.h>
#include <memetico/model_base/model.h>
#include <memetico/model_base/model_meme.h>
#include <memetico/gpu/cuda.cuh>
#include <memetico/globals.h>
#include <memetico/helpers/distance.h>
#include "finitediff_templated.hpp"

namespace objective {

// See Implementation for details

template <class U>
double mse(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected = vector<size_t>());

template <class U>
double mse_der(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected = vector<size_t>());

template <class U>
double mse_der_multiV(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected = vector<size_t>());

template <class U>
double mae(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected = vector<size_t>());

template <class U>
double mape(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected = vector<size_t>());

template <class U>
double rmse(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected = vector<size_t>());

template <class U>
double cuda_error(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected = vector<size_t>(), metric_t metric = metric_t::mean_square_error);

template <class U>
double nmse(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected = vector<size_t>());

template <class U>
double compare(MemeticModel<U>* m1, MemeticModel<U>* m2, DataSet* train);

template <class U>
double p_cor(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected = vector<size_t>());

template <class U>
double s_cor(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected = vector<size_t>());

template <class U>
vector<vector<double>> fornberg(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected = vector<size_t>());

template <class U>
vector<vector<double>> fornberg2(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected = vector<size_t>());

template <class U>
vector<vector<double>> derivative(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected = vector<size_t>());

/** 
 * Has similar functionality to fornberg method, evalutes the model and approximates its gradients and hessians
*/
template <class U>
tuple<vector<double>, vector<vector<double>>, vector<vector<vector<double>>>> multiV_derivative(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected = vector<size_t>()); 

template <class U>
tuple<vector<double>, vector<vector<double>>, vector<vector<vector<double>>>> LS_multiV(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected = vector<size_t>());


vector<double> s_rank(vector<double>& data);

}

#include <memetico/optimise/objective.tpp>

#endif

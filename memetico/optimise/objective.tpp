
/**
 * @file
 * @author Andrew Ciezak <andy@impv.au>
 * @version 1.0
 * @brief Implementation of objective functions
*/

using namespace meme;

/**
 * Mean Sqaure Error objective function
 * 
 * @param model Model to evaluate
 * @param train DataSet to determine error on
 * @param selected subset of data to evaluate. Empty subset indicates usage of all data
 * @return double
 * 
 */
template <class U>
double objective::mse(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected ) {

    auto start = chrono::system_clock::now();

    if( !train->get_gpu() ) {
        try {
            double weight_sum = 0;
            double error_sum = 0;
            double error;
            if( selected.size() == 0) {
                for(size_t i = 0; i < train->get_count(); i++) {
                    error = model->evaluate(train->samples[i]);
		            //error = (error-train->y_min)/(train->y_max-train->y_min);
 
                    // Determine residual and square
                    error = add(error, -train->y[i]);
                    error = multiply(error, error);
                    // Weight squared error
                    if(train->has_weight()) {
                        error = multiply(error, train->weight[i]);
                        weight_sum += train->weight[i];
                    }
                    
                    // Sum of squared error
                    error_sum = add(error_sum, error);
                }
                // After the loop, use weight_sum to calculate the average error
                if(weight_sum > 0)  model->set_error(error_sum / weight_sum);
                else                model->set_error(error_sum / train->get_count());
            } else {
                for(size_t i : selected) {
                    error = model->evaluate(train->samples[i]);
		            //error = (error-train->y_min)/(train->y_max-train->y_min);
                    
                    // Determine residual and square
                    error = add(error, -train->y[i]);
                    error = multiply(error, error);
                    
                    // Weight squared error
                    if(train->has_weight()) {
                        error = multiply(error, train->weight[i]);
                        weight_sum += train->weight[i];
                    }
                    // Sum of squared error
                    error_sum = add(error_sum, error);
                }
                // After the loop, use weight_sum to calculate the average error
                if(weight_sum > 0)  model->set_error(error_sum / weight_sum);
                else                model->set_error(error_sum / train->get_count());
            }
            model->set_penalty( 1+model->get_count_active()*meme::PENALTY );
            model->set_fitness( multiply(model->get_error(),model->get_penalty()) );
        } catch (exception& e) {
            model->set_error(numeric_limits<double>::max());
            model->set_penalty(numeric_limits<double>::max());
            model->set_fitness(numeric_limits<double>::max());
        }
    } else  {
        model->set_error( cuda_error(model, train, selected) );
        model->set_penalty( 1+model->get_count_active()*meme::PENALTY );
        model->set_fitness( multiply(model->get_error(),model->get_penalty()) );
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = end-start;
    
    return model->get_fitness();
}

/**
 * mse_der
 * 
 */
template <class U>
double objective::mse_der(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected ) {

    auto start = chrono::system_clock::now();
    if( !train->get_gpu() ) {
	
	vector<vector<double>> Ypreds;

	try {
	    double error_sum = 0;
	    double error;
	    int counter = 0;

        if( meme::IN_DER == "app-fd" )  Ypreds = fornberg(model, train, selected);
        else                            Ypreds = derivative(model, train, selected);

	    if( selected.size() == 0) {
            // 0th order derivative
            for(size_t i = 0; i < Ypreds[0].size(); i++) {
                error = (Ypreds[0][i] - train->y_min) / (train->y_max - train->y_min);
                error = add(error, -train->y[i]);
                error = multiply(error, error);
                error_sum = add(error_sum, error);
                counter++;
            }
            // higher order derivatives
            for(size_t k = 1; k < Ypreds.size(); k++) {
                for(size_t i = 0; i < Ypreds[k].size(); i++) {
                    error = (Ypreds[k][i] - train->yder_min[k-1]) / (train->yder_max[k-1] - train->yder_min[k-1]);
                    error = add(error, -train->Yder[k-1][i]);
                    error = multiply(error, error);
                    error_sum = add(error_sum, error);
                    counter++;
                }
            }
        } else {
            // 0th order derivative
            for(size_t i = 0; i < Ypreds[0].size(); i++) {
                error = (Ypreds[0][i] - train->y_min) / (train->y_max - train->y_min);
                error = add(error, -train->y[selected[i]]);
                error = multiply(error, error);
                error_sum = add(error_sum, error);
                counter++;
            }
            // higher order derivatives
            for(size_t k = 1; k < Ypreds.size(); k++) {
                for(size_t i = 0; i < Ypreds[k].size(); i++) {
                    error = (Ypreds[k][i] - train->yder_min[k-1]) / (train->yder_max[k-1] - train->yder_min[k-1]);
                    error = add(error, -train->Yder[k-1][selected[i]]);
                    error = multiply(error, error);
                    error_sum = add(error_sum, error);
                    counter++;
                }
            }
	    }
	    model->set_error(error_sum / counter);
	    model->set_penalty( 1+model->get_count_active()*meme::PENALTY );
	    model->set_fitness( multiply(model->get_error(),model->get_penalty()) );
	} catch (exception& e) {
	    // cout << "[objective.tpp/mse_der] numerical exception" << endl;
	    // model->print();
	    model->set_error(numeric_limits<double>::max());
	    model->set_penalty(numeric_limits<double>::max());
	    model->set_fitness(numeric_limits<double>::max());
	}
    } else  {
	model->set_error( cuda_error(model, train, selected) );
	model->set_penalty( 1+model->get_count_active()*meme::PENALTY );
	model->set_fitness( multiply(model->get_error(),model->get_penalty()) );
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = end-start;
    
    return model->get_fitness();
}



/**
 * mse_der_multiV
 * essentially the same as mse_der, but uses distance functions to handle comparing gradients and hessians
 */
template <class U>
double objective::mse_der_multiV(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected ) {

    auto start = chrono::system_clock::now();
    if( !train->get_gpu() ) {
	
    // Packages predictions, gradients and hessians
	std::tuple<vector<double>, vector<vector<double>>, vector<vector<vector<double>>>> Ypreds;

	try {
	    double error_sum = 0;
	    double error;

        if( meme::IN_DER == "app-multiV" )  Ypreds = LS_multiV(model, train, selected);
        // THERE IS NO IMLEMENTATION FOR multiV_derivative yet!!
        else                            Ypreds = multiV_derivative(model, train, selected);

        auto& preds = std::get<0>(Ypreds);  // vector<double>
        auto& grads = std::get<1>(Ypreds);  // vector<vector<double>>
        auto& hesses = std::get<2>(Ypreds); // vector<vector<vector<double>>>

        if (meme::NORMALIZATION_FLAG == true) {
            // TODO: implement normalization of grads and hessians

        }
        else {

            if( selected.size() == 0) {
                for(size_t i = 0; i < preds.size(); i++) {
                    // predictions
                    error = add(preds[i], -train->y[i]);
                    error = multiply(error, error);
                    error_sum = add(error_sum, error);
                    // gradients
                    error = euclidean_distance(grads[i], train->app_grads[i]);
                    error = multiply(error, error);
                    error_sum = add(error_sum, error);
                    // hessians
                    error = frobenius_metric_matrix(hesses[i], train->app_hesses[i]);
                    error = multiply(error, error);
                    error_sum = add(error_sum, error);   
                }
            } 
            
            else {
                for(size_t i = 0; i < preds.size(); i++) {
                    // predictions
                    error = add(preds[i], -train->y[selected[i]]);
                    error = multiply(error, error);
                    error_sum = add(error_sum, error);
                    // gradients
                    error = euclidean_distance(grads[i], train->app_grads[selected[i]]);
                    error = multiply(error, error);
                    error_sum = add(error_sum, error);
                    // hessians
                    error = frobenius_metric_matrix(hesses[i], train->app_hesses[selected[i]]);
                    error = multiply(error, error);
                    error_sum = add(error_sum, error);   
                }
            }
        }
	    model->set_error(error_sum / preds.size());
	    model->set_penalty( 1+model->get_count_active()*meme::PENALTY );
	    model->set_fitness( multiply(model->get_error(),model->get_penalty()) );
	} catch (exception& e) {
	    // cout << "[objective.tpp/mse_der] numerical exception" << endl;
	    // model->print();
	    model->set_error(numeric_limits<double>::max());
	    model->set_penalty(numeric_limits<double>::max());
	    model->set_fitness(numeric_limits<double>::max());
	}
    } else  {
	model->set_error( cuda_error(model, train, selected) );
	model->set_penalty( 1+model->get_count_active()*meme::PENALTY );
	model->set_fitness( multiply(model->get_error(),model->get_penalty()) );
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = end-start;
    return model->get_fitness();
}



template <class U>
double objective::mae(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected ) {

    auto start = chrono::system_clock::now();

    if( !train->get_gpu() ) {

        try {

            double error_sum = 0;
            double error;
        
            if( selected.size() == 0) {
                
                for(size_t i = 0; i < train->get_count(); i++) {

                    double frac_val = model->evaluate(train->samples[i]);

                    // Determine residual and square
                    error = add(frac_val, -train->y[i]);
                    error = fabs(error);

                    // Weight squared error
                    if(train->has_weight())
                        error = multiply(error, train->weight[i]);

                    // Sum of squared error
                    error_sum = add(error_sum, error);

                }

                model->set_error(error_sum / train->get_count());

            } else {

                for(size_t i : selected) {

                    double frac_val = model->evaluate(train->samples[i]);

                    // Determine residual and square
                    error = add(frac_val, -train->y[i]);
                    error = fabs(error);

                    // Weight squared error
                    if(train->has_weight())
                        error = multiply(error, train->weight[i]);

                    // Sum of squared error
                    error_sum = add(error_sum, error);
                }
                
                model->set_error(error_sum / selected.size());

            }
            
            model->set_penalty( 1+model->get_count_active()*meme::PENALTY );
            model->set_fitness( multiply(model->get_error(),model->get_penalty()) );

        } catch (exception& e) {

            model->set_error(numeric_limits<double>::max());
            model->set_penalty(numeric_limits<double>::max());
            model->set_fitness(numeric_limits<double>::max());

        }

    } else  {

        model->set_error( cuda_error(model, train, selected, metric_t::mean_absolute_error) );
        model->set_penalty( 1+model->get_count_active()*meme::PENALTY );
        model->set_fitness( multiply(model->get_error(),model->get_penalty()) );
    }
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = end-start;
    
    return model->get_fitness();

}

template <class U>
double objective::mape(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected) {

    auto start = chrono::system_clock::now();

    if( !train->get_gpu() ) {

        try {

            double error_sum = 0;
            double error;
        
            if( selected.size() == 0) {
                
                for(size_t i = 0; i < train->get_count(); i++) {

                    double pred_val = model->evaluate(train->samples[i]);

                    // Calculate absolute percentage error
                    if (train->y[i] != 0) { // Avoid division by zero
                        error = fabs((pred_val - train->y[i]) / train->y[i]);
                    } else {
                        error = 0; // Handle zero actual value case
                    }

                    // Weight error
                    if(train->has_weight())
                        error = multiply(error, train->weight[i]);

                    // Sum of errors
                    error_sum = add(error_sum, error);

                }

                model->set_error(error_sum / train->get_count());

            } else {

                for(size_t i : selected) {

                    double pred_val = model->evaluate(train->samples[i]);

                    // Calculate absolute percentage error
                    if (train->y[i] != 0) {
                        error = fabs((pred_val - train->y[i]) / train->y[i]);
                    } else {
                        error = 0;
                    }

                    // Weight error
                    if(train->has_weight())
                        error = multiply(error, train->weight[i]);

                    // Sum of errors
                    error_sum = add(error_sum, error);
                }
                
                model->set_error(error_sum / selected.size());

            }
            
            model->set_penalty( 1+model->get_count_active()*meme::PENALTY );
            model->set_fitness( multiply(model->get_error(),model->get_penalty()) );

        } catch (exception& e) {

            model->set_error(numeric_limits<double>::max());
            model->set_penalty(numeric_limits<double>::max());
            model->set_fitness(numeric_limits<double>::max());

        }

    } else  {

        throw std::runtime_error("GPU branch not implemented for MAPE");

    }
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = end-start;
    
    return model->get_fitness();

}

template <class U>
double objective::rmse(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected ) {

    mse(model, train, selected);

    try {

        model->set_error( sqrt(model->get_error()) );
        model->set_fitness( multiply(model->get_error(),model->get_penalty()));

    } catch (exception& e) {

        model->set_error(numeric_limits<double>::max());
        model->set_penalty(numeric_limits<double>::max());
        model->set_fitness(numeric_limits<double>::max());

    }

    return model->get_fitness();

}

template <class U>
double objective::cuda_error(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected, metric_t metric) {

    // Convert CFR to Program
    Program p;
    TreeNode *tn_frac = new TreeNode();
    model->get_node(tn_frac);
    get_prefix(p.prefix, tn_frac);
    p.length = p.prefix.size();

    // Setup call to GPU Calculate Fitness
    vector<Program> pop;
    pop.push_back(p);
    int blockNum;
    
    if(selected.size() == 0) {
        blockNum = (train->get_count() - 1) / THREAD_PER_BLOCK + 1;
        train->device_data.subset_size = 0;
    } else
        blockNum = (selected.size() - 1) / THREAD_PER_BLOCK + 1;


    double ret = cusr::calculateFitness(train->device_data, blockNum, pop, metric);
    delete tn_frac;
    return ret;

}

template <class U>
double objective::p_cor(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected) {

        auto start = chrono::system_clock::now();

    try {
        double pearson_correlation;
        const double epsilon = 1e-5; // Small threshold for variance

        if (train->has_weight()) {
            double sum_weight = 0;
            double weighted_mean_x = 0, weighted_mean_y = 0;
            double weighted_m2_x = 0, weighted_m2_y = 0, weighted_crossproduct = 0;

            auto calculateWeightedCorrelation = [&](size_t i) {
                double x = model->evaluate(train->samples[i]);
                double y = train->y[i];
                double weight = train->weight[i];
                sum_weight += weight;

                double delta_x = x - weighted_mean_x;
                double delta_y = y - weighted_mean_y;
                weighted_mean_x += (delta_x * weight) / sum_weight;
                weighted_mean_y += (delta_y * weight) / sum_weight;
                weighted_m2_x += weight * delta_x * (x - weighted_mean_x);
                weighted_m2_y += weight * delta_y * (y - weighted_mean_y);
                weighted_crossproduct += weight * delta_x * (y - weighted_mean_y);
            };

            if (selected.size() == 0) {
                for (size_t i = 0; i < train->get_count(); i++) {
                    calculateWeightedCorrelation(i);
                }
            } else {
                for (size_t i : selected) {
                    calculateWeightedCorrelation(i);
                }
            }

            double weighted_variance_x = weighted_m2_x / sum_weight;
            double weighted_variance_y = weighted_m2_y / sum_weight;
            double weighted_covariance = weighted_crossproduct / sum_weight;

            if (weighted_variance_x < epsilon || weighted_variance_y < epsilon) {
                model->set_error(1);
                model->set_penalty(1);
                model->set_fitness(1);
                return model->get_fitness();
            }

            pearson_correlation = weighted_covariance / sqrt(weighted_variance_x * weighted_variance_y);

        } else {
            size_t n = 0;
            double mean_x = 0, mean_y = 0;
            double m2_x = 0, m2_y = 0, crossproduct = 0;

            auto calculateCorrelation = [&](size_t i) {
                double x = model->evaluate(train->samples[i]);
                double y = train->y[i];
                n++;
                double delta_x = x - mean_x;
                double delta_y = y - mean_y;
                mean_x += delta_x / n;
                mean_y += delta_y / n;
                m2_x += delta_x * (x - mean_x);
                m2_y += delta_y * (y - mean_y);
                crossproduct += delta_x * (y - mean_y);
            };

            if (selected.size() == 0) {
                for(size_t i = 0; i < train->get_count(); i++) {
                    calculateCorrelation(i);
                }
            } else {
                for (size_t i : selected) {
                    calculateCorrelation(i);
                }
            }  

            double variance_x = m2_x / n;
            double variance_y = m2_y / n;

            if (variance_x < epsilon || variance_y < epsilon) {
                model->set_error(1);
                model->set_penalty(1);
                model->set_fitness(1);
                return model->get_fitness();
            }

            pearson_correlation = crossproduct / n / sqrt(variance_x * variance_y);
        }

        if (abs(pearson_correlation) > 1) {
            pearson_correlation = 1;
        }

        model->set_error(1 - abs(pearson_correlation));
        model->set_penalty(1+model->get_count_active()*meme::PENALTY);
        model->set_fitness(1 - abs(pearson_correlation/model->get_penalty()) );

    } catch (const std::exception& e) {
        model->set_error(1);
        model->set_penalty(1);
        model->set_fitness(1);
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> ms = end - start;

    if (std::isnan(model->get_error())) {
        model->set_error(1);
        model->set_penalty(1);
        model->set_fitness(1);
    }

    return model->get_fitness();

}

template <class U>
double objective::s_cor(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected) {

    auto start = chrono::system_clock::now();

    vector<double> x_values;
    vector<double> y_values;

    // Populate x_values and y_values
    if (selected.size() == 0) {
        for(size_t i = 0; i < train->get_count(); i++) {
            x_values.push_back(model->evaluate(train->samples[i]));
            y_values.push_back(train->y[i]);
        }
    } else {
        for (size_t i : selected) {
            x_values.push_back(model->evaluate(train->samples[i]));
            y_values.push_back(train->y[i]);
        }
    }

    // Get ranks for x and y values
    vector<double> x_ranks = s_rank(x_values);
    vector<double> y_ranks = s_rank(y_values);

    size_t n = x_ranks.size();
    double mean_x = 0, mean_y = 0;
    double m2_x = 0, m2_y = 0, crossproduct = 0;

    for (size_t i = 0; i < n; ++i) {
        double x = x_ranks[i];
        double y = y_ranks[i];
        double delta_x = x - mean_x;
        double delta_y = y - mean_y;
        mean_x += delta_x / (i + 1);
        mean_y += delta_y / (i + 1);
        m2_x += delta_x * (x - mean_x);
        m2_y += delta_y * (y - mean_y);
        crossproduct += delta_x * (y - mean_y);
    }

    double variance_x = m2_x / n;
    double variance_y = m2_y / n;

    // Check for near-constant ranks
    const double epsilon = 1e-8;  // Small threshold for variance
    if (variance_x < epsilon || variance_y < epsilon) {
        model->set_error(1);
        model->set_penalty(1);
        model->set_fitness(1);
        return model->get_fitness();
    }

    double covariance = crossproduct / n;
    double spearman_correlation = covariance / sqrt(variance_x * variance_y);

    if (abs(spearman_correlation) > 1) {
        return 1;
    }

    model->set_error(1 - abs(spearman_correlation));
    model->set_penalty(1);
    model->set_fitness(1 - abs(spearman_correlation));

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> ms = end-start;

    return model->get_fitness();
}

// Helper function to compute ranks
inline vector<double> objective::s_rank(vector<double>& data) {
    vector<size_t> indices(data.size());
    iota(indices.begin(), indices.end(), 0);

    sort(indices.begin(), indices.end(), [&data](size_t a, size_t b) { return data[a] < data[b]; });

    vector<double> ranks(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        ranks[indices[i]] = i + 1;
    }

    return ranks;
}

/**
 * Normalised Mean Sqaure Error objective function
 * 
 * @param model Model to evaluate
 * @param train DataSet to determine error on
 * @param selected subset of data to evaluate. Empty subset indicates usage of all data
 * @return double
 * 
 */
template <class U>
double objective::nmse(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected ) {

    try {

        // Basic vars
        double error_sum = 0;       // Sum of errors for all samples
        double target_sum = 0;      // Sum of target values for all samples
        double error;               // Error for a single sample
        double predict;             // Prediction for a single sample

        double target_avg;          // Average target value

        // Variance
        double variance_sample;     // Variance for a single sample
        double variance_sum = 0;    // Sum of variance for all samples
        double variance;            // Variance given model and train data

        double mse;                 // MSE given model and train data

        if( selected.size() == 0) {

            for(size_t i = 0; i < train->get_count(); i++) {

                predict = model->evaluate(train->samples[i]);

                // Add  target sums
                target_sum = add(target_sum, train->y[i]);

                // Determine error and square
                error = add(predict, -train->y[i]);
                error = multiply(error, error);    

                // Weight squared error
                if(train->has_weight())
                    error = multiply(error, train->weight[i]);

                // Sum of squared error
                error_sum = add(error_sum, error);

            }

            mse = error_sum/train->get_count();
            target_avg = target_sum/train->get_count();

            // Determine variance
            for(size_t i = 0; i < train->get_count(); i++) {                
                variance_sample = add(train->y[i],-target_avg);
                variance_sample = multiply(variance_sample,variance_sample);
                variance_sum = add(variance_sum, variance_sample);
            }
            variance = variance_sum/(train->get_count()-1);

            model->set_error(mse/variance);
            model->set_penalty(1+model->get_count_active()*meme::PENALTY);
            model->set_fitness(multiply(model->get_error(),model->get_penalty()));

        } else {

            for(size_t i : selected) {

                predict = model->evaluate(train->samples[i]);

                // Add  target sums
                target_sum = add(train->y[i], target_sum);

                // Determine error and square
                error = add(predict, -train->y[i]);
                error = multiply(error, error);    

                // Weight squared error
                if(train->has_weight())
                    error = multiply(error, train->weight[i]);

                // Sum of squared error
                error_sum = add(error_sum, error);             

            }

            mse = error_sum/train->get_count();
            target_avg = target_sum/train->get_count();

            // Determine variance
            for(size_t i : selected) {               
                variance_sample = add(train->y[i],-target_avg);
                variance_sum = add(variance_sum, variance_sample);
            }
            variance = variance_sum/(train->get_count()-1);

            model->set_error(mse/variance);
            model->set_penalty(1+model->get_count_active()*meme::PENALTY);
            model->set_fitness(multiply(model->get_error(),model->get_penalty()));

        }

    } catch (exception& e) {

        model->set_error(numeric_limits<double>::max());
        model->set_penalty(numeric_limits<double>::max());
        model->set_fitness(numeric_limits<double>::max());

    }

    return model->get_fitness();

}

template <class U>
double objective::compare(MemeticModel<U>* m1, MemeticModel<U>* m2, DataSet* train) {

    double error_dist = 0;
    double err1, err2;

    try {

        for(size_t i = 0; i < train->get_count(); i++) {

            double m1_frac_val = m1->evaluate(train->samples[i]);
            double m2_frac_val = m2->evaluate(train->samples[i]);
            
            // Determine residual and square
            err1 = add(m1_frac_val, -train->y[i]);
            err2 = add(m2_frac_val, -train->y[i]);
            
            // Square error
            err1 = multiply(err1, err1);
            err2 = multiply(err2, err2);
            
            // Sum of squared error
            error_dist = add(error_dist, fabs(err1-err2));
            
        }
      
    } catch (exception& e) {
        return numeric_limits<double>::max();    
    }
        
    return error_dist;
}

template <class U>
vector<vector<double>> objective::derivative(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected) {

    vector<vector<double>> Y{{}};

    // 0th derviative
    if(meme::MAX_DER_ORD==0) {

        vector<double> der_0_1;
        Y.push_back({});
        if( selected.size() == 0) {

            for(size_t i = 0; i < train->samples.size(); i++) {
                der_0_1 = model->evaluate_der(train->samples[i]);
                Y[0].push_back(der_0_1[0]);
                der_0_1.clear();
            }
        } else {

            for(size_t i : selected) {
                der_0_1 = model->evaluate_der(train->samples[i]);
                Y[0].push_back(der_0_1[0]);
                der_0_1.clear();		
            }
        }
    }

    // Exact derivative
    // 0th and 1st order derivative
    if(meme::MAX_DER_ORD==1) {

        vector<double> der_0_1;
        Y.push_back({});
        if( selected.size() == 0) {

            for(size_t i = 0; i < train->samples.size(); i++) {
                der_0_1 = model->evaluate_der(train->samples[i]);
                Y[0].push_back(der_0_1[0]);
                Y[1].push_back(der_0_1[1]);
                der_0_1.clear();
            }
        } else {

            for(size_t i : selected) {
                der_0_1 = model->evaluate_der(train->samples[i]);
                Y[0].push_back(der_0_1[0]);
                Y[1].push_back(der_0_1[1]);
                der_0_1.clear();		
            }
        }
    }

    // 0th, 1st and 2nd order derivative
    if(meme::MAX_DER_ORD==2) {

        vector<double> ders;
        Y.push_back({});
        Y.push_back({});
        if( selected.size() == 0) {
            for(size_t i = 0; i < train->samples.size(); i++) {
                ders = model->evaluate_der2(train->samples[i]);
                Y[0].push_back(ders[0]);
                Y[1].push_back(ders[1]);
                Y[2].push_back(ders[2]);
                ders.clear();
            }
        } else {
            for(size_t i : selected) {
                ders = model->evaluate_der2(train->samples[i]);
                Y[0].push_back(ders[0]);
                Y[1].push_back(ders[1]);
                Y[2].push_back(ders[2]);
                ders.clear();		
            }
        }
    }

    // 0th, 1st, 2nd and 3rd order derivative
    if(meme::MAX_DER_ORD==3) {

        vector<double> ders;
        Y.push_back({});
        Y.push_back({});
        Y.push_back({});
        if( selected.size() == 0) {
            for(size_t i = 0; i < train->samples.size(); i++) {
                ders = model->evaluate_der3(train->samples[i]);
                Y[0].push_back(ders[0]);
                Y[1].push_back(ders[1]);
                Y[2].push_back(ders[2]);
                Y[3].push_back(ders[3]);
                ders.clear();
            }
        } else {
            for(size_t i : selected) {
                ders = model->evaluate_der3(train->samples[i]);
                Y[0].push_back(ders[0]);
                Y[1].push_back(ders[1]);
                Y[2].push_back(ders[2]);
                Y[3].push_back(ders[3]);
                ders.clear();		
            }
        }
    }

    return Y;
}

// TODO: Implement forward recurrence algorithm for multiV derivatives and Hessian's?
// Another idea is to just use a very high order derivative finite difference scheme
template <class U>
tuple<vector<double>, vector<vector<double>>, vector<vector<vector<double>>>> objective::multiV_derivative(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected) {

}

// Like fornberg, uses approximation rule found in dataset object to approximate derivatives of CFR model
template <class U>
tuple<vector<double>, vector<vector<double>>, vector<vector<vector<double>>>> objective::LS_multiV(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected) {
    vector<double> predictions;
    vector<vector<double>> pred_grads;
    vector<vector<vector<double>>> pred_hesses;

    // 100% training samples
    if( selected.size() == 0) {

        for(size_t i = 0; i < train->samples.size(); i++) {
            predictions.push_back(model->evaluate(train->samples[i]));
        }

        for (size_t i = 0; i < train->samples.size(); i++) {
            const vector<size_t> neighbors = train->neighbor_indices[i];
            vector<double> fdiffs;
            for (size_t j = 0; j < neighbors.size(); j++) {
                double diff = predictions[neighbors[j]] - predictions[i];
                fdiffs.push_back(diff);
            }
            auto grad_hess_pair = train->apply_FDS_on_arbitrary_response(i, fdiffs);
            pred_grads.push_back(grad_hess_pair.first);
            pred_hesses.push_back(grad_hess_pair.second);
        }
    }

    else {
        for(size_t i : selected) {
            predictions.push_back(model->evaluate(train->samples[i]));
        }

        for (size_t i : selected) {
            const vector<size_t> neighbors = train->neighbor_indices[i];
            vector<double> fdiffs;
            for (size_t j = 0; j < neighbors.size(); j++) {
                double diff = predictions[neighbors[j]] - predictions[i];
                fdiffs.push_back(diff);
            }
            auto grad_hess_pair = train->apply_FDS_on_arbitrary_response(i, fdiffs);
            pred_grads.push_back(grad_hess_pair.first);
            pred_hesses.push_back(grad_hess_pair.second);
        }
    }
    return make_tuple(predictions, pred_grads, pred_hesses);
}




template <class U>
vector<vector<double>> objective::fornberg(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected) {
    
    vector<vector<double>> Y{{}};

    // 100% training samples
    if( selected.size() == 0) {
        
        // Store in first return the target value
        for(size_t i = 0; i < train->samples.size(); i++)
            Y[0].push_back(model->evaluate(train->samples[i]));

        // Calculate the 1st order derivative given the predicted values from model
        if(meme::MAX_DER_ORD>=1) {

            Y.push_back({});
            
            // first interval samples[0], samples[1]
            Y[1].push_back( train->fd_weights[0][0][0]*Y[0][0] + train->fd_weights[0][0][1]*Y[0][1] );
            
            // intermediate intervals samples[i-1], samples[i], samples[i+1]
            for(size_t i = 1; i < Y[0].size()-1; i++)
                Y[1].push_back( train->fd_weights[0][i][0]*Y[0][i-1] + train->fd_weights[0][i][1]*Y[0][i] + train->fd_weights[0][i][2]*Y[0][i+1] );
            
            // last interval samples[-2], samples[-1]
            Y[1].push_back( train->fd_weights[0][Y[0].size()-1][0]*Y[0][Y[0].size()-2] + train->fd_weights[0][Y[0].size()-1][1]*Y[0][Y[0].size()-1] );

        }
        
        // Calculate the 2nd order derivative given the predicted values from model
        if(meme::MAX_DER_ORD>=2) {

            Y.push_back({});

            // first interval samples[0], samples[1], samples[2]
            Y[2].push_back( train->fd_weights[1][0][0]*Y[0][0] + train->fd_weights[1][0][1]*Y[0][1] + train->fd_weights[1][0][2]*Y[0][2] );
            
            // second interval samples[0], samples[1], samples[2], samples[3]
            Y[2].push_back( train->fd_weights[1][1][0]*Y[0][0] + train->fd_weights[1][1][1]*Y[0][1] + train->fd_weights[1][1][2]*Y[0][2] + train->fd_weights[1][1][3]*Y[0][3] );
            
            // intermediate intervals samples[i-2], samples[i-1], samples[i], samples[i+1], samples[i+2]
            for(size_t i = 2; i < Y[0].size()-2; i++)
                Y[2].push_back( train->fd_weights[1][i][0]*Y[0][i-2] + train->fd_weights[1][i][1]*Y[0][i-1] + train->fd_weights[1][i][2]*Y[0][i] + train->fd_weights[1][i][3]*Y[0][i+1] + train->fd_weights[1][i][4]*Y[0][i+2] );
            
            // one before last interval samples[-4], samples[-3], samples[-2], samples[-1]
            Y[2].push_back( train->fd_weights[1][Y[0].size()-2][0]*Y[0][Y[0].size()-4] + train->fd_weights[1][Y[0].size()-2][1]*Y[0][Y[0].size()-3] + train->fd_weights[1][Y[0].size()-2][2]*Y[0][Y[0].size()-2] + train->fd_weights[1][Y[0].size()-2][3]*Y[0][Y[0].size()-1] );
            
            // last interval samples[-3], samples[-2], samples[-1]
            Y[2].push_back( train->fd_weights[1][Y[0].size()-1][0]*Y[0][Y[0].size()-3] + train->fd_weights[1][Y[0].size()-1][1]*Y[0][Y[0].size()-2] + train->fd_weights[1][Y[0].size()-1][2]*Y[0][Y[0].size()-1] );
        }

        // 3rd order derivative
        if(meme::MAX_DER_ORD>=3) {

            Y.push_back({});

            // first interval samples[0], samples[1], samples[2], samples[3]
            Y[3].push_back(
                train->fd_weights[2][0][0]*Y[0][0]
                + train->fd_weights[2][0][1]*Y[0][1]
                + train->fd_weights[2][0][2]*Y[0][2]
                + train->fd_weights[2][0][3]*Y[0][3] 
            );

            // second interval samples[0], samples[1], samples[2], samples[3], samples[4]
            Y[3].push_back(
                train->fd_weights[2][1][0]*Y[0][0]
                + train->fd_weights[2][1][1]*Y[0][1]
                + train->fd_weights[2][1][2]*Y[0][2]
                + train->fd_weights[2][1][3]*Y[0][3]
                + train->fd_weights[2][1][4]*Y[0][4] 
            );

            // third interval samples[0], samples[1], samples[2], samples[3], samples[4], samples[5]
            Y[3].push_back(
                train->fd_weights[2][2][0]*Y[0][0]
                + train->fd_weights[2][2][1]*Y[0][1]
                + train->fd_weights[2][2][2]*Y[0][2]
                + train->fd_weights[2][2][3]*Y[0][3]
                + train->fd_weights[2][2][4]*Y[0][4]
                + train->fd_weights[2][2][5]*Y[0][5] 
            );

            // intermediate intervals samples[i-3], samples[i-2], samples[i-1], samples[i], samples[i+1], samples[i+2], samples[i+3]
            for(size_t i = 3; i < Y[0].size()-3; i++) {
                Y[3].push_back(
                    train->fd_weights[2][i][0]*Y[0][i-3]
                    + train->fd_weights[2][i][1]*Y[0][i-2]
                    + train->fd_weights[2][i][2]*Y[0][i-1]
                    + train->fd_weights[2][i][3]*Y[0][i]
                    + train->fd_weights[2][i][4]*Y[0][i+1]
                    + train->fd_weights[2][i][5]*Y[0][i+2]
                    + train->fd_weights[2][i][6]*Y[0][i+3] 
                );
            }

            // two before last interval samples[-6], samples[-5], samples[-4], samples[-3], samples[-2], samples[-1]
            Y[3].push_back(
                train->fd_weights[2][Y[0].size()-3][0]*Y[0][Y[0].size()-6]
                + train->fd_weights[2][Y[0].size()-3][1]*Y[0][Y[0].size()-5]
                + train->fd_weights[2][Y[0].size()-3][2]*Y[0][Y[0].size()-4]
                + train->fd_weights[2][Y[0].size()-3][3]*Y[0][Y[0].size()-3]
                + train->fd_weights[2][Y[0].size()-3][4]*Y[0][Y[0].size()-2]
                + train->fd_weights[2][Y[0].size()-3][5]*Y[0][Y[0].size()-1] 
            );

            // one before last interval samples[-5], samples[-4], samples[-3], samples[-2], samples[-1]
            Y[3].push_back(
                train->fd_weights[2][Y[0].size()-2][0]*Y[0][Y[0].size()-5]
                + train->fd_weights[2][Y[0].size()-2][1]*Y[0][Y[0].size()-4]
                + train->fd_weights[2][Y[0].size()-2][2]*Y[0][Y[0].size()-3]
                + train->fd_weights[2][Y[0].size()-2][3]*Y[0][Y[0].size()-2]
                + train->fd_weights[2][Y[0].size()-2][4]*Y[0][Y[0].size()-1] 
            );

            // last interval samples[-4], samples[-3], samples[-2], samples[-1]
            Y[3].push_back(
                train->fd_weights[2][Y[0].size()-1][0]*Y[0][Y[0].size()-4]
                + train->fd_weights[2][Y[0].size()-1][1]*Y[0][Y[0].size()-3]
                + train->fd_weights[2][Y[0].size()-1][2]*Y[0][Y[0].size()-2]
                + train->fd_weights[2][Y[0].size()-1][3]*Y[0][Y[0].size()-1] 
            );
        }

    } else { 
        
        // 20% training samples
        
        if(meme::MAX_DER_ORD>=1)    Y.push_back({});
        if(meme::MAX_DER_ORD>=2)    Y.push_back({});
        if(meme::MAX_DER_ORD>=3)    cout << "";

        double ypred_after;
        double ypred_before;
        for(size_t i : selected) {

            // 0th order derivative
            Y[0].push_back(model->evaluate(train->samples[i]));
        
            // 1st order derivative
            if(meme::MAX_DER_ORD>=1) {
                
                // first interval samples[0], samples[1]
                if(i==0) {
                    ypred_after = model->evaluate(train->samples[1]);
                    Y[1].push_back( train->fd_weights[0][0][0]*Y[0][Y[0].size()-1] + train->fd_weights[0][0][1]*ypred_after );
                }

                // last interval samples[-2], samples[-1]
                if(i==train->samples.size()-1) {
                    ypred_before = model->evaluate(train->samples[train->samples.size()-2]);
                    Y[1].push_back( train->fd_weights[0][train->samples.size()-1][0]*ypred_before + train->fd_weights[0][train->samples.size()-1][1]*Y[0][Y[0].size()-1] );
                }
                
                // intermediate intervals samples[i-1], samples[i], samples[i+1]
                if(i>0 && i<train->samples.size()-1) {
                    ypred_before = model->evaluate(train->samples[i-1]);
                    ypred_after = model->evaluate(train->samples[i+1]);
                    Y[1].push_back( train->fd_weights[0][i][0]*ypred_before + train->fd_weights[0][i][1]*Y[0][Y[0].size()-1] + train->fd_weights[0][i][2]*ypred_after );
                }
            }
        
            // 2nd order derivative
            if(meme::MAX_DER_ORD>=2) {

                double ypred_after_2;
                double ypred_before_2;

                // first interval samples[0], samples[1], samples[2]
                if(i==0) {
                    ypred_after = model->evaluate(train->samples[1]);
                    ypred_after_2 = model->evaluate(train->samples[2]);
                    Y[2].push_back( train->fd_weights[1][0][0]*Y[0][Y[0].size()-1] + train->fd_weights[1][0][1]*ypred_after + train->fd_weights[1][0][2]*ypred_after_2 );
                }

                // second interval samples[0], samples[1], samples[2], samples[3]
                if(i==1) {
                    ypred_before = model->evaluate(train->samples[0]);
                    ypred_after = model->evaluate(train->samples[2]);
                    ypred_after_2 = model->evaluate(train->samples[3]);
                    Y[2].push_back( train->fd_weights[1][1][0]*ypred_before + train->fd_weights[1][1][1]*Y[0][Y[0].size()-1] + train->fd_weights[1][1][2]*ypred_after  + train->fd_weights[1][1][3]*ypred_after_2);
                }
                
                // one before last interval samples[-4], samples[-3], samples[-2], samples[-1]
                if(i==train->samples.size()-2) {
                    ypred_before_2 = model->evaluate(train->samples[train->samples.size()-4]);
                    ypred_before = model->evaluate(train->samples[train->samples.size()-3]);
                    ypred_after = model->evaluate(train->samples[train->samples.size()-1]);
                    Y[2].push_back( train->fd_weights[1][train->samples.size()-2][0]*ypred_before_2 + train->fd_weights[1][train->samples.size()-2][1]*ypred_before + train->fd_weights[1][train->samples.size()-2][2]*Y[0][Y[0].size()-1] + train->fd_weights[1][train->samples.size()-2][3]*ypred_after );
                }

                // last interval samples[-3], samples[-2], samples[-1]
                if(i==train->samples.size()-1) {
                    ypred_before_2 = model->evaluate(train->samples[train->samples.size()-3]);
                    ypred_before = model->evaluate(train->samples[train->samples.size()-2]);
                    Y[2].push_back( train->fd_weights[1][train->samples.size()-1][0]*ypred_before_2 + train->fd_weights[1][train->samples.size()-1][1]*ypred_before + train->fd_weights[1][train->samples.size()-1][2]*Y[0][Y[0].size()-1] );
                }

                // intermediate intervals
                if(i>1 && i<train->samples.size()-2) {
                    ypred_before_2 = model->evaluate(train->samples[i-2]);
                    ypred_before = model->evaluate(train->samples[i-1]);
                    ypred_after = model->evaluate(train->samples[i+1]);
                    ypred_after_2 = model->evaluate(train->samples[i+2]);
                    Y[2].push_back( train->fd_weights[1][i][0]*ypred_before_2 + train->fd_weights[1][i][1]*ypred_before + train->fd_weights[1][i][2]*Y[0][Y[0].size()-1] + train->fd_weights[1][i][3]*ypred_after + train->fd_weights[1][i][4]*ypred_after_2 );
                }
            }

            // 3rd order derivative
            if(meme::MAX_DER_ORD>=3) {
                
                double ypred_after_2;
                double ypred_before_2;
                double ypred_after_3;
                double ypred_before_3;

                // first interval samples[0], samples[1], samples[2], samples[3]
                if(i==0) {
                    ypred_after = model->evaluate(train->samples[1]);
                    ypred_after_2 = model->evaluate(train->samples[2]);
                    ypred_after_3 = model->evaluate(train->samples[3]);
                    Y[3].push_back( train->fd_weights[2][0][0]*Y[0][Y[0].size()-1] + train->fd_weights[2][0][1]*ypred_after + train->fd_weights[2][0][2]*ypred_after_2 + train->fd_weights[2][0][3]*ypred_after_3 );
                }

                // second interval samples[0], samples[1], samples[2], samples[3], samples[4]
                if(i==1) {
                    ypred_before = model->evaluate(train->samples[0]);
                    ypred_after = model->evaluate(train->samples[2]);
                    ypred_after_2 = model->evaluate(train->samples[3]);
                    ypred_after_3 = model->evaluate(train->samples[4]);
                    Y[3].push_back( train->fd_weights[2][1][0]*ypred_before + train->fd_weights[2][1][1]*Y[0][Y[0].size()-1] + train->fd_weights[2][1][2]*ypred_after  + train->fd_weights[2][1][3]*ypred_after_2  + train->fd_weights[2][1][4]*ypred_after_3 );
                }

                // third interval  samples[0], samples[1], samples[2], samples[3], samples[4], samples[5]
                if(i==2) {			
                    ypred_before_2 = model->evaluate(train->samples[0]);
                    ypred_before = model->evaluate(train->samples[1]);
                    ypred_after = model->evaluate(train->samples[3]);
                    ypred_after_2 = model->evaluate(train->samples[4]);
                    ypred_after_3 = model->evaluate(train->samples[5]);
                    Y[3].push_back( train->fd_weights[2][2][0]*ypred_before_2 + train->fd_weights[2][2][1]*ypred_before + train->fd_weights[2][2][2]*Y[0][Y[0].size()-1] + train->fd_weights[2][2][3]*ypred_after + train->fd_weights[2][2][4]*ypred_after_2 + train->fd_weights[2][2][5]*ypred_after_3 );
                }
                // two before last interval samples[-6], samples[-5], samples[-4], samples[-3], samples[-2], samples[-1]
                    if(i==train->samples.size()-3) {
                    ypred_before_3 = model->evaluate(train->samples[train->samples.size()-6]);
                    ypred_before_2 = model->evaluate(train->samples[train->samples.size()-5]);
                    ypred_before = model->evaluate(train->samples[train->samples.size()-4]);
                    ypred_after = model->evaluate(train->samples[train->samples.size()-2]);
                    ypred_after_2 = model->evaluate(train->samples[train->samples.size()-1]);
                    Y[3].push_back( train->fd_weights[2][train->samples.size()-3][0]*ypred_before_3 + train->fd_weights[2][train->samples.size()-3][1]*ypred_before_2 + train->fd_weights[2][train->samples.size()-3][2]*ypred_before + train->fd_weights[2][train->samples.size()-3][3]*Y[0][Y[0].size()-1] + train->fd_weights[2][train->samples.size()-3][4]*ypred_after + train->fd_weights[2][train->samples.size()-3][5]*ypred_after_2 );
                }

                // one before last interval samples[-5], samples[-4], samples[-3], samples[-2], samples[-1]
                if(i==train->samples.size()-2) {
                    ypred_before_3 = model->evaluate(train->samples[train->samples.size()-5]);
                    ypred_before_2 = model->evaluate(train->samples[train->samples.size()-4]);
                    ypred_before = model->evaluate(train->samples[train->samples.size()-3]);
                    ypred_after = model->evaluate(train->samples[train->samples.size()-1]);
                    Y[3].push_back( train->fd_weights[2][train->samples.size()-2][0]*ypred_before_3 + train->fd_weights[2][train->samples.size()-2][1]*ypred_before_2 + train->fd_weights[2][train->samples.size()-2][2]*ypred_before + train->fd_weights[2][train->samples.size()-2][3]*Y[0][Y[0].size()-1] + train->fd_weights[2][train->samples.size()-2][4]*ypred_after );
                }

                // last interval samples[-4], samples[-3], samples[-2], samples[-1]
                if(i==train->samples.size()-1) {
                    ypred_before_3 = model->evaluate(train->samples[train->samples.size()-4]);
                    ypred_before_2 = model->evaluate(train->samples[train->samples.size()-3]);
                    ypred_before = model->evaluate(train->samples[train->samples.size()-2]);
                    Y[3].push_back( train->fd_weights[2][train->samples.size()-1][0]*ypred_before_3 + train->fd_weights[2][train->samples.size()-1][1]*ypred_before_2 + train->fd_weights[2][train->samples.size()-1][2]*ypred_before + train->fd_weights[2][train->samples.size()-1][3]*Y[0][Y[0].size()-1] );
                }

                // intermediate intervals
                if(i>2 && i<train->samples.size()-3) {
                    ypred_before_3 = model->evaluate(train->samples[i-3]);
                    ypred_before_2 = model->evaluate(train->samples[i-2]);
                    ypred_before = model->evaluate(train->samples[i-1]);
                    ypred_after = model->evaluate(train->samples[i+1]);
                    ypred_after_2 = model->evaluate(train->samples[i+2]);
                    ypred_after_3 = model->evaluate(train->samples[i+3]);
                    Y[3].push_back( train->fd_weights[2][i][0]*ypred_before_3 + train->fd_weights[2][i][1]*ypred_before_2 + train->fd_weights[2][i][2]*ypred_before + train->fd_weights[2][i][3]*Y[0][Y[0].size()-1] + train->fd_weights[2][i][4]*ypred_after + train->fd_weights[2][i][5]*ypred_after_2 + train->fd_weights[2][i][6]*ypred_after_3 );
                }
            }
        }
    }
	    
    return Y;
}

template <class U>
vector<vector<double>> objective::fornberg2(MemeticModel<U>* model, DataSet* train, vector<size_t>& selected) {
    
    vector<vector<double>> Y{{}};

    const unsigned max_deriv = MAX_DER_ORD;
    vector<string> labels {"0th derivative", "1st derivative", "2nd derivative", "3rd derivative"};

    vector<double> x;
    for(size_t i = 0; i < train->y.size(); i++)
        x.push_back(train->y[i]);

    vector<vector<double>> test = fornberg(model, train, selected);
    auto coeffs = finitediff::generate_weights(x, max_deriv);

    for (unsigned deriv_i = 0; deriv_i <= max_deriv; deriv_i++) {
        cout << labels[deriv_i] << ": ";
        for (unsigned idx = 0; idx < x.size(); idx++){
            cout << coeffs[deriv_i*x.size() + idx] << " ";
        }
        cout << endl;
    }

    return Y;
}

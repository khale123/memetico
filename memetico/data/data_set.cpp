
/**
 * @file
 * @author andy@impv.au
 * @version 1.0
 * @brief Implementation of the DataSet class
*/

#include <memetico/data/data_set.h>

vector<string> DataSet::IVS;

void DataSet::load() {

    // Ensure we can open file
    //cout << "Loading " << filename << endl;
    ifstream f;
    f.open(filename);
    if (!f.is_open())
        throw runtime_error("Unable to open file "+ filename);

    // Load data
    bool is_first = true;
    string line;
    while ( f >> line ) { 

        if(is_first)    load_header(line);
        else            load_data(line);

        is_first = false;
    }

    if( get_gpu() )
        setup_gpu();

    if (meme::IN_DER == "app-multiV") {
        compute_app_der_multiV();
    }
}
// TODO: Generalize loading derivative information to multiple dimensions
void DataSet::load_header(string line) {
    
    stringstream ss(line);
    string part;

    DataSet::IVS.clear();

    size_t column = 0;
    while(ss.good()) {

        getline(ss, part, ',');    
        
        // Trim and remove line carrage if created from different os
        string word = trim(part);
        size_t index = word.find("\r", 0);
        if (index != string::npos)
            word.replace(index, 1, ""); 

        // Process element between commans
        if( word.compare("w") == 0 )                // If weight header
            weight_column = column;
        else if( word.compare("dy") == 0 )          // If uncertainty header
            uncertainty_column  = column;
        else if( word.compare("y") == 0 )           // If target header
            target_column = column;
        else if( word.compare("yd") == 0 ) {      // If derivative header
            derivative_column = column;
            if(meme::MAX_DER_ORD>=1) {
                Yder.push_back({});
                yder_min.push_back(0.0);
                yder_max.push_back(1.0);
            }
        }
        else if( word.compare("ydd") == 0 ) {     // If derivative header
            derivative2_column = column;
            if(meme::MAX_DER_ORD>=2) {
                Yder.push_back({});
                yder_min.push_back(0.0);
                yder_max.push_back(1.0);
            }
        }
        else if( word.compare("yddd") == 0 ) {     // If derivative header
            derivative3_column = column;
            if(meme::MAX_DER_ORD>=3) {
                Yder.push_back({});
                yder_min.push_back(0.0);
                yder_max.push_back(1.0);
            }
        }
        else                                    // else its a variable
            DataSet::IVS.push_back(word);    

        column++;
        
    }    
}

void DataSet::load_data(string line ) {

    stringstream ss(line);
    string part;

    size_t column = 0;
    vector<double> vars;
    while(ss.good()) {

        getline(ss, part, ',');    
        
        // Trim and remove line carrage if created from different os
        string word = trim(part);
        size_t index = word.find("\r", 0);
        if (index != string::npos)
            word.replace(index, 1, ""); 

        // Process element between commans
        if( column == weight_column )          
            weight.push_back(stod(word));
        else if( column == uncertainty_column )     
            dy.push_back(stod(word));
        else if( column == target_column )      
            y.push_back(stod(word));
        else if( column == derivative_column ) {
            if( meme::MAX_DER_ORD >= 1 )
                Yder[0].push_back(stod(word));
        }
        else if( column == derivative2_column ) {
            if( meme::MAX_DER_ORD >= 2 )
                Yder[1].push_back(stod(word));
        }
        else if( column == derivative3_column ) {
            if( meme::MAX_DER_ORD >= 3 )
                Yder[2].push_back(stod(word));
        }
        else           
            vars.push_back(stod(word));

        column++;
        
    } 
    samples.push_back(vars);

}

vector<size_t> DataSet::subset(float pct, bool to_GPU) {

    size_t ret_count = (long) (pct * get_count());
    vector<size_t> ret = RandInt::RANDINT->unique_set(ret_count, 0, get_count());

    if( gpu ) {

        // Free subset if already exists
        if(device_data.subset_size > 0) {
            freeSubset(&device_data);
            device_data.subset_size = 0;
        }

        // Copy new subset
        copySubset(&device_data, ret);
    }

    return ret;

}

/**
 * Output DataSet state
 * 
 * @param   out         Output stream to write to
 *                      Defaults to cout
 * 
 * @param   precision   resolution of real numerical output
 *                      Defaults to meme::PREC
 * 
 * @return          void
 */
/*
void DataSet::show(ostream& out, size_t precision) {

    size_t temp_precision = out.precision();
    out.precision(precision);

    size_t pad = precision+8;
    
    for(size_t i = 0; i < count; i++) {

        // Header
        if( i == 0) {
                
            out << setw(8) << "#" << setw(8) << "Sample" << setw(pad) << "y";
            for(size_t j = 0; j < ivs; j++)
                out << setw(pad) << names[j];
            
            if(has_uncertainty())
                out << setw(pad) << "dy";

            if(has_weight())
                out << setw(pad) << "weight";

            out << endl;
            
        }

        out << setw(8) << i+1;
        out << setw(8) << number[i];
        out << setw(pad) << y[i];

        for(size_t j = 0; j < ivs; j++)                    
            out << setw(pad) << variables[i][j];

        if(has_uncertainty())
            out << setw(pad) << dy[i];

        if(has_weight())
            out << setw(pad) << weight[i];
            
        out << endl;
    }

    cout << endl;
    out.precision(temp_precision);
    
}
*/

void DataSet::csv(string file_name) {

    // Extract the directory path
    path dir = path(file_name).parent_path();

    // Create directory if it doesn't exist
    if (!exists(dir)) {
        try {
            create_directories(dir); // This function creates all parent directories if they don't exist
        } catch (const filesystem_error& e) {
            throw runtime_error("Failed to create directory: " + string(e.what()));
        }
    }

    ofstream f;
    f.open(file_name);
    if (!f.is_open())
        throw runtime_error("Unable to open file "+ file_name);
    
    f << setprecision(meme::PREC);

    for(size_t i = 0; i < get_count(); i++) {

        // Header
        if( i == 0) {
                
            f << "y";
            for(size_t j = 0; j < DataSet::IVS.size(); j++)
                f << "," << DataSet::IVS[j];
            
            if(has_uncertainty())
                f << "," << "dy";

            if(has_weight())
                f << "," << "weight";

            f << endl;
        }

        f << y[i];

        for(size_t j = 0; j < DataSet::IVS.size(); j++)                    
            f << ","  << samples[i][j];

        if(has_uncertainty())
            f << ","  << dy[i];

        if(has_weight())
            f << ","  << weight[i];
            
        f << endl;
    }  
}

// Computes the approximate first and second order derivatives of the underlying function in the data using least squares finite difference method
void DataSet::compute_app_der_multiV() {
    compute_LS_FDS();
    for (size_t i = 0; i < samples.size(); ++i) {
        auto result = apply_FDS_on_data(i);
        app_grads.push_back(result.first);
        app_hesses.push_back(result.second);
    }
}

pair<vector<double>, vector<vector<double>>> DataSet::apply_FDS_on_data(size_t i) {
    // Construct constraints (f_diffs) for the linear system
    vector<size_t> neighbors = neighbor_indices[i];
    double response = y[i];
    vector<double> f_diffs;
    for (size_t& idx : neighbors) {
        double neighbor_response = y[idx];
        f_diffs.push_back(neighbor_response - response);
    }
    // Use Eigen to solve the linear system using the precomputed QR decomposition 
    Eigen::VectorXd coeff_eig = qr_decompositions[i].solve(Eigen::Map<Eigen::VectorXd>(f_diffs.data(), f_diffs.size())).eval();
    // Converts the Eigen vector to a standard vector
    std::vector<double> coeffs(coeff_eig.data(), coeff_eig.data() + coeff_eig.size());
    
    // This section also uses magic numbers extensively because it is not generalized to N dimensions
    // Construct gradient
    vector<double> gradient;
    if (coeffs.size() == 5) {
        // 2D case
        gradient.push_back(coeffs[0]);
        gradient.push_back(coeffs[1]);
    } else if (coeffs.size() == 9) {
        // 3D case
        gradient.push_back(coeffs[0]);
        gradient.push_back(coeffs[1]);
        gradient.push_back(coeffs[2]);
    } else {
        throw logic_error("Unsupported number of coefficients");
    }
    // Construct Hessian
    vector<vector<double>> hessian;
    if (coeffs.size() == 5) {
        // 2D case
        hessian = {
            {2*coeffs[2], coeffs[3]},
            {coeffs[3], 2*coeffs[4]}
        };
    } else if (coeffs.size() == 9) {
        // 3D case
        hessian = {
            {2*coeffs[3], coeffs[6], coeffs[7]},
            {coeffs[6], 2*coeffs[4], coeffs[8]},
            {coeffs[7], coeffs[8], 2*coeffs[5]}
        };
    } else {
        throw logic_error("Unsupported number of coefficients");
    }
    return make_pair(gradient, hessian);
}


pair<vector<double>, vector<vector<double>>> DataSet::apply_FDS_on_arbitrary_response(size_t i, vector<double> f_diffs) {
    // Construct constraints (f_diffs) for the linear system
    vector<size_t> neighbors = neighbor_indices[i];

    // Use Eigen to solve the linear system using the precomputed QR decomposition 
    Eigen::VectorXd coeff_eig = qr_decompositions[i].solve(Eigen::Map<Eigen::VectorXd>(f_diffs.data(), f_diffs.size())).eval();
    // Converts the Eigen vector to a standard vector
    std::vector<double> coeffs(coeff_eig.data(), coeff_eig.data() + coeff_eig.size());
    
    // This section also uses magic numbers extensively because it is not generalized to N dimensions
    // Construct gradient
    vector<double> gradient;
    if (coeffs.size() == 5) {
        // 2D case
        gradient.push_back(coeffs[0]);
        gradient.push_back(coeffs[1]);
    } else if (coeffs.size() == 9) {
        // 3D case
        gradient.push_back(coeffs[0]);
        gradient.push_back(coeffs[1]);
        gradient.push_back(coeffs[2]);
    } else {
        throw logic_error("Unsupported number of coefficients");
    }
    // Construct Hessian
    vector<vector<double>> hessian;
    if (coeffs.size() == 5) {
        // 2D case
        hessian = {
            {2*coeffs[2], coeffs[3]},
            {coeffs[3], 2*coeffs[4]}
        };
    } else if (coeffs.size() == 9) {
        // 3D case
        hessian = {
            {2*coeffs[3], coeffs[6], coeffs[7]},
            {coeffs[6], 2*coeffs[4], coeffs[8]},
            {coeffs[7], coeffs[8], 2*coeffs[5]}
        };
    } else {
        throw logic_error("Unsupported number of coefficients");
    }
    return make_pair(gradient, hessian);
}

// Designed to only be ran once per dataset
void DataSet::compute_LS_FDS() {
    compute_nearest_neighbors();
    size_t dim = samples[0].size();
    if (dim == 2) {
        for (size_t i = 0; i < samples.size(); ++i) {
            Eigen::MatrixXd A = set_up_linear_system_quad_2D(i);
            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
            qr_decompositions.push_back(qr);
        }
    }
    else if (dim == 3) {
        for (size_t i = 0; i < samples.size(); ++i) {
            Eigen::MatrixXd A = set_up_linear_system_quad_3D(i);
            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
            qr_decompositions.push_back(qr);
        }
    }
    else {
        throw logic_error("LS Derivative Approximation only supports 2D and 3D data");
    }
}



void DataSet::compute_nearest_neighbors(){
    size_t num_features = samples[0].size();
    neighbor_indices.resize(samples.size());
    // Loop to perform KNN on each sample
    for (size_t j = 0; j < samples.size(); ++j) {
        vector<pair<double, size_t>> distances;
        // Loop that implements KNN
        for (size_t i = 0; i < samples.size(); ++i) {
            if (j == i) continue;
            double dist = euclidean_distance(samples[j], samples[i]);
            distances.push_back(make_pair(dist, i));
        }
        // Partially sort distances to get the k smallest elements
        size_t num_neighbors = std::min(distances.size(), static_cast<size_t>(meme::NUM_NEIGHBORS));
        if (num_neighbors > 0 && distances.size() > 0) {
            std::nth_element(distances.begin(), distances.begin() + num_neighbors, distances.end());
        }
        // Store the indices of the k nearest neighbors
        for (size_t k = 0; k < num_neighbors; ++k) {
            neighbor_indices[j].push_back(distances[k].second);
        }
    }
}

// Takes a sample index and returns the data matrix for fitting a 2D quadratic function to it
Eigen::MatrixXd DataSet::set_up_linear_system_quad_2D(size_t j) {
    vector<double> sample = samples[j]; 
    vector<size_t> neighbors = neighbor_indices[j];
    // 5 comes from dx, dy, dx^2, dy^2, dx*dy
    // A future implementation goal is to general to N dimensions
    Eigen::MatrixXd A(neighbors.size(), 5);
    for (size_t i = 0; i < neighbors.size(); ++i) {
        size_t idx = neighbors[i];
        vector<double> neighbor = samples[idx];
        double dx = neighbor[0] - sample[0];
        double dy = neighbor[1] - sample[1];
        // Fill the matrix with the appropriate values
        // Code can be generalized to N dimensions which would avoid the magic numbers
        A(i, 0) = dx;
        A(i, 1) = dy;
        A(i, 2) = dx * dx;
        A(i, 3) = dx * dy;
        A(i, 4) = dy * dy;
    }
    return A;
}

// Takes a sample index and returns the data matrix for fitting a 3D quadratic function to it
Eigen::MatrixXd DataSet::set_up_linear_system_quad_3D(size_t j) {
    vector<double> sample = samples[j]; 
    vector<size_t> neighbors = neighbor_indices[j];
    // 9 comes from dx, dy, dz, dx^2, dy^2, dz^2, dx*dy, dx*dz, dy*dz
    // A future implementation goal is to general to N dimensions
    Eigen::MatrixXd A(neighbors.size(), 9);
    for (size_t i = 0; i < neighbors.size(); ++i) {
        size_t idx = neighbors[i];
        vector<double> neighbor = samples[idx];
        double dx = neighbor[0] - sample[0];
        double dy = neighbor[1] - sample[1];
        double dz = neighbor[2] - sample[2];
        // Fill the matrix with the appropriate values
        // Code can be generalized to N dimensions which would avoid the magic numbers
        A(i, 0) = dx;
        A(i, 1) = dy;
        A(i, 2) = dz;
        A(i, 3) = dx * dx;
        A(i, 4) = dy * dy;
        A(i, 5) = dz * dz;
        A(i, 6) = dx * dy;
        A(i, 7) = dx * dz;
        A(i, 8) = dy * dz;
    }
    return A;
}




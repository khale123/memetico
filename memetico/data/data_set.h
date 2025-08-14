
/**
 * @file
 * @author andy@impv.au
 * @version 1.0
 * @brief Header for the DataSet class
 */

#ifndef MEMETICO_DATA_SET_H
#define MEMETICO_DATA_SET_H
// Std
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <memetico/gpu/gpu_dataset.h>
// temp if for testing
#include <memetico/gpu/cuda.cuh>
#include <memetico/helpers/rng.h>
#include <memetico/helpers/text.h>
#include <memetico/helpers/distance.h>
// Eigen
#include <Eigen/Dense>

using namespace cusr;
using namespace filesystem;

/**
 * Class representing a set of data samples
 */
class DataSet {

    public:

        /** Construct DataSet with file */
        DataSet(string file_name, bool do_gpu = false) { 
            filename = file_name; 
            gpu = do_gpu;
            target_column = -1;
            weight_column = -1;
            uncertainty_column = -1;
            derivative_column = -1;
            derivative2_column = -1;
            derivative3_column = -1;
        };

        /** @brief Free GPU data */
        ~DataSet() {
            if( gpu ) {
                if(device_data.subset_size > 0) {
                    freeSubset(&device_data);
                    device_data.subset_size = 0;
                }

                freeDataSetAndLabel(&device_data);
            }
        }

        /** Target values for samples */
        vector<double>          y;

        /** minimum y value */
        double y_min = 0.0;

        /** maximum y value */
        double y_max = 1.0;
        
        /** Independent variable values for samples */
        vector<vector<double>>  samples;

        /** Weight values for samples */
        vector<double>          weight;

        /** Uncertainty values for samples  */
        vector<double>          dy;

        /** Derivative target values for samples */
        vector<vector<double>> Yder;
        // Yder[0] 1st order derivative, Yder[1] 2nd order derivative and so on

        /** Minimum derivative value for the 0th, 1st, 2nd etc. */
        vector<double> yder_min;

        /** Maximum derivative value for the 0th, 1st, 2nd etc. */
        vector<double> yder_max;

        /** Finite Difference weights */
        vector<vector<vector<double>>> fd_weights;
        // fd_weights[0] stores weights for 1st order derivative
        // fd_weights[1] stores weights for 2nd order derivative
        // and so on...

        /** Names of the independent varaibles */
        static vector<string>   IVS;

        // For Multi-Variable Derivatives
        /** */
        vector<vector<double>> app_grads;
        
        /** */
        vector<vector<vector<double>>> app_hesses;
        
        /** */
        vector<Eigen::ColPivHouseholderQR<Eigen::MatrixXd>> qr_decompositions;

        /** */
        vector<vector<size_t>> neighbor_indices;

        /** @brief Return filename for DataSet */
        string get_file()       { return filename; };

        /** @brief Return number of samples */
        size_t get_count()      { return y.size(); };

        /** @brief Return if GPU is being used */
        bool get_gpu()          {return gpu;};

        /** @brief Return if all samples have weight */
        bool has_weight()       { return weight.size() == get_count();}

        /** @brief Return if all samples have uncertainty */
        bool has_uncertainty()  { return dy.size() == get_count();}

        /** @brief Return if all samples have derivatives  */    
        bool has_derivative()  { return Yder.size() > 0; }

        /** @brief Load data from filename into the object */
        void load();

        /** @brief Output DataSet to CSV */
        void csv(string file_name);

        /** @brief Get indexes for a percentage of the DataSet uniformly and at random */
        vector<size_t> subset(float pct, bool to_GPU = true);

        /**
         * Multivariable Derivative Approximation Methods
         */

        /**KIERAN DESCRIPTION */
        void compute_app_der_multiV();

        /** */
        pair<vector<double>, vector<vector<double>>> apply_FDS_on_data(size_t i);
        
        /** */
        void compute_LS_FDS();

        /** */
        void compute_nearest_neighbors();

        /** */
        Eigen::MatrixXd set_up_linear_system_quad_2D(size_t j);

        /** */
        Eigen::MatrixXd set_up_linear_system_quad_3D(size_t j);

        // GPU specific 

        /** @brief GPU dataset for use in cuda.cuh */
        GPUDataset device_data;

        /** @brief GPU initialisation */
        void setup_gpu() {

            // GPU must be configured 
            if(!gpu)    return;

            vector<vector<float>> float_samples;
            for(size_t i = 0; i < samples.size(); i++) {
                vector<float> temp;
                for(size_t j = 0; j < IVS.size(); j++) {
                    temp.push_back(samples[i][j]);
                }
                float_samples.push_back(temp);
            }

            vector<float> float_y;
            for(size_t i = 0; i < y.size(); i++)
                float_y.push_back(y[i]);

            vector<float> float_w;
            if( has_weight() ) {
                for(size_t i = 0; i < y.size(); i++)
                    float_w.push_back(weight[i]);
            }

            copyDatasetAndLabel(&device_data, float_samples, float_y, float_w); 
            device_data.subset_size = 0;
        }

        /** @brief Print the dataset */
        void print() {

            cout << "y" << endl;
            for(int i=0; i<y.size(); i++)
                cout << y[i] <<endl;

            cout << "y_min = " << y_min << endl;
            cout << "y_max = " << y_max << endl;
            cout << endl;

            cout << "x" << endl;
            for(int i=0; i<samples.size(); i++)
                cout << samples[i][0] <<endl;

            cout << endl;

            for(int k=0; k<Yder.size(); k++) {

                cout << "yder" << k+1 <<  endl;

                for(int i=0; i<Yder[k].size(); i++)     cout << Yder[k][i] <<endl;

                if(yder_min.size()>0)   cout << "yder_min[k] = "<< yder_min[k] << endl;
                if(yder_max.size()>0)   cout << "yder_max[k] = "<< yder_max[k] << endl;

                cout << endl;
            }
        }

        /** @brief Normalise the target and derivative data */
        void normalise() {
        
            // For target (0th order derivative)
            y_min = numeric_limits<double>::max();
            y_max = -y_min;
            for(int i=0; i<y.size(); i++) {

                if( y[i] > y_max )  y_max = y[i];
                if( y[i] < y_min )  y_min = y[i];

            } 
            for(int i=0; i<y.size(); i++)
                y[i] = ( y[i] - y_min )/( y_max - y_min );

            // Higher order derivatives
            double min;
            double max;
            yder_min.clear();
            yder_max.clear();

            // For each derivative
            for(int k=0; k<Yder.size(); k++) {

                min = numeric_limits<double>::max();
                max = -min;
                for(int i=0; i<Yder[k].size(); i++) {
                    if( Yder[k][i] > max )  max = Yder[k][i];
                    if( Yder[k][i] < min )  min = Yder[k][i];
                }

                for(int i=0; i<Yder[k].size(); i++)     Yder[k][i] = ( Yder[k][i] - min )/( max - min );

                yder_min.push_back(min);
                yder_max.push_back(max);
            }
        
        }

        /**
         * @brief Compute finite difference weights for Fornberg method
         * 
         * The function \f$f(x)\f$ is approximated at point \f$x_0\f$ using a set of points n+1 \f$\{x_i\}\f$ around \f$x_0\f$
         * This is done for any order derivative m.
         * 
         */
        void compute_app_der(size_t mdo) {

            // computes the approximate 1st order derivative
            // !!! WE HAVE TO NORMALISE AFTER calling this function
            
            Yder.clear();
            yder_min.clear();
            yder_max.clear();
            fd_weights.clear();
            vector<vector<double>> weights;

            // 1st derivatives
            if(mdo>=1) {

                Yder.push_back({});
                yder_min.push_back(0.0);
                yder_max.push_back(1.0);
                fd_weights.push_back({});

                // first interval [samples[0], samples[1]]
                weights = compute_FD_weights(1, {samples[0], samples[1]}, samples[0][0]);
                fd_weights[0].push_back(weights[1]);
                Yder[0].push_back( weights[1][0]*y[0] + weights[1][1]*y[1] );
                weights.clear();

                // intermediate intervals
                for(size_t i = 1; i < y.size()-1; i++){
                    weights = compute_FD_weights(1, {samples[i-1], samples[i], samples[i+1]}, samples[i][0]);
                    fd_weights[0].push_back(weights[1]);
                    Yder[0].push_back( weights[1][0]*y[i-1] + weights[1][1]*y[i] + weights[1][2]*y[i+1] );
                    weights.clear();
                }

                // last interval [samples[-2], samples[-1]]
                weights = compute_FD_weights(1, {samples[y.size()-2], samples[y.size()-1]}, samples[y.size()-1][0]);
                fd_weights[0].push_back(weights[1]);
                Yder[0].push_back( weights[1][0]*y[y.size()-2] + weights[1][1]*y[y.size()-1] );
                weights.clear();

            }


            // 2nd derivatives
            if(mdo>=2) {

                Yder.push_back({});
                yder_min.push_back(0.0);
                yder_max.push_back(1.0);
                fd_weights.push_back({});

                // first interval [samples[0], samples[1], samples[2]]
                weights = compute_FD_weights(2, {samples[0], samples[1], samples[2]}, samples[0][0]);
                fd_weights[1].push_back(weights[2]);
                Yder[1].push_back( weights[2][0]*y[0] + weights[2][1]*y[1] + weights[2][2]*y[2] );
                weights.clear();

                // second interval [samples[0], samples[1], samples[2], samples[3]]
                weights = compute_FD_weights(2, {samples[0], samples[1], samples[2], samples[3]}, samples[1][0]);
                fd_weights[1].push_back(weights[2]);
                Yder[1].push_back( weights[2][0]*y[0] + weights[2][1]*y[1] + weights[2][2]*y[2] + weights[2][3]*y[3] );
                weights.clear();

                // intermediate intervals
                for(size_t i = 2; i < y.size()-2; i++){
                    weights = compute_FD_weights(2, {samples[i-2], samples[i-1], samples[i], samples[i+1], samples[i+2]}, samples[i][0]);
                    fd_weights[1].push_back(weights[2]);
                    Yder[1].push_back( weights[2][0]*y[i-2] + weights[2][1]*y[i-1] + weights[2][2]*y[i] + weights[2][3]*y[i+1] + weights[2][4]*y[i+2] );
                    weights.clear();
                }

                // interval [samples[-4], samples[-3], samples[-2], samples[-1]]
                weights = compute_FD_weights(2, {samples[y.size()-4], samples[y.size()-3], samples[y.size()-2], samples[y.size()-1]}, samples[y.size()-2][0]);
                fd_weights[1].push_back(weights[2]);
                Yder[1].push_back( weights[2][0]*y[y.size()-4] + weights[2][1]*y[y.size()-3] + weights[2][2]*y[y.size()-2] + weights[2][3]*y[y.size()-1] );
                weights.clear();

                // interval [samples[-3], samples[-2], samples[-1]]
                weights = compute_FD_weights(2, {samples[y.size()-3], samples[y.size()-2], samples[y.size()-1]}, samples[y.size()-1][0]);
                fd_weights[1].push_back(weights[2]);
                Yder[1].push_back( weights[2][0]*y[y.size()-3] + weights[2][1]*y[y.size()-2] + weights[2][2]*y[y.size()-1] );
                weights.clear();

            }

            // 3rd derivatives
            if(mdo>=3) {

                Yder.push_back({});
                yder_min.push_back(0.0);
                yder_max.push_back(1.0);
                fd_weights.push_back({});

                // first interval samples[0], samples[1], samples[2], samples[3]
                weights = compute_FD_weights(3, {samples[0], samples[1], samples[2], samples[3]}, samples[0][0]);
                fd_weights[2].push_back(weights[3]);
                Yder[2].push_back( weights[3][0]*y[0] + weights[3][1]*y[1] + weights[3][2]*y[2] + weights[3][3]*y[3] );
                weights.clear();

                // second interval samples[0], samples[1], samples[2], samples[3], samples[4]
                weights = compute_FD_weights(3, {samples[0], samples[1], samples[2], samples[3], samples[4]}, samples[1][0]);
                fd_weights[2].push_back(weights[3]);
                Yder[2].push_back( weights[3][0]*y[0] + weights[3][1]*y[1] + weights[3][2]*y[2] + weights[3][3]*y[3] + weights[3][4]*y[4] );
                weights.clear();

                // third interval samples[0], samples[1], samples[2], samples[3], samples[4], samples[5]
                weights = compute_FD_weights(3, {samples[0], samples[1], samples[2], samples[3], samples[4], samples[5]}, samples[2][0]);
                fd_weights[2].push_back(weights[3]);
                Yder[2].push_back( weights[3][0]*y[0] + weights[3][1]*y[1] + weights[3][2]*y[2] + weights[3][3]*y[3] + weights[3][4]*y[4] + weights[3][5]*y[5] );
                weights.clear();

                // intermediate intervals
                for(size_t i = 3; i < y.size()-3; i++){
                    weights = compute_FD_weights(3, {samples[i-3], samples[i-2], samples[i-1], samples[i], samples[i+1], samples[i+2], samples[i+3]}, samples[i][0]);
                    fd_weights[2].push_back(weights[3]);
                    Yder[2].push_back( weights[3][0]*y[i-3] + weights[3][1]*y[i-2] + weights[3][2]*y[i-1] + weights[3][3]*y[i] + weights[3][4]*y[i+1] + weights[3][5]*y[i+2] + weights[3][6]*y[i+3] );
                    weights.clear();
                }

                // interval samples[-6], samples[-5], samples[-4], samples[-3], samples[-2], samples[-1]
                weights = compute_FD_weights(3, {samples[y.size()-6], samples[y.size()-5], samples[y.size()-4], samples[y.size()-3], samples[y.size()-2], samples[y.size()-1]}, samples[y.size()-3][0]);
                fd_weights[2].push_back(weights[3]);
                Yder[2].push_back( weights[3][0]*y[y.size()-6] + weights[3][1]*y[y.size()-5] + weights[3][2]*y[y.size()-4] + weights[3][3]*y[y.size()-3] + weights[3][4]*y[y.size()-2] + weights[3][5]*y[y.size()-1 ]);
                weights.clear();

                // interval samples[-5], samples[-4], samples[-3], samples[-2], samples[-1]
                weights = compute_FD_weights(3, {samples[y.size()-5], samples[y.size()-4], samples[y.size()-3], samples[y.size()-2], samples[y.size()-1]}, samples[y.size()-2][0]);
                fd_weights[2].push_back(weights[3]);
                Yder[2].push_back( weights[3][0]*y[y.size()-5] + weights[3][1]*y[y.size()-4] + weights[3][2]*y[y.size()-3] + weights[3][3]*y[y.size()-2] + weights[3][4]*y[y.size()-1] );
                weights.clear();

                // interval samples[-4], samples[-3], samples[-2], samples[-1]
                weights = compute_FD_weights(3, {samples[y.size()-4], samples[y.size()-3], samples[y.size()-2], samples[y.size()-1]}, samples[y.size()-1][0]);
                fd_weights[2].push_back(weights[3]);
                Yder[2].push_back( weights[3][0]*y[y.size()-4] + weights[3][1]*y[y.size()-3] + weights[3][2]*y[y.size()-2] + weights[3][3]*y[y.size()-1] );
                weights.clear();

            }

        }
        
        /**
         * @brief Compute finite difference weights for Fornberg method
         * 
         * In the Fornberg method we need to compute the weights \f$w_i\f$ for each point in \f$ \{x_i\} \f$
         * To approximate the derivative of  \f$f(x)\f$ at \f$x_0\f$ using the weighted sum:
         * 
         * \f$f^{m}(x_0) \approx \sum_limits_{i=0}^n w_i\cdot f(x_i) \f$
         * 
         * @param mdo derivative or m value
         * @param samples set of points for determining the derivative.
         * These will be a subset of the full dataset \f$\mathbf{x}\f$ centered around \f$x_0\f$
         * @param around the \f$x\f$ value to compute the derivative around
         * 
         */
        static vector<vector<double>> compute_FD_weights(const unsigned mdo, vector<vector<double>> samples, const double around) {
  
            // Ensure we have the minimum sample counts
            vector<vector<double>> weights;
            if (samples.size() < mdo + 1){
                cout << "[data_set.h] size of samples insufficient" << endl;
                throw std::logic_error("size of samples insufficient");
            }

            // Prepare for the n weights that will be summed for the approximate derivative
            // For each derivative
            for( unsigned i = 0; i <= mdo; i++ ) {

                // Create a weights array and initialise with 0 value
                weights.push_back({});
                for( unsigned j = 0; j < samples.size(); j++ ) {
                    weights[i].push_back(0.0);
                }
            }
        
            // Set the weight of the first derivatives first sample to 1 !!! not clear why
            weights[0][0] = 1;

            // compute the weights
            double 
                scaleFactor,        // Initially 1, updated with each iteration to reflect cumulative scaling based on point differences
                prodDiff,           // Product of differences between the current and all previous points
                invProdDiff,        // Inverse of prodDiff, used to normalize weight calculations
                pointDiff,          // Difference between two sample points being considered
                invPointDiff,       // Inverse of pointDiff, for scaling weights inversely to point distances.
                curDist,            // Distance from the current sample point to the target point.
                prevDist;           // Distance from the previous sample point to the target, carried over from the last iteration.

            int mn;
            scaleFactor = 1;
            curDist = samples[0][0] - around;

            // For each sample in the subset of x
            for (unsigned i=1; i < samples.size(); ++i) {

                mn = min(i, mdo);       // get at most the derivative number or points
                                        // and if we are on early samples, we get fewer

                prodDiff = 1;
                prevDist = curDist;
                curDist = samples[i][0] - around;

                // For the samples before the current sample
                for (unsigned j=0; j<i; ++j) {

                    // Determine the difference between the current point and the previous point being processed
                    pointDiff = samples[i][0] - samples[j][0];
                    invPointDiff = 1/pointDiff;

                    // Multiply the differences of each point as we go through them
                    prodDiff = prodDiff*pointDiff;

                    // If its the last point before the current
                    if (j == i-1) {

                        invProdDiff = 1/prodDiff;

                        // For the number of points we have, or the number of points for the derivative order should we have that many
                        for (int k=mn; k>=1; --k)

                            // Set the weight for the current sample for the kth derivative 
                            // to some hideous thing that is hard to interpret the meaning of
                            // - With k*weights[k-1][i-1] we are looking at the previous derivative
                            // - At the start scaleFactor us 1, but this changes after processing each sample
                            // - 

                            weights[k][i] = scaleFactor*(k*weights[k-1][i-1] - prevDist*weights[k][i-1])*invProdDiff;

                        // For the 1st derivative, 
                        weights[0][i] = -scaleFactor*prevDist*weights[0][i-1]*invProdDiff;
                    }

                    for (unsigned k=mn; k>=1; --k)
                        weights[k][j] = (curDist*weights[k][j] - k*weights[k-1][j])*invPointDiff;

                    weights[0][j] = curDist*weights[0][j]*invPointDiff;
                }
                scaleFactor = prodDiff;
            }

            return weights;
            
        }
        

        /**
         */
        static vector<vector<double>> compute_FD_weights2(const unsigned mdo, vector<vector<double>> samples, const double around) {
  
            /*
            // Example grid and function values
            vector<double> x = {0, 1, 2, 3, 4}; // Grid points
            vector<double> y = {0, 1, 4, 9, 16}; // Function values at grid points, y = x^2

            // Maximum derivative order to compute (k derivatives)
            unsigned max_deriv = 2; // Change this value to compute higher derivatives

            // Compute and apply weights for each point, adjusting the subset size based on max_deriv
            for (size_t point = 0; point < x.size(); ++point) {
                // Calculate the range of points to use
                int start = max(0, static_cast<int>(point) - max_deriv);
                int end = min(static_cast<int>(x.size()) - 1, static_cast<int>(point) + max_deriv);

                // Subset of points and function values
                vector<double> subset_x(x.begin() + start, x.begin() + end + 1);
                vector<double> subset_y(y.begin() + start, y.begin() + end + 1);

                // Compute weights for the subset
                vector<double> weights(subset_x.size() * (max_deriv + 1));
                calculate_weights<double>(&subset_x[0], subset_x.size(), max_deriv, &weights[0], x[point]);

                // Apply the weights to compute the derivatives up to max_deriv
                cout << "Derivatives at x = " << x[point] << ":\n";
                for (unsigned deriv = 0; deriv <= max_deriv; ++deriv) {
                    double derivative = 0.0;
                    for (size_t i = 0; i < subset_x.size(); ++i) {
                        derivative += weights[i + deriv * subset_x.size()] * subset_y[i];
                    }
                    cout << "  Order " << deriv << " derivative is approximately: " << derivative << endl;
                }
            }
            */
            vector<vector<double>> weights;
            return weights;
        }










    private: 

        /** Load first row of file */
        void load_header(string s);

        /** Load data line in file */
        void load_data(string s);

        /** Location of file used to generate dataset */
        string          filename;

        /** Weight column */
        int             weight_column;

        /** Uncertainty column */
        int             uncertainty_column;

        /** Derivative target column */
        int             derivative_column;

        /** Derivative target column */
        int             derivative2_column;

        /** Derivative target column */
        int             derivative3_column;
        
        /** Target column */
        int             target_column;

        /** Flag for GPU operation */
        bool            gpu;

};

#endif

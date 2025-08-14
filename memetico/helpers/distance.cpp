/**
 * KIERAN HEADER
 */

#include <memetico/helpers/distance.h>

// Assumes that size of a and b are the same

double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
        double dist = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            double diff = a[i] - b[i];
            dist += diff*diff;
        }
        return sqrt(dist);
}

double frobenius_metric_matrix(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
    double dist = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[i].size(); ++j) {
            double diff = a[i][j] - b[i][j];
            dist += diff * diff;
        }
    }
    return sqrt(dist);
}    
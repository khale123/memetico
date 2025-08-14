/**
 * KIERAN HEADER
 */

#ifndef MEMETICO_HELPER_DISTANCE_H_
#define MEMETICO_HELPER_DISTANCE_H_

#include <cstdlib>
#include <cmath>
#include <vector>
#include <Eigen/Dense>


double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b);

double frobenius_metric_matrix(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b);


#endif
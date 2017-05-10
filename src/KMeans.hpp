#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Core>

struct Statistics
{
    size_t numIterations;
    double totalError;
};

class KMeans
{
public:
    // K = number of clusters, d = number of dimensions in feature vectors,
    // threshold = convergence threshold for error of an individual centroid,
    // maxIterations = max number of iterations to run
    KMeans(size_t K, size_t d, double threshold, double maxIterations);
    
    // runs KMeans on the points (feature vectors) passed in
    // NOTE: The user must maintain the points vector for use before and after
    // the run function. Other getter functions (below) refer to the ordering 
    // used in the points vector
    void run(const std::vector<Eigen::VectorXd> &points);
    
    // returns multimap of cluster indexes to indexes of points in vector of 
    // points passed into the run function.
    // User can use this map to display cluster membership in any way
    const std::multimap<size_t, size_t>& clusters() const { return clusters_; }
    
    // returns the centroids of the K clusters
    const std::vector<Eigen::VectorXd>& centroids() const { return centroids_; }

    // returns statistics for the algorithm's performance
    Statistics stats() const { return stats_; }

private:
    double threshold_;
    size_t maxIterations_;
    Statistics stats_; 

    std::multimap<size_t, size_t> clusters_;
    std::vector<Eigen::VectorXd> centroids_;
    std::vector<Eigen::VectorXd> prevCentroids_;

    double dist(const Eigen::VectorXd &a, const Eigen::VectorXd &b);
    bool converged();
    void assignPoint(size_t p, const std::vector<Eigen::VectorXd> &points);
    void calcCentroid(size_t c, const std::vector<Eigen::VectorXd> &points);
};

#endif /* KMEANS_HPP */

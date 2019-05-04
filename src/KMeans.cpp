#include "KMeans.hpp"
#include <limits>

using std::make_pair;
using std::vector;
using std::multimap;
using Eigen::VectorXd;

KMeans::KMeans(size_t K, size_t d, double threshold, size_t maxIterations)
    : threshold_(threshold), maxIterations_(maxIterations)
{
    prevCentroids_.resize(K);
    centroids_.resize(K);
    
    for (size_t i = 0; i < K; ++i) 
		prevCentroids_[i] = centroids_[i] = VectorXd::Zero(d);
	
    clusters_.clear();
	
    stats_.numIterations = 0;
    stats_.totalError = 0;
} 

void KMeans::run(const std::vector<Eigen::VectorXd> &points)
{
    while (stats_.numIterations < maxIterations_)
    {
        clusters_.clear(); // reset clusters
        
        for (size_t p = 0; p < points.size(); ++p) assignPoint(p, points);

        for (size_t c = 0; c < centroids_.size(); ++c) calcCentroid(c, points);
        
        ++stats_.numIterations;
        if (converged()) break;
    }
}

void KMeans::assignPoint(size_t p, const vector<VectorXd> &points)
{
    size_t clusterId = 0;
    double minDist = std::numeric_limits<double>::max();
    
    for (size_t i = 0; i < centroids_.size(); ++i)
    {
        double d = dist(points[p], centroids_[i]);
        if (d < minDist)
        {
            minDist = d;
            clusterId = i;
        }
    }

    clusters_.insert(make_pair(clusterId, p));
}

void KMeans::calcCentroid(size_t c, const vector<VectorXd> &points)
{
    prevCentroids_[c] = centroids_[c];

    centroids_[c].setZero();

    if (clusters_.count(c) == 0) return;

    auto range = clusters_.equal_range(c);
    for (auto it = range.first; it != range.second; ++it)
        centroids_[c] += points[it->second];
    
    centroids_[c] /= (double)clusters_.count(c);
}

double KMeans::dist(const VectorXd &a, const VectorXd &b)
{
    VectorXd c = a - b;
    return c.norm();
}

bool KMeans::converged()
{
    stats_.totalError = 0;
    for (size_t i = 0; i < centroids_.size(); ++i)
        stats_.totalError += dist(centroids_[i], prevCentroids_[i]);
    
    return stats_.totalError <= ((double)centroids_.size() * threshold_);
}

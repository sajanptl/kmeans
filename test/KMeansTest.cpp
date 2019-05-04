#define BOOST_TEST_MODULE "KmeansTest"
#include <boost/test/unit_test.hpp>

#include "KMeans.hpp"

#include <iostream>
#include <map>
#include <vector>
#include <Eigen/Dense>

using std::cout;
using std::endl;
using std::vector;
using std::multimap;
using Eigen::VectorXd;
using Eigen::Vector2d;

struct Fixture
{
	Fixture() 
		: k(2), niters(100), thresh(0.1), alg(KMeans(2, 2, 0.1, 100))
	{
		points.resize(10, VectorXd::Zero(2));

		// cluster 1, mean [0, 0], 5 members
		points[0] = Vector2d(0.0, 0.0);
		points[1] = Vector2d(1.0, 0.0);
		points[2] = Vector2d(0.0, 1.0);
		points[3] = Vector2d(-1.0, 0.0);
		points[4] = Vector2d(0.0, -1.0);
		
		// cluster 2, mean [10, 10], 5 members
		points[5] = Vector2d(10.0, 10.0);
		points[6] = Vector2d(9.0, 10.0);
		points[7] = Vector2d(10.0, 9.0);
		points[8] = Vector2d(11.0, 10.0);
		points[9] = Vector2d(10.0, 11.0);
	}

	size_t k;
	size_t niters;
	double thresh;
	KMeans alg;
	vector<VectorXd> points;
};

BOOST_AUTO_TEST_CASE(initTest)
{
	Fixture f;
	f.alg.run(f.points);
	auto stats = f.alg.stats();
	auto clusters = f.alg.clusters();
	auto centroids = f.alg.centroids(); 
	BOOST_CHECK(stats.numIterations <= f.niters);
	BOOST_CHECK(stats.totalError <= static_cast<double>(centroids.size()) * f.thresh);
	BOOST_CHECK_EQUAL(centroids.size(), f.k);
	BOOST_CHECK_EQUAL(clusters.size(), f.points.size());
	
	cout << "Centroids" << endl;
	for (const auto &c : centroids) cout << c.transpose() << endl;

	cout << "Cluster ID, Point" << endl;
	for (const auto &p : clusters) 
		cout << p.first << " " << f.points[p.second].transpose() << endl;
	cout << "Niters Taken: " << stats.numIterations << endl;
	cout << "Total Error: " << stats.totalError << endl;
}

#include "opencv2/core/core.hpp"

const double DOT_M1 = 1.20694;
const double DOT_M1_STD = 0.0681875;
const double DOT_M3 = 0.000161952;
const double DOT_M3_STD = 0.000199846;
const double DOT_M7 = 0.349098;
const double DOT_M7_STD = 0.0487571;

const double P_M1 = 0.723017;
const double P_M1_STD = 0.0377325;
const double P_M3 = 0.00325343;
const double P_M3_STD = 0.00128721;
const double P_M7 = 0.120465;
const double P_M7_STD = 0.018598;

const double K_M1 = 0.925492;
const double K_M1_STD = 0.0881796;
const double K_M3 = 0.047983;
const double K_M3_STD = 0.0196914;
const double K_M7 = 0.0913253;
const double K_M7_STD = 0.0117454;

const double O_M1 = 0.943521;
const double O_M1_STD = 0.192505;
const double O_M3 = 0.0294774;
const double O_M3_STD = 0.0180845;
const double O_M7 = 0.110737;
const double O_M7_STD = 0.0496763;

//const double ALL_M1 = 0.527759;
//const double ALL_M1_STD = 0.0228398;
//const double ALL_M2 = 0.0508169;
//const double ALL_M2_STD = 0.0233806;
//const double ALL_M3 = 0.0203999;
//const double ALL_M3_STD = 0.00527949;
//const double ALL_M7 = 0.0570556;
//const double ALL_M7_STD = 0.00165529;


int S(cv::Mat& I);

int L(cv::Mat& I);

double W3(cv::Mat& I);

double m(cv::Mat& I, int p, int q);

double M(cv::Mat& I, int p, int q);

double M_20(cv::Mat& I);

double M_02(cv::Mat& I);

double M_11(cv::Mat& I);

double M_03(cv::Mat& I);

double M_30(cv::Mat& I);

double M_21(cv::Mat& I);

double M_12(cv::Mat& I);

double M1(cv::Mat& I);

double M2(cv::Mat& I);

double M3(cv::Mat& I);

double M7(cv::Mat& I);

void calculate_features(const cv::Mat& I, bool oneObject);

double average(std::vector<double> const& v);

double std_dev(const std::vector<double>& v);

std::vector<BoundingBox> detect_logo(const cv::Mat& I, bool predicate(const HSV& hsv));

Point geo_center(BoundingBox box);

double euclidean_distance(const Point& p1, const Point& p2);

std::vector<int> DBSCAN(const std::vector<Point>& points, double eps, int minPts, int maxClusterSize);

std::vector<double> calculate_knn(const std::vector<Point>& points, int k);

#include <numeric>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <corecrt_math_defines.h>
#include "utils.hpp"
#include "features.hpp"
#include "processing.hpp"
#include <iostream>
#include "converter.hpp"
#include <map>


int S(cv::Mat& I) {

	int counter = 0;

	switch (I.channels()) {
	case 1:

		for (int i = 0; i < I.rows; i++)
			for (int j = 0; j < I.cols; j++)
				if (I.at<uchar>(i, j) != 255)
					counter++;
		break;
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;

		for (int i = 0; i < _I.rows; i++)
			for (int j = 0; j < _I.cols; j++)
				if (_I(i, j)[0] != 255 || _I(i, j)[1] != 255 || _I(i, j)[2] != 255)
					counter++;
		break;
	}


	return counter;
}


int L(cv::Mat& I) {

	int counter = 0;
	uchar channels_num = I.channels();

	if (channels_num == 1) {

		cv::Mat_<uchar> _I = I;

		for (int i = 1; i < _I.rows - 1; i++)
			for (int j = 1; j < _I.cols - 1; j++) {

				if (_I(i, j) != 255 &&
					(_I(i - 1, j - 1) != _I(i, j) ||
						_I(i - 1, j) != _I(i, j) ||
						_I(i - 1, j + 1) != _I(i, j) ||
						_I(i, j - 1) != _I(i, j) ||
						_I(i, j + 1) != _I(i, j) ||
						_I(i + 1, j - 1) != _I(i, j) ||
						_I(i + 1, j) != _I(i, j) ||
						_I(i + 1, j + 1) != _I(i, j))
					)
					counter++;
			}
	}
	else if (channels_num == 3) {

		cv::Mat_<cv::Vec3b> _I = I;

		for (int i = 1; i < _I.rows - 1; i++)
			for (int j = 1; j < _I.cols - 1; j++) {
				for (int channel = 0; channel < 3; channel++) {
					if (_I(i, j)[channel] != 255 &&
						(_I(i - 1, j - 1)[channel] != _I(i, j)[channel] ||
							_I(i - 1, j)[channel] != _I(i, j)[channel] ||
							_I(i - 1, j + 1)[channel] != _I(i, j)[channel] ||
							_I(i, j - 1)[channel] != _I(i, j)[channel] ||
							_I(i, j + 1)[channel] != _I(i, j)[channel] ||
							_I(i + 1, j - 1)[channel] != _I(i, j)[channel] ||
							_I(i + 1, j)[channel] != _I(i, j)[channel] ||
							_I(i + 1, j + 1)[channel] != _I(i, j)[channel])
						) {
						counter++;
						break;
					}
				}

			}
	}
	return counter;
}



double W3(cv::Mat& I) {

	return L(I) / (2 * sqrt(M_PI * S(I))) - 1;
}

double m(cv::Mat& I, int p, int q) {

	double result = 0.0;

	switch (I.channels()) {
	case 1:

		for (int i = 0; i < I.rows; i++)
			for (int j = 0; j < I.cols; j++) {
				if (I.at<uchar>(i, j) != 255)
					result += pow(i, p) * pow(j, q);
			}
		break;
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;

		for (int i = 0; i < I.rows; i++)
			for (int j = 0; j < I.cols; j++) {
				if (_I(i, j)[0] != 255 || _I(i, j)[1] != 255 || _I(i, j)[2] != 255)
					result += pow(i, p) * pow(j, q);
			}
		break;
	}

	return result;
}

double M(cv::Mat& I, int p, int q) {

	double result = 0.0;
	double m_10 = m(I, 1, 0);
	double m_01 = m(I, 0, 1);
	double m_00 = m(I, 0, 0);

	switch (I.channels()) {
	case 1:

		for (int i = 0; i < I.rows; i++)
			for (int j = 0; j < I.cols; j++) {
				if (I.at<uchar>(i, j) != 255)
					result += pow(i - (m_01 / m_00), p) * pow(j - (m_01 / m_00), q);
			}
		break;
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;

		for (int i = 0; i < I.rows; i++)
			for (int j = 0; j < I.cols; j++) {
				if (_I(i, j)[0] != 255 || _I(i, j)[1] != 255 || _I(i, j)[2] != 255)
					result += pow(i - (m_01 / m_00), p) * pow(j - (m_01 / m_00), q);
			}
		break;
	}

	return result;
}

double M_20(cv::Mat& I) {
	return m(I, 2, 0) - pow(m(I, 1, 0), 2) / m(I, 0, 0);
}

double M_02(cv::Mat& I) {
	return m(I, 0, 2) - pow(m(I, 0, 1), 2) / m(I, 0, 0);
}

double M_11(cv::Mat& I) {
	return m(I, 1, 1) - m(I, 1, 0) * m(I, 0, 1) / m(I, 0, 0);
}

double M_12(cv::Mat& I) {
	double i = m(I, 1, 0) / m(I, 0, 0);
	double j = m(I, 0, 1) / m(I, 0, 0);
	return m(I, 1, 2) - 2 * m(I, 1, 1) * j - m(I, 0, 2) * i + 2 * m(I, 1, 0) * pow(j, 2);
}

double M_21(cv::Mat& I) {
	double i = m(I, 1, 0) / m(I, 0, 0);
	double j = m(I, 0, 1) / m(I, 0, 0);
	return m(I, 2, 1) - 2 * m(I, 1, 1) * i - m(I, 2, 0) * j + 2 * m(I, 0, 1) * pow(i, 2);
}

double M_30(cv::Mat& I) {
	double i = m(I, 1, 0) / m(I, 0, 0);
	return m(I, 3, 0) - 3 * m(I, 2, 0) * i + 2 * m(I, 1, 0) * pow(i, 2);
}

double M_03(cv::Mat& I) {
	double j = m(I, 0, 1) / m(I, 0, 0);
	return m(I, 0, 3) - 3 * m(I, 0, 2) * j + 2 * m(I, 0, 1) * pow(j, 2);
}

double M1(cv::Mat& I) {
	return (M_20(I) + M_02(I)) / pow(m(I, 0, 0), 2);
}

double M2(cv::Mat& I) {
	return (pow(M_20(I) - M_02(I), 2) + 4 * pow(M_11(I), 2)) / pow(m(I, 0, 0), 4);
}

double M3(cv::Mat& I) {
	return (pow(M_30(I) - 3 * M_12(I), 2) + pow(3 * M_21(I) - M_03(I), 2)) / pow(m(I, 0, 0), 5);
}

double M7(cv::Mat& I) {
	return (M_20(I) * M_02(I) - pow(M_11(I), 2)) / pow(m(I, 0, 0), 4);
}

void calculate_features(const cv::Mat& I, bool oneObject) {

	std::vector<double> m1_features, m3_features, m7_features;
	std::vector<std::vector<double>> m1_features_areas(4);
	std::vector<std::vector<double>> m3_features_areas(4);
	std::vector<std::vector<double>> m7_features_areas(4);

	for (int i = -25; i <= 25; i = i + 10) // rozci¹ganie na ukos
		for (int j = -15; j <= 15; j = j + 5) { // zgniatanie i rozci¹ganie horyzontalne

			cv::Mat transformed_image;

			cv::Point2f srcPoints[3] = { cv::Point2f(0, 0), cv::Point2f(100, 0), cv::Point2f(0, 100) };
			cv::Point2f dstPoints[3] = { cv::Point2f(0, 0 + i), cv::Point2f(100 + j, 0), cv::Point2f(0, 100 + i) };

			cv::Mat warp_mat = cv::getAffineTransform(srcPoints, dstPoints);

			cv::warpAffine(I, transformed_image, warp_mat, I.size());
			transformed_image = open(transformed_image);

			if (oneObject) {

				BoundingBox box = find_object(transformed_image);
				cv::Mat clip = transformed_image(cv::Rect(cv::Point(box.x0, box.y0), cv::Point(box.x1, box.y1)));

				m1_features.push_back(M1(clip));
				m3_features.push_back(M3(clip));
				m7_features.push_back(M7(clip));
			}
			else {
				auto contours = find_contours(transformed_image);

				std::vector<BoundingBox> close_areas = find_close_areas(transformed_image, contours);

				uchar box_number = 1;
				uchar list_index = 0;
				for (BoundingBox box : close_areas) {

					if (box_number != 3 && box_number != 6) {
						cv::Mat clip = transformed_image(cv::Rect(cv::Point(box.x0, box.y0), cv::Point(box.x1, box.y1)));

						m1_features_areas[list_index].push_back(M1(clip));
						m3_features_areas[list_index].push_back(M3(clip));
						m7_features_areas[list_index].push_back(M7(clip));
						list_index++;
					}
					
					box_number++;
				}
			}

		}

	if (oneObject) {
		std::cout << "M1" << std::endl;
		std::cout << "Min: " << *std::min_element(m1_features.begin(), m1_features.end()) << std::endl;
		std::cout << "Max: " << *std::max_element(m1_features.begin(), m1_features.end()) << std::endl;
		std::cout << "Mean: " << average(m1_features) << std::endl;
		std::cout << "Std: " << std_dev(m1_features) << std::endl << std::endl;

		std::cout << "M3" << std::endl;
		std::cout << "Min: " << *std::min_element(m3_features.begin(), m3_features.end()) << std::endl;
		std::cout << "Max: " << *std::max_element(m3_features.begin(), m3_features.end()) << std::endl;
		std::cout << "Mean: " << average(m3_features) << std::endl;
		std::cout << "Std: " << std_dev(m3_features) << std::endl << std::endl;

		std::cout << "M7" << std::endl;
		std::cout << "Min: " << *std::min_element(m7_features.begin(), m7_features.end()) << std::endl;
		std::cout << "Max: " << *std::max_element(m7_features.begin(), m7_features.end()) << std::endl;
		std::cout << "Mean: " << average(m7_features) << std::endl;
		std::cout << "Std: " << std_dev(m7_features) << std::endl << std::endl;
	}
	else {
		uchar nr = 1;
		for (std::vector<double> area : m1_features_areas) {

			std::cout << "Obszar " << nr << std::endl;
			std::cout << "M1" << std::endl;
			std::cout << "Min: " << *std::min_element(area.begin(), area.end()) << std::endl;
			std::cout << "Max: " << *std::max_element(area.begin(), area.end()) << std::endl;
			std::cout << "Mean: " << average(area) << std::endl;
			std::cout << "Std: " << std_dev(area) << std::endl << std::endl;

			nr++;
		}

		nr = 1;
		for (std::vector<double> area : m3_features_areas) {

			std::cout << "Obszar " << nr << std::endl;
			std::cout << "M3" << std::endl;
			std::cout << "Min: " << *std::min_element(area.begin(), area.end()) << std::endl;
			std::cout << "Max: " << *std::max_element(area.begin(), area.end()) << std::endl;
			std::cout << "Mean: " << average(area) << std::endl;
			std::cout << "Std: " << std_dev(area) << std::endl << std::endl;

			nr++;
		}

		nr = 1;
		for (std::vector<double> area : m7_features_areas) {

			std::cout << "Obszar " << nr << std::endl;
			std::cout << "M7" << std::endl;
			std::cout << "Min: " << *std::min_element(area.begin(), area.end()) << std::endl;
			std::cout << "Max: " << *std::max_element(area.begin(), area.end()) << std::endl;
			std::cout << "Mean: " << average(area) << std::endl;
			std::cout << "Std: " << std_dev(area) << std::endl << std::endl;

			nr++;
		}
	}
	
}

double average(std::vector<double> const& v) {

	if (v.empty()) {
		return 0;
	}

	auto const count = static_cast<float>(v.size());
	return std::reduce(v.begin(), v.end()) / count;
}

double std_dev(const std::vector<double>& v) {

	double mean = average(v);

	std::vector<double> diff(v.size());
	std::transform(v.begin(), v.end(), diff.begin(), [mean](double x) { return x - mean; });

	double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
	double stdev = std::sqrt(sq_sum / (v.size() - 1));

	return stdev;
}

Point geo_center(BoundingBox box) {
	return Point((box.x1 - box.x0) / 2 + box.x0, (box.y1 - box.y0) / 2 + box.y0);
}

double euclidean_distance(const Point& p1, const Point& p2) {
	return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

std::vector<double> calculate_knn(const std::vector<Point>& points, int k) {

	std::vector<double> knn_distances;
	for (Point p1 : points) {
		std::vector<double> distances;
		for (Point p2 : points) {
			if (p1 != p2)
				distances.push_back(euclidean_distance(p1, p2));
		}
		std::sort(distances.begin(), distances.end());
		if (distances.size() >= k)
			knn_distances.push_back(distances[k - 1]);
		else if (!distances.empty())
			knn_distances.push_back(distances.back());
	}

	return knn_distances;
}

std::vector<int> DBSCAN(const std::vector<Point>& points, double eps, int minPts, int maxClusterSize) {
	int n = points.size();
	std::vector<int> labels(n, 0);
	int cluster_id = 0;

	for (int i = 0; i < n; i++) {
		if (labels[i] != 0) continue;

		std::vector<int> neighbors;
		for (int j = 0; j < n; ++j) {
			if (euclidean_distance(points[i], points[j]) <= eps) {
				neighbors.push_back(j);
			}
		}

		if (neighbors.size() < minPts) {
			labels[i] = -1;
			continue;
		}

		cluster_id++;
		labels[i] = cluster_id;

		std::vector<int> seed_set = neighbors;
		int cluster_size = 1;
		for (size_t k = 0; k < seed_set.size(); k++) {
			int idx = seed_set[k];
			if (labels[idx] == -1) {
				labels[idx] = cluster_id;
			}
			if (labels[idx] != 0) continue;

			labels[idx] = cluster_id;
			cluster_size++;
			if (cluster_size >= maxClusterSize)
				break;

			std::vector<int> neighbors2;
			for (int j = 0; j < n; j++) {
				if (euclidean_distance(points[idx], points[j]) <= eps) {
					neighbors2.push_back(j);
				}
			}

			if (neighbors2.size() >= minPts) {
				seed_set.insert(seed_set.end(), neighbors2.begin(), neighbors2.end());
			}
		}
	}

	return labels;
}

std::vector<BoundingBox> detect_logo(const cv::Mat& I, bool predicate(const HSV& hsv)) {

	std::vector<BoundingBox> final_logos;
	Image image_hsv = convertBGRtoHSV(I);

	cv::Mat bin_image = segmentize_image(image_hsv, predicate);
	bin_image = open(bin_image);
	bin_image = close(bin_image);

	auto contours = find_contours(bin_image);

	std::vector<BoundingBox> close_areas = find_close_areas(bin_image, contours);

	cv::Mat cloned = bin_image.clone();
	for (BoundingBox area : close_areas) {
		cv::rectangle(cloned, cv::Rect(cv::Point(area.x0, area.y0), cv::Point(area.x1, area.y1)), cv::Scalar(127), 1);
	}
	cv::imshow("image", cloned);
	cv::waitKey(-1);

	std::vector<BoundingBox> logo_elems;

	for (BoundingBox box : close_areas) {

		cv::Mat clip = bin_image(cv::Rect(cv::Point(box.x0, box.y0), cv::Point(box.x1, box.y1)));
		double m1 = M1(clip);
		double m7 = M7(clip);
		double m3 = M3(clip);

		if (((abs(m1 - DOT_M1) <= DOT_M1_STD * 6 && abs(m7 - DOT_M7) <= DOT_M7_STD * 6 && abs(m3 - DOT_M3) <= DOT_M3_STD * 2) ||
			(abs(m1 - P_M1) <= P_M1_STD * 2 && abs(m7 - P_M7) <= P_M7_STD * 2 && abs(m3 - P_M3) <= P_M3_STD * 2) ||
			(abs(m1 - K_M1) <= K_M1_STD * 2 && abs(m7 - K_M7) <= K_M7_STD * 2 && abs(m3 - K_M3) <= K_M3_STD * 2) ||
			(abs(m1 - O_M1) <= O_M1_STD * 2 && abs(m7 - O_M7) <= O_M7_STD * 2 && abs(m3 - O_M3) <= O_M3_STD * 2)))
				logo_elems.push_back(box);
	}

	cloned = I.clone();
	for (BoundingBox area : logo_elems) {
		cv::rectangle(cloned, cv::Rect(cv::Point(area.x0, area.y0), cv::Point(area.x1, area.y1)), cv::Scalar(0, 0, 0), 1);
	}
	cv::imshow("image", cloned);
	cv::waitKey(-1);

	std::vector<Point> centers;
	for (BoundingBox box : logo_elems) {
		centers.push_back(geo_center(box));
	}

	std::vector<double> knn_distances = calculate_knn(centers, 3);

	double eps = average(knn_distances);

	if (eps < I.cols / 2 && eps < I.rows / 2) {
		std::vector<int> labels = DBSCAN(centers, eps, 2, 4);

		std::map<int, BoundingBox> logos;

		for (size_t i = 0; i < labels.size(); ++i) {
			if (labels[i] != -1) {

				BoundingBox logo_box(I.cols, I.rows);

				if (logos.find(labels[i]) != logos.end()) {
					logo_box = logos[labels[i]];
				}

				logo_box.x0 = std::min(logo_elems[i].x0, logo_box.x0);
				logo_box.y0 = std::min(logo_elems[i].y0, logo_box.y0);
				logo_box.x1 = std::max(logo_elems[i].x1, logo_box.x1);
				logo_box.y1 = std::max(logo_elems[i].y1, logo_box.y1);

				logos[labels[i]] = logo_box;
			}
		}

		for (auto& elem : logos) {
			final_logos.push_back(elem.second);
		}
	}
	else {
		final_logos = logo_elems;
	}

	return final_logos;
}
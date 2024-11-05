

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "converter.hpp"
#include "processing.hpp"
#include "features.hpp"


int main()
{
    cv::Mat image = cv::imread("logo1.jpg");
    auto logo_boxes = detect_logo(image, [](const HSV& hsv) { return hsv.value > 85; });

    for (BoundingBox logo_box : logo_boxes) {
        cv::rectangle(image, cv::Rect(cv::Point(logo_box.x0, logo_box.y0), cv::Point(logo_box.x1, logo_box.y1)), cv::Scalar(0, 0, 0), 1);
    }
   
    cv::imshow("image", image);
    cv::waitKey(-1);

    // ANALIZA KOLORÓW
    //analyze_colors(image_hsv);
    

    // OBLICZANIE CECH DLA SEGMENTÓW TRAKTOWANYCH JAKO JEDEN OBIEKT I JAKO KILKA ELEMENTÓW
    // Image image_hsv = convertBGRtoHSV(image);
    //cv::Mat all_image = segmentize_image(image_hsv, [](const HSV& hsv) { return hsv.value >= 80 && hsv.value <= 100; });

    //calculate_features(all_image, false);
    //calculate_features(all_image, true);

    return 0;
}


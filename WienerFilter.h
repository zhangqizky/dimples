#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>

#include<assert.h>


double WienerFilter(const cv::Mat& src, cv::Mat& dst, const cv::Size& block = cv::Size(5, 5));

void WienerFilter(const cv::Mat& src, cv::Mat& dst, double noiseVariance, const cv::Size& block = cv::Size(5, 5));


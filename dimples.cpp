//
//  main.cpp
//  Dimples_xcode
//
//  Created by zhangqizky on 2019/12/23.
//  Copyright © 2019年 tangxi. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include "WienerFilter.h"
using namespace std;
using namespace cv;


void zero_mean(const Mat &input,Mat &output)
{
    double mean_value= mean(input)[0];
    output = input-mean_value;
}


double PCE(Mat img)
{
    //设置模板，按照每8*8的左上角值为1
    Mat T_ = Mat(img.rows, img.cols, CV_8UC1);
    for(int i =0;i<img.rows;i+=8)
    {
        for(int j =0;j<img.cols;j+=8)
        {
            T_.at<uchar>(i,j) = 1;
        }
    }
    Mat T;
    zero_mean(T_,T);
    // img.convertTo(img,CV_64FC1);
    cout<<img.channels()<<endl;
    // imshow("test",T);
    // waitKey(0);
    // imwrite("T_cpp.jpg",T);
    //计算F，F是一个8*8的矩阵，其计算公式如论文中式(11)
    Mat F = Mat(8,8,CV_64FC1);
    for(int u =0;u<8;u++)
    {
        for(int v =0;v<8;v++)
        {
            float sum =0;
            for(int x = 0;x<img.rows-8;x++)
            {
                for(int y =0;y<img.cols-8;y++)
                {
                    sum+=img.at<uchar>(x,y) * T.at<uchar>(x+u,y+v);
                }
            }
            F.at<double>(u,v) = sum;
        }
    }
    //求F矩阵的平方
    multiply(F,F,F);
    double p_sum = 0.0;
    double minValue=0, maxValue=0;
    Point minLoc, maxLoc;
    //求平方后F矩阵的最大值maxValue和最小值minValue，以及对应的坐标minLoc，maxLoc
    //maxLoc contains coordinate of maximum value
    minMaxLoc(F, &minValue, &maxValue, &minLoc, &maxLoc);
    
    //第一种方式计算式(10)的分母，先求全部的和，再把最大值减去
    Scalar s = sum(F);
    p_sum = s[0]-maxValue;
    cout<<p_sum<<endl;
    //第二种方式计算式(10)的分母，求和的时候不把最大值算进去
    double sum_value = 0.0;
    for(int i =0;i<8;i++)
    {
        for(int j = 0;j<8;j++)
        {
            if(!(i==maxLoc.y &&j==maxLoc.x))
            {
                sum_value+=F.at<double>(i,j);
            }
        }
    }
    //实验过两者是相同的,也说明计算的没有问题
    cout<<"第一种方式求分母:"<<p_sum<<endl;
    cout<<"第二种方式求分母:"<<sum_value<<endl;
    p_sum = p_sum/63;
    return maxValue/p_sum;
}

//每个对应元素相乘再相加
double multiplyAdd(const Mat &T,const Mat &block)
{
    double sum =0.0;
    for(int i =0;i<T.rows;i++)
    {
        for(int j =0;j<T.cols;j++)
        {
            sum+=T.at<double>(i,j)*block.at<double>(i,j);
        }
    }
    return sum;
}

double pce(const Mat &input)
{
    //设置8*8的模板
    Mat T_ = Mat(8, 8, CV_64FC1);
    T_.at<double>(0,0) = 1.0;
    Mat T=Mat(8, 8, T_.type());
    zero_mean(T_,T);
//    cout<<T<<endl;
    
    int rows = input.rows;
    int cols = input.cols;
    Mat res_pce = Mat(Size(rows-8,cols-8),CV_64FC1);
    
    for(int j = 0;j<cols-8;j++)
    {
        for(int i = 0;i<rows-8;i++)
        {
            Rect r= Rect(j,i,8,8);
            Mat roi = input(r);
            res_pce.at<double>(i,j)=multiplyAdd(roi,T);
        }
    }
    //找到所有PCE值里面最大的
    double maxValue,minValue;
    Point maxLoc,minLoc;
    minMaxLoc(res_pce,&minValue,&maxValue, &minLoc, &maxLoc);
    return maxValue;
}

/*两图像相减，不做截断的减法*/
void sub(const Mat &input1,const Mat &input2,Mat&output)
{
    assert(input1.size()==input2.size());
    for(int i =0;i<input1.rows;i++)
    {
        for(int j =0;j<input1.cols;j++)
        {
            double res= input1.at<uchar>(i,j) - input2.at<uchar>(i,j);
            // cout<<res<<endl;
            output.at<double>(i,j) = res;
        }
    }
}
/**
 将画图像划分为不重叠的图像块，参数依次为图像，竖直方向的块数，水平方向的块数，返回的blocks
 */
int subdivide(const cv::Mat &img, const int rowDivisor, const int colDivisor, std::vector<cv::Mat> &blocks)
{
    /* 图像简单判断 */
    if(!img.data || img.empty())
        std::cerr << "Problem Loading Image" << std::endl;
    
//    Mat maskImg = img.clone();
    
    //看图像的宽和高是否能除尽想要划分的块数
    bool co = (img.cols % colDivisor == 0);
    bool ro = (img.rows % rowDivisor == 0);
    
    if(co && ro)
    {
        for(int y = 0; y < img.cols; y += img.cols / colDivisor)
        {
            for(int x = 0; x < img.rows; x += img.rows / rowDivisor)
            {
                Mat roi =img(cv::Rect(y, x, (img.cols / colDivisor), (img.rows / rowDivisor))).clone();
                Mat roi_zero_mean = Mat(Size(roi.rows,roi.cols),roi.type());
                zero_mean(roi, roi_zero_mean);
                blocks.push_back(roi_zero_mean);
                
//                rectangle(maskImg, Point(y, x), Point(y + (maskImg.cols / colDivisor) - 1, x + (maskImg.rows / rowDivisor) - 1), CV_RGB(255, 0, 0), 1); // visualization
                
            }
        }
//        imshow("Image", maskImg); // visualization
//        waitKey(0); // visualization
    }
    else if(img.cols % colDivisor != 0)
    {
        cerr << "Error: Please use another divisor for the column split." << endl;
        exit(1);
    }else if(img.rows % rowDivisor != 0)
    {
        cerr << "Error: Please use another divisor for the row split." << endl;
        exit(1);
    }
    return EXIT_SUCCESS;
}

int main(int argc, char**argv)
{
    
//    if(argc<2)
//    {
//        cout<<"not enough arguments..."<<endl;
//        return -1;
//    }
    vector<cv::String> imagenames;
    string path ="/Users/tangxi/Documents/code/dimples/test_images/";
    glob(path+"*.jpg", imagenames, false);
    
    cout<<imagenames.size()<<endl;
    for(int i =0;i<imagenames.size();i++)
    {
//        cout<<imagenames[i]<<endl;
        Mat img = imread(imagenames[i],0);
        // img.convertTo(img,CV_64FC1);

        // Mat img = Mat(Size(5,5),CV_8UC1,Scalar(240));
        // img.at<double>(2,2) = 20;

        //计算维纳滤波之后的残差
        Mat wiener=Mat(img.size(),img.type());

        double estimatedNoiseVariance = WienerFilter(img,wiener,Size(3,3));
        // cout << wiener<<endl;

        Mat noise = Mat(img.size(),CV_64FC1);
        sub(wiener,img,noise);
        // subtract(wiener,img,noise);

        // cout<<noise<<endl;

//        imshow("wiener filter",noise);
//        waitKey(0);

        //并将残差进行零均值化
        // Mat noise_zeromean = Mat(noise.size(),noise.type());
        // zero_mean(noise,noise_zeromean);

        // double minValue,maxValue;
        // Point minLoc,maxLoc;
        // minMaxLoc(noise_zeromean,&minValue,&maxValue,&minLoc,&maxLoc);

        // cout<<minValue<<endl;

        // //将图像分为32*32的小块，并分别进行零均值化，再加在一起求平均，得到一个32*32的输出
        img.convertTo(img, CV_64FC1);
        int block_size = 32;
        int rows = img.rows;
        int cols = img.cols;
//        cout<<"原来的图像高和宽分别为:"<<rows<<" "<<cols<<endl;
        int rows_ = rows-rows%32;
        int cols_ = cols-cols%32;
//        cout<<"现在的图像高和宽分别为:"<<rows_<<" "<<cols_<<endl;
        //将噪声图像划分为32*32的块，并对所有的32*32的块求和之后再平均d，得到一个32*32的小图像
        vector<Mat> blocks;
        int colDivisor = cols_/block_size;
        int rowDivisor = rows_/block_size;
        Mat img_ = noise(Rect(0,0,cols_,rows_));
//        imshow("test", img_);
//        waitKey();
        subdivide(img_, rowDivisor,colDivisor , blocks);
        
//        for (int r =0;r<rows_-32;r+=32)
//        {
//            for(int c =0;c<cols_-32;c+=32)
//            {
//                cout<<r<<" "<<r+32<<endl;
//                cout<<c<<" "<<c+32<<endl;
//                Rect re = Rect(c,r,32,32);
//                // cout<<re<<endl;
//                Mat block = img(re);
//                count_block++;
////                imshow("block",block);
////                waitKey(0);
//                Mat block_zero_mean = Mat(block.size(),block.type());
//                //做一下零均值
//                zero_mean(block,block_zero_mean);
//                blocks.push_back(block_zero_mean);
//            }
//        }
//        cout<<"block num:"<<blocks.size()<<endl;
        
//        cout<<"done."<<endl;
        //求平均block,大小为32*32
        Mat sum = Mat(Size(32,32),CV_64FC1);
        for(int x=0;x<32;x++)
        {
            for(int y=0;y<32;y++)
            {
                double sum_value =0.0;
                for(int k =0;k<blocks.size();k++)
                {
//                    cout<<blocks[k].at<double>(x,y)<<endl;
                    sum_value+=blocks[k].at<double>(x,y);
                }
                sum_value = sum_value/double(blocks.size());
//                cout<<sum_value<<endl;
//                cout<<sum_value/double(blocks.size());
                sum.at<double>(x,y) = sum_value;
            }
        }
        cout<<"done."<<endl;
//        sum = sum*(1/32);
//        cout<<sum<<endl;
        Mat sum_zero_mean = Mat(Size(sum.rows,sum.cols),sum.type());
        zero_mean(sum, sum_zero_mean);
        //对这个32*32的block求PCE，一共有25*25个pce，取其中最大的返回
        double pce_res=pce(sum_zero_mean);


        // // cout<<img<<endl;
        // cout<<img.channels()<<endl;
        // cout<<img.type()<<endl;

        // double res = PCE(noise_zeromean);
        cout<<imagenames[i]<<":"<<pce_res<<endl;
    }
    return 0;
}

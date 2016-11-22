//
//  main.cpp
//  ORB
//
//  Created by lidongxuan on 2016/11/7.
//  Copyright © 2016年 lidongxuan. All rights reserved.
//
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

double RotationAngle(int frame_num_1,int frame_num_2,
                     int rect_1_x,int rect_1_y,int rect_1_width,int rect_1_height,
                     int rect_2_x,int rect_2_y,int rect_2_width,int rect_2_height)
{
    double pi=3.141592654;
    struct Point{
        double x;
        double y;
    };
    //string path="/Users/lidongxuan/developer/surf/surf/";
    string path="/Users/lidongxuan/developer/matlab/CF2-master/data/car/img/";
    //string path="/Users/lidongxuan/developer/matlab/CF2-master/data/MotorRolling/img/";
    string s;
    char frame_num[64];
    sprintf(frame_num, "%d", frame_num_1+10000);
    s=frame_num;
    Mat img = imread(path+s.substr(1,4)+".jpg");
    Mat img_1 (img, Rect(rect_1_x, rect_1_y, rect_1_width, rect_1_height));
    sprintf(frame_num, "%d", frame_num_2+10000);
    s=frame_num;
    img = imread(path+s.substr(1,4)+".jpg");
    Mat img_2 (img, Rect(rect_2_x, rect_2_y, rect_2_width, rect_2_height) );
    
    imshow("sd",img_1);
    waitKey();
    imshow("sdd",img_2);
    waitKey();
    
//         Mat img_1=imread("/Users/lidongxuan/developer/matlab/CF2-master/data/car/img/0001.jpg");
//         Mat img_2=imread("/Users/lidongxuan/developer/matlab/CF2-master/data/car/img/0002.jpg");
    
    // -- Step 1: Detect the keypoints using STAR Detector
    vector<KeyPoint> keypoints_1,keypoints_2;
    ORB orb;
    orb.detect(img_1, keypoints_1);
    orb.detect(img_2, keypoints_2);
    
    if(keypoints_1.size()==0||keypoints_2.size()==0) return 0;
    
    // -- Stpe 2: Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    orb.compute(img_1, keypoints_1, descriptors_1);
    orb.compute(img_2, keypoints_2, descriptors_2);
    
    //-- Step 3: Matching descriptor vectors with a brute force matcher
    BFMatcher matcher(NORM_HAMMING, true);//NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);
    
    //    //只画25个
        cout << "Number of matched points: " << matches.size() << endl;
    
        nth_element(matches.begin(),    // initial position
                         matches.begin()+9, // position of the sorted element
                         matches.end());     // end position
        // remove all elements after the 25th
        matches.erase(matches.begin()+10, matches.end());
        cout << "Number of new matched points: " << matches.size() << endl;
    
    int rand_num_1,rand_num_2;
    Point p1,p2,_p1,_p2;
    double angle_1,angle_2,angle_err;
    vector<double>  angle_err_group(matches.size(),0);
    if(matches.size()>3)
    {
        for(int i=0;i<matches.size();i++)
        {
            rand_num_1=rand()%(matches.size());
            p1.x=keypoints_1[matches[rand_num_1].queryIdx].pt.x;
            p1.y=keypoints_1[matches[rand_num_1].queryIdx].pt.y;
            _p1.x=keypoints_2[matches[rand_num_1].trainIdx].pt.x;
            _p1.y=keypoints_2[matches[rand_num_1].trainIdx].pt.y;
            
            rand_num_2=rand()%(matches.size());
            if(rand_num_2==rand_num_1&&rand_num_1==0) rand_num_2=rand_num_2+1;
            if(rand_num_2==rand_num_1&&rand_num_1==matches.size()-1) rand_num_2=rand_num_2-1;
            
            p2.x=keypoints_1[matches[rand_num_2].queryIdx].pt.x;
            p2.y=keypoints_1[matches[rand_num_2].queryIdx].pt.y;
            _p2.x=keypoints_2[matches[rand_num_2].trainIdx].pt.x;
            _p2.y=keypoints_2[matches[rand_num_2].trainIdx].pt.y;
            
            if((p2.x-p1.x)>=0)
                angle_1=atan((p2.y-p1.y)/(p2.x-p1.x+2.3e-300));
            else
                angle_1=atan((p2.y-p1.y)/(p2.x-p1.x))+pi;
            
            if((_p2.x-_p1.x)>=0)
                angle_2=atan((_p2.y-_p1.y)/(_p2.x-_p1.x+2.3e-300));
            else
                angle_2=atan((_p2.y-_p1.y)/(_p2.x-_p1.x))+pi;
            
            if(angle_1-angle_2>=0)
                angle_err=angle_1-angle_2;
            else
                angle_err=angle_1-angle_2+2*pi;
            
            angle_err_group[i]=angle_err;
            cout<<p1.y<<","<<p2.y<<"     "<<p1.x<<","<<p2.x<<endl;
            //cout<<"第"<<i<<"个差:"<<angle_err<<endl;
            //angle_sum=angle_sum+angle_err;
        }
        
        //bubble sort angle_err_group
        double temp;
        for(int i = 0;i<matches.size()-1;i++)
        {
            for(int j = 0;j<matches.size()-1-i;j++)
            {
                if(angle_err_group[j] > angle_err_group[j+1])
                {
                    temp=angle_err_group[j];
                    angle_err_group[j]=angle_err_group[j+1];
                    angle_err_group[j+1]=temp;
                }
            }
        }
        
        double err_sum = 0;double count=0;
        for(unsigned long i = matches.size()*4/10;i<=matches.size()*6/10;i++)
        {
            err_sum=err_sum+angle_err_group[i];
            count=count+1;
            //cout<<angle_err_group[i]<<endl;
        }
        
            // -- dwaw matches
            Mat img_mathes;
            drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_mathes);
            // -- show
            imshow("Mathces", img_mathes);
            waitKey();
        
        return err_sum/count;
    }
    else
        return 0;
}
int main(int argc, char** argv)
{
    cout<<RotationAngle(1,4,378,316,77,114,375,318,75,112)*180/3.141592654<<endl;
    return 0;
}

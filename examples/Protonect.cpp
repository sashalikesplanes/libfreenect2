/*
-- Georgia Tech 2016 Spring
--
-- This is a sample code to show how to use the libfreenet2 with OpenCV
--
-- The code will streams RGB, IR and Depth images from an Kinect sensor.
-- To use multiple Kinect sensor, simply initial other "listener" and "frames"

-- This code refered from sample code provided from libfreenet2: Protonect.cpp
-- https://github.com/OpenKinect/libfreenect2
-- and another discussion from: http://answers.opencv.org/question/76468/opencvkinect-onekinect-for-windows-v2linuxlibfreenect2/


-- Contact: Chih-Yao Ma at <cyma@gatech.edu>

--Build option
g++ KinectOneStream.cpp -std=c++11 -o out `pkg-config opencv --cflags --libs` `pkg-config freenect2 --cflags --libs` 

*/

//! [headers]
#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <time.h>
#include <signal.h>
#include <opencv2/opencv.hpp>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>
//! [headers]
//
#include "nanodet.h"
#include <thread>         // std::thread
#include <mutex>          // std::mutex, std::lock

using namespace std;
using namespace cv;

//! [context]
libfreenect2::Freenect2 freenect2;
libfreenect2::Freenect2Device *dev = 0;
libfreenect2::PacketPipeline *pipeline = 0;
//! [context]

//! [listeners]
libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color |
                                              libfreenect2::Frame::Depth |
                                              libfreenect2::Frame::Ir);
libfreenect2::FrameMap frames;
libfreenect2::Registration* registration;

bool protonect_shutdown = false; // Whether the running application should shut down.

cv::Mat rgbmat, depthmat, depthmatUndistorted, irmat, rgbd, rgbd2, test_ir, test_ir3;
libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4), depth2rgb(1920, 1080 + 2, 4); // check here (https://github.com/OpenKinect/libfreenect2/issues/337) and here (https://github.com/OpenKinect/libfreenect2/issues/464) why depth2rgb image should be bigger

std::mutex mat_lock;

void sigint_handler(int s)
{
  protonect_shutdown = true;
}

void get_latest_frames()
{
    //! [loop start]
    while(!protonect_shutdown)
    {
        listener.waitForNewFrame(frames);
        libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
        libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
        libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
        //! [loop start]

        mat_lock.lock();
        cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data).copyTo(rgbmat);
        cv::Mat(ir->height, ir->width, CV_32FC1, ir->data).copyTo(irmat);
        cv::Mat(depth->height, depth->width, CV_32FC1, depth->data).copyTo(depthmat);
        
        irmat.convertTo(test_ir, CV_8UC1, 255.0/4096.0);
        cv::merge(std::vector<cv::Mat>(3, test_ir), test_ir3);
        
        //! [registration]
        registration->apply(rgb, depth, &undistorted, &registered, true, &depth2rgb);
        //! [registration]

        cv::Mat(undistorted.height, undistorted.width, CV_32FC1, undistorted.data).copyTo(depthmatUndistorted);
        cv::Mat(registered.height, registered.width, CV_8UC4, registered.data).copyTo(rgbd);
        cv::Mat(depth2rgb.height, depth2rgb.width, CV_32FC1, depth2rgb.data).copyTo(rgbd2);
        mat_lock.unlock();
        

    //! [loop end]
        listener.release(frames);
    }
    //! [loop end]

}

void get_detections(bool output_dets, bool output_image, bool input_image, std::string base_path)
{
    NanoDet detector = NanoDet("./nanodet-train2.param", "./nanodet-train2.bin", true);
    int i = 0;
    int img_multiple = 10;
    while (!protonect_shutdown)
    {
        mat_lock.lock();
        if (test_ir3.empty())
        {
            mat_lock.unlock();
            continue;
        }
        if (input_image && i++ % img_multiple == 0)
        {
            i++;
            cv::imwrite(base_path + "/" + std::to_string(i) + ".jpg", test_ir3);
        }

        auto results = detector.resize_detect_and_draw(test_ir3);
        mat_lock.unlock();
        std::vector<BoxInfo> dets = std::get<0>(results);
        cv::Mat result_img = std::get<1>(results);
        
        if (output_dets)
        {
            std::string json = "JSON$$$[";
            for (int i = 0; i < dets.size(); i++)
            {
                if (i != 0)
                    json += ", ";
                json += "{\"label\": " + std::to_string(dets[i].label) + ", \"score\": " + std::to_string(dets[i].score) + ", \"x1\": " + std::to_string(dets[i].x1) + ", \"y1\": " + std::to_string(dets[i].y1) + ", \"x2\": " + std::to_string(dets[i].x2) + ", \"y2\": " + std::to_string(dets[i].y2) + "}";
            }
            json += "]";
            std::cout << json << std::endl;
        }
        
        if (output_image)
        {
            cv::imwrite(base_path + "/result.jpg", result_img);
        }
    }
    std::cerr << "Finished detection loop" << std::endl;
}


int main(int argc, char *argv[])
{
    std::cout << "Streaming from Kinect One sensor!" << std::endl;
    protonect_shutdown = false;

    bool output_dets = false;
    bool output_image = false;
    bool input_image = false;
    std::string base_path = "";
    
    if (argc >= 5)
    {
        if (strcmp(argv[1], "1") == 0)
            output_dets = true;
        if (strcmp(argv[2], "1") == 0)
            output_image = true;
        if (strcmp(argv[3], "1") == 0)
            input_image = true;
        base_path = argv[4];
    }
    else
    {
        std::cout << "Usage: ./Protonect dets (0|1) out (0|1) in (0|1) path/to/save" << std::endl;
        return -1;
    }
    
    
    
    std::cout << "Output detections: " << output_dets << std::endl;
    std::cout << "Output image: " << output_image << std::endl;
    std::cout << "Input image: " << input_image << std::endl;


    //! [discovery]
    if(freenect2.enumerateDevices() == 0)
    {
        std::cout << "no device connected!" << std::endl;
        return -1;
    }

    string serial = freenect2.getDefaultDeviceSerialNumber();

    std::cout << "SERIAL: " << serial << std::endl;

    if(pipeline)
    {
        //! [open]
        dev = freenect2.openDevice(serial, pipeline);
        //! [open]
    } else {
        dev = freenect2.openDevice(serial);
    }

    if(dev == 0)
    {
        std::cout << "failure opening device!" << std::endl;
        return -1;
    }

    signal(SIGINT, sigint_handler);


    dev->setColorFrameListener(&listener);
    dev->setIrAndDepthFrameListener(&listener);
    //! [listeners]

    //! [start]
    dev->start();

    std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
    std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;
    //! [start]

    //! [registration setup]
    registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
    //! [registration setup]
    //
    std::thread frame_thread(get_latest_frames);
    frame_thread.detach();
    std::thread nanodet_thread(get_detections, output_dets, output_image, input_image, base_path);
    nanodet_thread.detach();
    
    int i = 1;
    while(!protonect_shutdown)
    {
        cin>>i;
        if (!i)
            protonect_shutdown = true;
    } 

    //! [stop]
    dev->stop();
    dev->close();
    //! [stop]

    delete registration;

    std::cout << "Streaming Ends!" << std::endl;
    return 0;
}


// void display()
// {
//     std::vector<BoxInfo> dets;
//     switch (arg)
//     {
//     case 1:
//         cv::imshow("rgb", rgbmat);
//         break;
//     case 2:
//         cv::imshow("ir", irmat / 4096.0f);
//         break;
//     case 3:
//         cv::imshow("depth", depthmat / 4096.0f);
//         break;
//     case 4:
//         cv::imshow("undistorted", depthmatUndistorted / 4096.0f);
//         break;
//     case 5:
//         cv::imshow("registered", rgbd);
//         break;
//     case 6:
//         cv::imshow("depth2RGB", rgbd2 / 4096.0f);
//         break;
//     case 7:
//         cv::imshow("test_ir", test_ir3);
//         break;
//     case 8:
//         
//         break;
//     case 9:
//         break;
//     }
// }

// //! [nanodet]
// //! [nanodet]

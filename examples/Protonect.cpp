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
#include <thread> // std::thread
#include <mutex>  // std::mutex, std::lock

using namespace std;
using namespace cv;

//! [context]
libfreenect2::Freenect2 freenect2;
// make a second device
libfreenect2::Freenect2Device *device_window = 0;
libfreenect2::Freenect2Device *device_corridor = 0;
// make a second pipeline
libfreenect2::PacketPipeline *pipeline_window = 0;
libfreenect2::PacketPipeline *pipeline_corridor = 0;
//! [context]

//! [listeners]
// make a second listener
libfreenect2::SyncMultiFrameListener listener_window(libfreenect2::Frame::Ir);
libfreenect2::SyncMultiFrameListener listener_corridor(libfreenect2::Frame::Ir);
// make second frames
libfreenect2::FrameMap frames;
libfreenect2::FrameMap frames1;

bool protonect_shutdown = false; // Whether the running application should shut down.

// double all these
cv::Mat irmat_window, scaled_ir_window, stacked_ir_window;
cv::Mat irmat_corridor, scaled_ir_corridor, stacked_ir_corridor;

std::mutex mat_lock_window;
std::mutex mat_lock_corridor;

void sigint_handler(int s)
{
    protonect_shutdown = true;
}

void get_latest_frames()
{
    while (!protonect_shutdown)
    {
        listener_window.waitForNewFrame(frames);
        libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];

        mat_lock_window.lock();
        cv::Mat(ir->height, ir->width, CV_32FC1, ir->data).copyTo(irmat_window);
        irmat_window.convertTo(scaled_ir_window, CV_8UC1, 255.0 / 4096.0);
        cv::merge(std::vector<cv::Mat>(3, scaled_ir_window), stacked_ir_window);
        mat_lock_window.unlock();

        listener_window.release(frames);
        // repeat for second device
    }
}

void get_detections(bool output_dets, bool output_image, bool input_image, std::string base_path, float score_threshold, float nms_threshold)
{
    NanoDet detector = NanoDet("./nanodet-train2.param", "./nanodet-train2.bin", true);
    int i = 0;
    int img_multiple = 10;
    bool next_window = true;
    while (!protonect_shutdown)
    {
        if (next_window)
        {
            next_window = true; // TODO set this to false to get it all from corridor
            mat_lock_window.lock();
            if (stacked_ir_window.empty())
            {
                mat_lock_window.unlock();
                continue;
            }
            if (input_image && i++ % img_multiple == 0)
            {
                i++;
                cv::imwrite(base_path + "/" + std::to_string(i) + ".jpg", stacked_ir_window);
            }

            auto results = detector.resize_detect_and_draw(stacked_ir_window, output_image, score_threshold, nms_threshold);
            mat_lock_window.unlock();
            std::vector<BoxInfo> dets = std::get<0>(results);
            cv::Mat result_img = std::get<1>(results);

            if (output_dets)
            {
                std::string json = "JSON$$$[";
                for (int i = 0; i < dets.size(); i++)
                {
                    if (i != 0)
                        json += ", ";
                    json += "{\"source\": \"windowCam\", \"label\": " + std::to_string(dets[i].label) + ", \"score\": " + std::to_string(dets[i].score) + ", \"x1\": " + std::to_string(dets[i].x1) + ", \"y1\": " + std::to_string(dets[i].y1) + ", \"x2\": " + std::to_string(dets[i].x2) + ", \"y2\": " + std::to_string(dets[i].y2) + "}";
                }
                json += "]$$$";
                std::cout << json << std::endl;
            }

            if (output_image)
            {
                cv::imwrite(base_path + "/result_window.jpg", result_img);
            }
        }
        else
        {
            // do second device
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
    float score_threshold = 0.5;
    float nms_threshold = 0.3;

    if (argc >= 7)
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

    score_threshold = std::stof(argv[5]);
    nms_threshold = std::stof(argv[6]);

    std::cout << "Output detections: " << output_dets << std::endl;
    std::cout << "Output image: " << output_image << std::endl;
    std::cout << "Input image: " << input_image << std::endl;

    //! [discovery]
    if (freenect2.enumerateDevices() == 0)
    {
        std::cout << "no device connected!" << std::endl;
        return -1;
    }

    string serial = freenect2.getDefaultDeviceSerialNumber();

    std::cout << "SERIAL: " << serial << std::endl;

    if (pipeline_window)
    {
        //! [open]
        device_window = freenect2.openDevice(serial, pipeline_window);
        //! [open]
    }
    else
    {
        device_window = freenect2.openDevice(serial);
    }

    if (device_window == 0)
    {
        std::cout << "failure opening device!" << std::endl;
        return -1;
    }

    signal(SIGINT, sigint_handler);

    device_window->setIrAndDepthFrameListener(&listener_window);
    //! [listeners]

    //! [start]
    device_window->start();

    std::cout << "device serial: " << device_window->getSerialNumber() << std::endl;
    std::cout << "device firmware: " << device_window->getFirmwareVersion() << std::endl;
    //! [start]

    //! [registration setup]
    //! [registration setup]
    //
    std::thread frame_thread(get_latest_frames);
    frame_thread.detach();
    std::thread nanodet_thread(get_detections, output_dets, output_image, input_image, base_path, score_threshold, nms_threshold);
    nanodet_thread.detach();

    int i = 1;
    while (!protonect_shutdown)
    {
        cin >> i;
        if (!i)
            protonect_shutdown = true;
    }

    //! [stop]
    device_window->stop();
    device_window->close();
    //! [stop]

    std::cout << "Streaming Ends!" << std::endl;
    return 0;
}
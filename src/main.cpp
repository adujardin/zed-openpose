// Standard includes
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>

// ZED includes
#include <sl/Camera.hpp>

// Sample includes
#include "GLViewer.hpp"
#include "utils.hpp"

// GFlags: DEFINE_bool, _int32, _int64, _uint64, _double, _string
#include <gflags/gflags.h>

// OpenPose dependencies
#include <openpose/headers.hpp>

float thres_score = 0.6;

// Debugging
DEFINE_int32(logging_level, 3, "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
        " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
        " low priority messages and 4 for important ones.");
// OpenPose
DEFINE_string(model_pose, "COCO", "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
        "`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(model_folder, "models/", "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(net_resolution, /*"320x240"*/"656x368", "Multiples of 16. If it is increased, the accuracy potentially increases. If it is"
        " decreased, the speed increases. For maximum speed-accuracy balance, it should keep the"
        " closest aspect ratio possible to the images or videos to be processed. Using `-1` in"
        " any of the dimensions, OP will choose the optimal aspect ratio depending on the user's"
        " input value. E.g. the default `-1x368` is equivalent to `656x368` in 16:9 resolutions,"
        " e.g. full HD (1980x1080) and HD (1280x720) resolutions.");
DEFINE_string(output_resolution, "-1x-1", "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
        " input image resolution.");
DEFINE_int32(num_gpu_start, 0, "GPU device start number.");
DEFINE_double(scale_gap, 0.3, "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
        " If you want to change the initial scale, you actually want to multiply the"
        " `net_resolution` by your desired initial scale.");
DEFINE_int32(scale_number, 1, "Number of scales to average.");
// OpenPose Rendering
DEFINE_bool(disable_blending, false, "If enabled, it will render the results (keypoint skeletons or heatmaps) on a black"
        " background, instead of being rendered into the original image. Related: `part_to_show`,"
        " `alpha_pose`, and `alpha_pose`.");
DEFINE_double(render_threshold, 0.12, "Only estimated keypoints whose score confidences are higher than this threshold will be"
        " rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
        " while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
        " more false positives (i.e. wrong detections).");
DEFINE_double(alpha_pose, 0.6, "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
        " hide it. Only valid for GPU rendering.");
DEFINE_string(svo_path, "", "SVO filepath");
DEFINE_bool(ogl_ptcloud, false, "Display the point cloud in the OpenGL window");
DEFINE_bool(estimate_floor_plane, true, "Initialize the camera position from the floor plan detected in the scene");
DEFINE_bool(opencv_display, true, "Enable the 2D view of openpose output");
DEFINE_bool(depth_display, false, "Enable the depth display with openCV");

// Using std namespace
using namespace std;
using namespace sl;

// Create ZED objects
sl::Camera zed;
sl::Pose camera_pose;
std::thread zed_callback, openpose_callback;
std::mutex data_in_mtx, data_out_mtx;
std::vector<op::Array<float>> netInputArray;
op::Array<float> poseKeypoints;
op::Point<int> imageSize, outputSize, netInputSize, netOutputSize;
op::PoseModel poseModel;
std::vector<double> scaleInputToNetInputs;
PointObject cloud;
PeoplesObject peopleObj;

bool quit = false;

// OpenGL window to display camera motion
GLViewer viewer;

const int MAX_CHAR = 128;
const sl::UNIT unit = sl::UNIT_METER;
const float MAX_DISTANCE_LIMB = 1; //0.8;
const float MAX_DISTANCE_CENTER = 1.8; //1.5;

// Sample functions
void startZED();
void startOpenpose();
void run();
void close();
void findpose();

int image_width = 720;
int image_height = 405;

bool need_new_image = true;
bool ready_to_start = false;

#define ENABLE_FLOOR_PLANE_DETECTION 1 // Might be disable to use older ZED SDK

// Debug options
#define DISPLAY_BODY_BARYCENTER 0
#define PATCH_AROUND_KEYPOINT 1 

bool initFloorZED(sl::Camera &zed) {
    bool init = false;
#if ENABLE_FLOOR_PLANE_DETECTION
    sl::Plane plane;
    sl::Transform resetTrackingFloorFrame;
    const int timeout = 20;
    int count = 0;

    cout << "Looking for the floor plane to initialize the tracking..." << endl;

    while (!init && count++ < timeout) {
        zed.grab();
        init = (zed.findFloorPlane(plane, resetTrackingFloorFrame) == sl::ERROR_CODE::SUCCESS);
        resetTrackingFloorFrame.getInfos();
        if (init) {
            zed.getPosition(camera_pose, sl::REFERENCE_FRAME_WORLD);
            cout << "Floor found at : " << plane.getClosestDistance(camera_pose.pose_data.getTranslation()) << " m" << endl;
            zed.resetTracking(resetTrackingFloorFrame);
        }
        sl::sleep_ms(20);
    }
    if (init) for (int i = 0; i < 4; i++) zed.grab();
    else cout << "Floor plane not found, starting anyway" << endl;
#endif
    return init;
}

int main(int argc, char **argv) {

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Set configuration parameters for the ZED
    InitParameters initParameters;
    initParameters.camera_resolution = RESOLUTION_HD2K;
    initParameters.depth_mode = DEPTH_MODE_ULTRA; // Might be GPU memory intensive combine with openpose
    initParameters.coordinate_units = unit;
    initParameters.coordinate_system = COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP;
    initParameters.sdk_verbose = 0;
    initParameters.depth_stabilization = true;
    initParameters.svo_real_time_mode = 0;

    if (std::string(FLAGS_svo_path).find(".svo")) {
        cout << "Opening " << FLAGS_svo_path << endl;
        initParameters.svo_input_filename.set(std::string(FLAGS_svo_path).c_str());
    }

    // Open the camera
    ERROR_CODE err = zed.open(initParameters);
    if (err != sl::SUCCESS) {
        std::cout << err << std::endl;
        zed.close();
        return 1; // Quit if an error occurred
    }

    if (FLAGS_estimate_floor_plane)
        initFloorZED(zed);

    // Initialize OpenGL viewer
    viewer.init();

    // init OpenPose
    cout << "OpenPose : loading models..." << endl;
    // ------------------------- INITIALIZATION -------------------------
    // Read Google flags (user defined configuration)
    outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
    netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");

    cout << netInputSize.x << "x" << netInputSize.y << endl;
    netOutputSize = netInputSize;
    poseModel = op::flagsToPoseModel(FLAGS_model_pose);
    // Check no contradictory flags enabled
    if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.) op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
    if (FLAGS_scale_gap <= 0. && FLAGS_scale_number > 1) op::error("Incompatible flag configuration: scale_gap must be greater than 0 or scale_number = 1.", __LINE__, __FUNCTION__, __FILE__);

    // Start ZED callback
    startZED();

    startOpenpose();

    // Set the display callback
    glutCloseFunc(close);
    glutMainLoop();
    return 0;
}

void startZED() {
    quit = false;
    zed_callback = std::thread(run);
}

void startOpenpose() {
    openpose_callback = std::thread(findpose);
}

void findpose() {

    while (!ready_to_start) sl::sleep_ms(2); // Waiting for the ZED

    op::PoseExtractorCaffe poseExtractorCaffe{poseModel, FLAGS_model_folder, FLAGS_num_gpu_start,{}, op::ScaleMode::ZeroToOne, 1};
    poseExtractorCaffe.initializationOnThread();

    while (!quit) {
        INIT_TIMER;
        //  Estimate poseKeypoints
        if (!need_new_image) { // No new image
            data_in_mtx.lock();
            need_new_image = true;
            poseExtractorCaffe.forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
            data_in_mtx.unlock();

            // Extract poseKeypoints
            data_out_mtx.lock();
            poseKeypoints = poseExtractorCaffe.getPoseKeypoints();
            data_out_mtx.unlock();
            //STOP_TIMER("OpenPose");
        } else sl::sleep_ms(1);
    }
}

// The 3D of the point is not directly taken 'as is'. If the measurement isn't valid, we look around the point in 2D to find a close point with a valid depth
sl::float4 getPatchIdx(const int &center_i, const int &center_j, sl::Mat &xyzrgba) {
    sl::float4 out(NAN, NAN, NAN, NAN);
    bool valid_measure;
    int i, j;

    const int R_max = 10;

    for (int R = 0; R < R_max; R++) {
        for (int y = -R; y <= R; y++) {
            for (int x = -R; x <= R; x++) {
                i = center_i + x;
                j = center_j + y;
                xyzrgba.getValue<sl::float4>(i, j, &out, sl::MEM_CPU);
                valid_measure = isfinite(out.z);
                if (valid_measure) return out;
            }
        }
    }

    out = sl::float4(NAN, NAN, NAN, NAN);
    return out;
}

void fill_people_ogl(op::Array<float> &poseKeypoints, sl::Mat &xyz) {
    // Common parameters needed
    const auto numberPeopleDetected = poseKeypoints.getSize(0);
    const auto numberBodyParts = poseKeypoints.getSize(1);

    //https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/media/keypoints_pose.png
    std::vector<int> partsLink = {
        //0, 1,
        2, 1,
        1, 5,
        8, 11,
        1, 8,
        11, 1,
        8, 9,
        9, 10,
        11, 12,
        12, 13,
        2, 3,
        3, 4,
        5, 6,
        6, 7,
        //0, 15,
        //15, 17,
        //0, 14,
        //14, 16,
        16, 1,
        17, 1,
        16, 17
    };

    sl::float4 v1, v2;
    int i, j;

    std::vector<sl::float3> vertices;
    std::vector<sl::float3> clr;

    for (int person = 0; person < numberPeopleDetected; person++) {

        std::map<int, sl::float4> keypoints_position; // 3D + score for each keypoints

        sl::float4 center_gravity(0, 0, 0, 0);
        int count = 0;
        float score;

        for (int k = 0; k < numberBodyParts; k++) {

            score = poseKeypoints[{person, k, 2}
            ];
            keypoints_position[k] = sl::float4(NAN, NAN, NAN, score);

            if (score < FLAGS_render_threshold) continue; // skip low score

            i = round(poseKeypoints[{person, k, 0}
            ]);
            j = round(poseKeypoints[{person, k, 1}
            ]);

#if PATCH_AROUND_KEYPOINT
            xyz.getValue<sl::float4>(i, j, &keypoints_position[k], sl::MEM_CPU);
            if (!isfinite(keypoints_position[k].z))
                keypoints_position[k] = getPatchIdx((const int) i, (const int) j, xyz);
#else
            xyz.getValue<sl::float4>(i, j, &keypoints_position[k], sl::MEM_CPU);
#endif
            keypoints_position[k].w = score; // the score was overridden by the getValue

            if (score >= FLAGS_render_threshold && isfinite(keypoints_position[k].z)) {
                center_gravity += keypoints_position[k];
                count++;
            }
        }

        ///////////////////////////
        center_gravity.x /= (float) count;
        center_gravity.y /= (float) count;
        center_gravity.z /= (float) count;

#if DISPLAY_BODY_BARYCENTER
        float size = 0.1;
        vertices.emplace_back(center_gravity.x, center_gravity.y + size, center_gravity.z);
        vertices.emplace_back(center_gravity.x, center_gravity.y - size, center_gravity.z);
        clr.push_back(generateColor(person));
        clr.push_back(generateColor(person));

        vertices.emplace_back(center_gravity.x + size, center_gravity.y, center_gravity.z);
        vertices.emplace_back(center_gravity.x - size, center_gravity.y, center_gravity.z);
        clr.push_back(generateColor(person));
        clr.push_back(generateColor(person));

        vertices.emplace_back(center_gravity.x, center_gravity.y, center_gravity.z + size);
        vertices.emplace_back(center_gravity.x, center_gravity.y, center_gravity.z - size);
        clr.push_back(generateColor(person));
        clr.push_back(generateColor(person));
#endif
        ///////////////////////////  


        for (int part = 0; part < partsLink.size() - 1; part += 2) {
            v1 = keypoints_position[partsLink[part]];
            v2 = keypoints_position[partsLink[part + 1]];

            // Filtering 3D Skeleton
            // Compute euclidian distance
            float distance = sqrt((v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z) * (v1.z - v2.z));

            float distance_gravity_center = sqrt(pow((v2.x + v1.x)*0.5f - center_gravity.x, 2) +
                    pow((v2.y + v1.y)*0.5f - center_gravity.y, 2) +
                    pow((v2.z + v1.z)*0.5f - center_gravity.z, 2));
            if (isfinite(distance_gravity_center) && distance_gravity_center < MAX_DISTANCE_CENTER && distance < MAX_DISTANCE_LIMB) {
                vertices.emplace_back(v1.x, v1.y, v1.z);
                vertices.emplace_back(v2.x, v2.y, v2.z);
                clr.push_back(generateColor(person));
                clr.push_back(generateColor(person));
            }
        }
    }
    peopleObj.setVert(vertices, clr);
}

void fill_ptcloud(sl::Mat &xyzrgba) {
    std::vector<sl::float3> pts;
    std::vector<sl::float3> clr;
    int total = xyzrgba.getResolution().area();

    float factor = 1;

    pts.resize(total / factor);
    clr.resize(total / factor);

    sl::float4* p_mat = xyzrgba.getPtr<sl::float4>(sl::MEM_CPU);

    sl::float3* p_f3;
    sl::float4* p_f4;
    unsigned char *color_uchar;

    int j = 0;
    for (int i = 0; i < total; i += factor, j++) {
        p_f4 = &p_mat[i];
        p_f3 = &pts[j];
        p_f3->x = p_f4->x;
        p_f3->y = p_f4->y;
        p_f3->z = p_f4->z;
        p_f3 = &clr[j];
        color_uchar = (unsigned char *) &p_f4->w;
        p_f3->x = color_uchar[0] * 0.003921569; // /255
        p_f3->y = color_uchar[1] * 0.003921569;
        p_f3->z = color_uchar[2] * 0.003921569;
    }
    cloud.setVert(pts, clr);
}

void run() {
    sl::RuntimeParameters rt;
    rt.enable_depth = 1;
    rt.enable_point_cloud = 1;
    rt.measure3D_reference_frame = sl::REFERENCE_FRAME_WORLD;

    sl::Mat img_buffer, depth_img_buffer, depth_buffer, depth_buffer2;
    op::Array<float> outputArray, outputArray2;
    cv::Mat inputImage, depthImage, inputImageRGBA, outputImage;

    // ---- OPENPOSE INIT (io data + renderer) ----
    op::ScaleAndSizeExtractor scaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap);
    op::CvMatToOpInput cvMatToOpInput;
    op::CvMatToOpOutput cvMatToOpOutput;

    op::PoseCpuRenderer poseRenderer{poseModel, (float) FLAGS_render_threshold, !FLAGS_disable_blending, (float) FLAGS_alpha_pose};
    op::OpOutputToCvMat opOutputToCvMat;
    // Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
    poseRenderer.initializationOnThread();

    // Init
    imageSize = op::Point<int>{image_width, image_height};
    // Get desired scale sizes
    std::vector<op::Point<int>> netInputSizes;
    double scaleInputToOutput;
    op::Point<int> outputResolution;
    std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution) = scaleAndSizeExtractor.extract(imageSize);

    bool chrono_zed = false;

    while (!quit && zed.getSVOPosition() != zed.getSVONumberOfFrames() - 1) {
        INIT_TIMER
        if (need_new_image) {
            if (zed.grab(rt) == SUCCESS) {

                zed.retrieveImage(img_buffer, VIEW::VIEW_LEFT, sl::MEM_CPU, image_width, image_height);
                data_out_mtx.lock();
                depth_buffer2 = depth_buffer;
                data_out_mtx.unlock();
                zed.retrieveMeasure(depth_buffer, MEASURE::MEASURE_XYZRGBA, sl::MEM_CPU, image_width, image_height);

                cout << zed.getCurrentFPS() << "        \r" << flush;

                inputImageRGBA = slMat2cvMat(img_buffer);
                cv::cvtColor(inputImageRGBA, inputImage, CV_RGBA2RGB);

                if (FLAGS_depth_display)
                    zed.retrieveImage(depth_img_buffer, VIEW::VIEW_DEPTH, sl::MEM_CPU, image_width, image_height);

                if (FLAGS_opencv_display) {
                    data_out_mtx.lock();
                    outputArray2 = outputArray;
                    data_out_mtx.unlock();
                    outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
                }
                data_in_mtx.lock();
                netInputArray = cvMatToOpInput.createArray(inputImage, scaleInputToNetInputs, netInputSizes);
                need_new_image = false;
                data_in_mtx.unlock();

                ready_to_start = true;
                chrono_zed = true;
            } else sl::sleep_ms(1);
        } else sl::sleep_ms(1);

        // -------------------------  RENDERING -------------------------------
        // Render poseKeypoints
        if (data_out_mtx.try_lock()) {

            fill_people_ogl(poseKeypoints, depth_buffer2);
            viewer.update(peopleObj);

            if (FLAGS_ogl_ptcloud) {
                fill_ptcloud(depth_buffer2);
                viewer.update(cloud);
            }

            if (FLAGS_opencv_display) {
                if (!outputArray2.empty())
                    poseRenderer.renderPose(outputArray2, poseKeypoints, scaleInputToOutput);

                // OpenPose output format to cv::Mat
                if (!outputArray2.empty())
                    outputImage = opOutputToCvMat.formatToCvMat(outputArray2);
                data_out_mtx.unlock();
                // Show results
                if (!outputArray2.empty())
                    cv::imshow("Pose", outputImage);
                if (FLAGS_depth_display)
                    cv::imshow("Depth", slMat2cvMat(depth_img_buffer));
                
                cv::waitKey(10);
            }
        }

        if (chrono_zed) {
            //STOP_TIMER("ZED")
            chrono_zed = false;
        }
    }
}

void close() {
    quit = true;
    openpose_callback.join();
    zed_callback.join();
    zed.close();
    viewer.exit();
}

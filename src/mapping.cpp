/*
 * Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion
 * Copyright (C) 2025 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "mapping.h"
#include "gaussian.h"

#include <atomic>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

std::mutex m_buf;
std::condition_variable con;

std::queue<sensor_msgs::PointCloud2ConstPtr> point_buf;
std::queue<geometry_msgs::PoseStampedConstPtr> pose_buf;
std::queue<sensor_msgs::ImageConstPtr> image_buf;
std::queue<sensor_msgs::ImageConstPtr> depth_buf;

std::atomic<bool> exit_flag(false);
std::atomic<double> last_point_time(0.0);
std::atomic<bool> gaussians_initialized(false);

void pointCallback(const sensor_msgs::PointCloud2ConstPtr& point_msg) 
{
    m_buf.lock();
    point_buf.push(point_msg);
    last_point_time = ros::WallTime::now().toSec();
    m_buf.unlock();
}

void poseCallback(const geometry_msgs::PoseStampedConstPtr& pose_msg) 
{
    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
}

void imageCallback(const sensor_msgs::ImageConstPtr& image_msg) 
{
    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();
}

void depthCallback(const sensor_msgs::ImageConstPtr& depth_msg) 
{
    m_buf.lock();
    depth_buf.push(depth_msg);
    m_buf.unlock();
}

bool getAlignedData(Frame& cur_frame)
{
    if (point_buf.empty() || pose_buf.empty() || image_buf.empty() || depth_buf.empty()) 
    {
        return false;
    }

    double frame_time = point_buf.front()->header.stamp.toSec();

    while (1) 
    {
        if (pose_buf.front()->header.stamp.toSec() < frame_time - 0.01) 
        {
            pose_buf.pop();
            if (pose_buf.empty()) 
            {
                return false;
            }
        } 
        else break;
    }
    if (pose_buf.front()->header.stamp.toSec() > frame_time + 0.01) 
    {
        point_buf.pop();
        return false;
    }

    while (1) 
    {
        if (image_buf.front()->header.stamp.toSec() < frame_time - 0.01) 
        {
            image_buf.pop();
            if (image_buf.empty()) 
            {
                return false;
            }
        } 
        else break;
    }
    if (image_buf.front()->header.stamp.toSec() > frame_time + 0.01) 
    {
        point_buf.pop();
        return false;
    }

    while (1) 
    {
        if (depth_buf.front()->header.stamp.toSec() < frame_time - 0.01) 
        {
            depth_buf.pop();
            if (depth_buf.empty()) 
            {
                return false;
            }
        } 
        else break;
    }
    if (depth_buf.front()->header.stamp.toSec() > frame_time + 0.01) 
    {
        point_buf.pop();
        return false;
    }

    auto cur_point = point_buf.front();
    auto cur_pose = pose_buf.front();
    auto cur_image = image_buf.front();
    auto cur_depth = depth_buf.front();

    cur_frame.point_msg = cur_point;
    cur_frame.pose_msg = cur_pose;
    cur_frame.image_msg = cur_image;
    cur_frame.depth_msg = cur_depth;

    point_buf.pop();
    pose_buf.pop();
    image_buf.pop();
    depth_buf.pop();

    return true;
}

/// Prints two tab-separated rows for copy-pasting into the "Runs" and
/// "Evaluations" tracking spreadsheets. This is pure terminal formatting of
/// values already computed and printed by the runtime-statistics and
/// visual-quality summaries above; it performs no training/eval/timing work
/// and cannot affect numerical results. Stream format state is saved and
/// restored so nothing downstream is perturbed. Unknown metadata (experiment
/// id, dataset, sequence, platform, notes) and the hardcoded iterations/KF are
/// left as placeholders; depth-error columns are left empty (not measured).
static void printSpreadsheetRows(const std::shared_ptr<GaussianModel>& gaussians,
                                 const std::shared_ptr<Dataset>& dataset,
                                 double total_extending_time,
                                 const VisualQualityMetrics& metrics)
{
    const char TAB = '\t';
    std::ostream& os = std::cout;
    const std::ios_base::fmtflags saved_flags = os.flags();
    const std::streamsize saved_precision = os.precision();

    // Runs sheet columns: ID, Map Method, Dataset, Sequence, Training platform,
    // Iterations/KF, Extra downsample stride, SH Degree, Final Gauss, Fwd (s),
    // Backward (s), Step (s), CPU-GPU (s), Extending (s), Notes.
    os << "\n===== COPY TO RUNS SHEET =====\n";
    os << "<experiment_id>" << TAB          // ID
       << "glic2" << TAB                    // Map Method
       << "<dataset>" << TAB                // Dataset
       << "<sequence>" << TAB               // Sequence
       << "<platform>" << TAB               // Training platform
       << "<iterations_per_keyframe>" << TAB // Iterations/KF (hardcoded local in optimize())
       << dataset->point_stride_ << TAB     // Extra downsample stride (config point_stride)
       << gaussians->sh_degree_ << TAB      // SH Degree (model/config)
       << gaussians->getXYZ().size(0) << TAB; // Final Gauss
    os << std::fixed << std::setprecision(2)
       << gaussians->t_forward_ << TAB      // Fwd (s)
       << gaussians->t_backward_ << TAB     // Backward (s)
       << gaussians->t_step_ << TAB         // Step (s)
       << gaussians->t_tocuda_ << TAB       // CPU-GPU (s)
       << total_extending_time << TAB;      // Extending (s)
    os << "\n";                             // Notes (empty)

    // Evaluations sheet columns: Experiment, Evaluator/renderer, Test PSNR,
    // Test SSIM, Test LPIPS, Test Depth MAE, Test Depth RMSE, Train PSNR,
    // Train SSIM, Train LPIPS, Train Depth MAE, Train Depth RMSE. Test metrics
    // are the In-Sequence Novel View results; train are the Training View
    // results. Depth-error columns are unmeasured and left empty.
    os << "\n===== COPY TO EVALUATIONS SHEET =====\n";
    os << "<experiment_id>" << TAB          // Experiment
       << "GLIC2" << TAB;                   // Evaluator/renderer
    os << std::fixed << std::setprecision(2) << metrics.test_psnr << TAB;
    os << std::setprecision(3) << metrics.test_ssim << TAB
       << metrics.test_lpips << TAB
       << TAB                               // Test Depth MAE (empty)
       << TAB;                              // Test Depth RMSE (empty)
    os << std::setprecision(2) << metrics.train_psnr << TAB;
    os << std::setprecision(3) << metrics.train_ssim << TAB
       << metrics.train_lpips << TAB
       << TAB                               // Train Depth MAE (empty)
       << "\n";                             // Train Depth RMSE (empty)

    os.flags(saved_flags);
    os.precision(saved_precision);
}

void mapping(const YAML::Node& node, const std::string& result_path, const std::string& lpips_path)
{
    torch::jit::setGraphExecutorOptimize(false);

    Params prm(node);
    std::shared_ptr<GaussianModel> gaussians = std::make_shared<GaussianModel>(prm);
    std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(prm);

    std::chrono::steady_clock::time_point t_start, t_end;
    double total_mapping_time = 0;
    double total_adding_time = 0;
    double total_extending_time = 0;

    Frame cur_frame;
    while (true)
    {
        /// [1] data alignment
        m_buf.lock();
        bool align_flag = getAlignedData(cur_frame);
        m_buf.unlock();
        if (!align_flag)
        {
            /// The bag has stopped and no aligned frame could be formed.
            /// getAlignedData() only returns false *without consuming* a
            /// message when one of the buffers is empty; every other false
            /// path pops a stale point (i.e. makes progress). So once any
            /// buffer is empty, no further aligned frame can ever be built.
            /// This drains the backlog left after the bag ends and also
            /// covers the case of a leftover LiDAR scan whose matching
            /// pose/image never arrived because recording was cut short.
            if (exit_flag)
            {
                m_buf.lock();
                bool exhausted = point_buf.empty() || pose_buf.empty()
                              || image_buf.empty() || depth_buf.empty();
                m_buf.unlock();
                if (exhausted) break;
            }
            continue;
        }
        
        /// [2] add every frame
        t_start = std::chrono::steady_clock::now();
        dataset->addFrame(cur_frame);
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
        if (dataset->is_keyframe_current_)
        {
            total_adding_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
            std::cout << "\033[1;33m     Cur Frame " << dataset->all_frame_num_ - 1 << ",\033[0m";
        }
        else continue;

        if (!gaussians->is_init_)
        {
            /// [3] initialize map
            gaussians->is_init_ = true;
            gaussians_initialized = true;
            gaussians->initialize(dataset);
            gaussians->trainingSetup();
        }
        else 
        {
            /// [4] extend map
            t_start = std::chrono::steady_clock::now();
            extend(dataset, gaussians);
            torch::cuda::synchronize();
            t_end = std::chrono::steady_clock::now();
            total_extending_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        }

        /// [5] optimize map
        t_start = std::chrono::steady_clock::now();
        double updated_num = optimize(dataset, gaussians);
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
        total_mapping_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
        std::cout << std::fixed << std::setprecision(2) 
                  << "\033[1;36m Update " << updated_num / 10000 
                  << "w GS per Iter \033[0m" << std::endl;
    }

    /// [6] evaluation
    std::cout << "\n     🎉 Runtime Statistics 🎉\n";
    std::cout << std::fixed << std::setprecision(2) << "\n        [Total Mapping Time] " << total_mapping_time << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         1) Forward " << gaussians->t_forward_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         2) Backward " << gaussians->t_backward_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         3) Step " << gaussians->t_step_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         4) CPU2GPU " << gaussians->t_tocuda_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "        [Total Adding Time] " << total_adding_time << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "        [Total Extending Time] " << total_extending_time << "s" << std::endl;
    torch::NoGradGuard no_grad;
    VisualQualityMetrics metrics = evaluateVisualQuality(dataset, gaussians, result_path, lpips_path);
    printSpreadsheetRows(gaussians, dataset, total_extending_time, metrics);
    gaussians->saveMap(result_path);

    std::cout << "\n\n😋 Gaussian-LIC Done!\n\n\n";
    ros::shutdown();
}

int main(int argc, char** argv)
{
    std::cout << "\n\n😋 Gaussian-LIC Ready!\n\n\n";
    ros::init(argc, argv, "gaussianlic");
    ros::NodeHandle nh("~");
    ros::Rate loop_rate(1000);
    image_transport::ImageTransport it_(nh);

    ros::Subscriber sub_point = nh.subscribe("/points_for_gs", 10000, pointCallback);
    ros::Subscriber sub_pose = nh.subscribe("/pose_for_gs", 10000, poseCallback);
    image_transport::Subscriber image_sub = it_.subscribe("/image_for_gs", 10000, imageCallback);
    image_transport::Subscriber depth_sub = it_.subscribe("/depth_for_gs", 10000, depthCallback);

    std::string config_path;
    nh.param<std::string>("config_path", config_path, "");
    YAML::Node config_node = YAML::LoadFile(config_path);
    std::string result_path;
    nh.param<std::string>("result_path", result_path, "");
    std::string lpips_path;
    nh.param<std::string>("lpips_path", lpips_path, "");

    std::thread mapping_process(mapping, config_node, result_path, lpips_path);
    std::thread monitor_thread([](){
        while (!exit_flag)
        {
            double now = ros::WallTime::now().toSec();
            if (gaussians_initialized && (now - last_point_time > 5.0))
            {
                exit_flag = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    });
    
    ros::spin();

    mapping_process.join();
    monitor_thread.join();
    
    return 0;
}
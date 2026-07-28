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
#include <filesystem>
#include <iomanip>
#include <system_error>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

#ifndef GAUSSIAN_LIC_ENABLE_ONLINE_METRICS
#define GAUSSIAN_LIC_ENABLE_ONLINE_METRICS 0
#endif

#if GAUSSIAN_LIC_ENABLE_ONLINE_METRICS
#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#if __has_include(<c10/core/CachingDeviceAllocator.h>)
#include <c10/core/CachingDeviceAllocator.h>
#define GLIC_STAT_TYPE_NS c10::CachingDeviceAllocator
#else
#define GLIC_STAT_TYPE_NS c10::cuda::CUDACachingAllocator
#endif
#endif

namespace
{
void saveRunConfigCopy(const std::string& config_path, const std::string& result_path)
{
    if (config_path.empty() || result_path.empty())
    {
        return;
    }

    const std::filesystem::path source_path(config_path);
    const std::filesystem::path destination_path =
        std::filesystem::path(result_path) / "config.yaml";

    std::error_code ec;
    std::filesystem::create_directories(destination_path.parent_path(), ec);
    if (ec)
    {
        std::cerr << "[run config] Failed to create result directory: "
                  << ec.message() << std::endl;
        return;
    }

    std::filesystem::copy_file(source_path, destination_path,
                               std::filesystem::copy_options::overwrite_existing,
                               ec);
    if (ec)
    {
        std::cerr << "[run config] Failed to copy " << source_path
                  << " to " << destination_path << ": "
                  << ec.message() << std::endl;
        return;
    }

    std::cout << "[run config] Saved config copy to "
              << destination_path << std::endl;
}
}  // namespace

std::mutex m_buf;
std::condition_variable con;

std::queue<sensor_msgs::PointCloud2ConstPtr> point_buf;
std::queue<geometry_msgs::PoseStampedConstPtr> pose_buf;
std::queue<sensor_msgs::ImageConstPtr> image_buf;
std::queue<sensor_msgs::ImageConstPtr> depth_buf;

std::atomic<bool> exit_flag(false);
std::atomic<double> last_point_time(0.0);
std::atomic<bool> gaussians_initialized(false);

#if GAUSSIAN_LIC_ENABLE_ONLINE_METRICS
namespace
{
struct TorchGpuMemoryUsage
{
    size_t allocated_current = 0;
    size_t allocated_peak = 0;
    size_t reserved_current = 0;
    size_t reserved_peak = 0;
};

struct OnlineMetricsSample
{
    int sample_index = 0;
    int input_frame_idx = 0;
    int processed_frame_count = 0;
    int keyframe_count = 0;
    int test_frame_count = 0;
    int optimizer_iteration = 0;
    int optimizer_iterations_delta = 0;
    double elapsed_seconds = 0.0;
    double add_frame_seconds = 0.0;
    double initialize_seconds = 0.0;
    double extend_seconds = 0.0;
    double optimize_seconds = 0.0;
    double batch_seconds = 0.0;
    double total_adding_time_seconds = 0.0;
    double total_extending_time_seconds = 0.0;
    double total_mapping_time_seconds = 0.0;
    double processed_fps = 0.0;
    double keyframes_per_second = 0.0;
    double optimizer_iter_per_second = 0.0;
    double optimize_iter_per_second = 0.0;
    double num_gauss = 0.0;
    double scene_gauss = 0.0;
    double allocated_memory_mb = 0.0;
    double allocated_peak_memory_mb = 0.0;
    double reserved_memory_mb = 0.0;
    double reserved_peak_memory_mb = 0.0;
    double ram_usage_mb = 0.0;
};

TorchGpuMemoryUsage getTorchGpuMemoryUsage()
{
    TorchGpuMemoryUsage usage;
    if (!torch::cuda::is_available())
    {
        return usage;
    }

    namespace cudaAlloc = c10::cuda::CUDACachingAllocator;
    auto mem_stats = cudaAlloc::getDeviceStats(0);
    constexpr auto kAggregateStatIndex =
        static_cast<size_t>(GLIC_STAT_TYPE_NS::StatType::AGGREGATE);

    auto allocated_bytes = mem_stats.allocated_bytes[kAggregateStatIndex];
    usage.allocated_current = static_cast<size_t>(allocated_bytes.current);
    usage.allocated_peak = static_cast<size_t>(allocated_bytes.peak);

    auto reserved_bytes = mem_stats.reserved_bytes[kAggregateStatIndex];
    usage.reserved_current = static_cast<size_t>(reserved_bytes.current);
    usage.reserved_peak = static_cast<size_t>(reserved_bytes.peak);

    return usage;
}

double getCurrentRAMUsageMB()
{
    std::ifstream statm("/proc/self/statm");
    long pages = 0;
    long resident_pages = 0;
    if (!(statm >> pages >> resident_pages))
    {
        return 0.0;
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
    {
        return 0.0;
    }
    return static_cast<double>(resident_pages) * static_cast<double>(page_size) /
           (1024.0 * 1024.0);
}

double secondsSince(const std::chrono::steady_clock::time_point& start,
                    const std::chrono::steady_clock::time_point& end)
{
    return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
}

void writeMetricRow(std::ofstream& os,
                    const OnlineMetricsSample& sample,
                    const std::string& name,
                    double value)
{
    if (!std::isfinite(value))
    {
        value = 0.0;
    }
    os << sample.optimizer_iteration << ','
       << sample.elapsed_seconds << ','
       << sample.processed_frame_count << ','
       << "scalar" << ','
       << name << ','
       << value << ','
       << value << ','
       << value << ','
       << 1 << ','
       << value << '\n';
}

void writeOnlineMetricsCsv(const std::vector<OnlineMetricsSample>& samples,
                           const std::string& result_path)
{
    if (samples.empty() || result_path.empty())
    {
        return;
    }

    std::filesystem::create_directories(result_path);
    const std::filesystem::path metrics_path =
        std::filesystem::path(result_path) / "metrics.csv";
    std::ofstream os(metrics_path);
    os << std::fixed << std::setprecision(6);
    os << "iteration,elapsed_seconds,processed_frame_count,type,name,avg,min,max,samples,total\n";

    for (const OnlineMetricsSample& sample : samples)
    {
        writeMetricRow(os, sample, "input_frame_idx", sample.input_frame_idx);
        writeMetricRow(os, sample, "keyframes", sample.keyframe_count);
        writeMetricRow(os, sample, "test_frames", sample.test_frame_count);
        writeMetricRow(os, sample, "processed_fps", sample.processed_fps);
        writeMetricRow(os, sample, "keyframes_per_sec", sample.keyframes_per_second);
        writeMetricRow(os, sample, "iter_per_sec", sample.optimizer_iter_per_second);
        writeMetricRow(os, sample, "optimize_iter_per_sec", sample.optimize_iter_per_second);
        writeMetricRow(os, sample, "optimizer_iters_delta", sample.optimizer_iterations_delta);
        writeMetricRow(os, sample, "total_optimizer_iters", sample.optimizer_iteration);
        writeMetricRow(os, sample, "add_frame_ms", sample.add_frame_seconds * 1000.0);
        writeMetricRow(os, sample, "initialize_enqueue_ms", sample.initialize_seconds * 1000.0);
        writeMetricRow(os, sample, "extend_ms", sample.extend_seconds * 1000.0);
        writeMetricRow(os, sample, "optimize_ms", sample.optimize_seconds * 1000.0);
        writeMetricRow(os, sample, "mapping_batch_ms", sample.batch_seconds * 1000.0);
        writeMetricRow(os, sample, "total_adding_time_s", sample.total_adding_time_seconds);
        writeMetricRow(os, sample, "total_extending_time_s", sample.total_extending_time_seconds);
        writeMetricRow(os, sample, "total_mapping_time_s", sample.total_mapping_time_seconds);
        writeMetricRow(os, sample, "num_gauss", sample.num_gauss);
        writeMetricRow(os, sample, "scene_gauss", sample.scene_gauss);
        writeMetricRow(os, sample, "allocated_memory_mb", sample.allocated_memory_mb);
        writeMetricRow(os, sample, "allocated_peak_memory_mb", sample.allocated_peak_memory_mb);
        writeMetricRow(os, sample, "reserved_memory_mb", sample.reserved_memory_mb);
        writeMetricRow(os, sample, "reserved_peak_memory_mb", sample.reserved_peak_memory_mb);
        writeMetricRow(os, sample, "ram_usage_mb", sample.ram_usage_mb);
    }

    os.close();
    std::cout << "[online metrics] Saved " << samples.size()
              << " samples to " << metrics_path << std::endl;
}
}  // namespace
#undef GLIC_STAT_TYPE_NS
#endif

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
/// left as placeholders; depth-error columns are left empty here because
/// canonical depth metrics are computed from result/eval by JetFast eval_core.
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
    // results.
    os << "\n===== COPY TO EVALUATIONS SHEET =====\n";
    os << "<experiment_id>" << TAB          // Experiment
       << "GLIC2" << TAB;                   // Evaluator/renderer
    os << std::fixed << std::setprecision(2) << metrics.test_psnr << TAB;
    os << std::setprecision(3) << metrics.test_ssim << TAB
       << metrics.test_lpips << TAB
       << TAB                               // Test Depth MAE (canonical eval_core)
       << TAB;                              // Test Depth RMSE (canonical eval_core)
    os << std::setprecision(2) << metrics.train_psnr << TAB;
    os << std::setprecision(3) << metrics.train_ssim << TAB
       << metrics.train_lpips << TAB
       << TAB                               // Train Depth MAE (canonical eval_core)
       << "\n";                             // Train Depth RMSE (canonical eval_core)

    os.flags(saved_flags);
    os.precision(saved_precision);
}

void mapping(const YAML::Node& node, const std::string& config_path,
             const std::string& result_path, const std::string& lpips_path)
{
    torch::jit::setGraphExecutorOptimize(false);

    Params prm(node);
    std::shared_ptr<GaussianModel> gaussians = std::make_shared<GaussianModel>(prm);
    std::shared_ptr<Dataset> dataset = std::make_shared<Dataset>(prm);

    std::chrono::steady_clock::time_point t_start, t_end;
    double total_mapping_time = 0;
    double total_adding_time = 0;
    double total_extending_time = 0;
    int total_iters = 0;
#if GAUSSIAN_LIC_ENABLE_ONLINE_METRICS
    std::vector<OnlineMetricsSample> online_metrics;
    online_metrics.reserve(1024);
    bool online_start_set = false;
    std::chrono::steady_clock::time_point online_start;
    int last_logged_optimizer_iteration = 0;
    double last_logged_elapsed_seconds = 0.0;
#endif

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
#if GAUSSIAN_LIC_ENABLE_ONLINE_METRICS
        if (!online_start_set)
        {
            online_start = t_start;
            online_start_set = true;
        }
#endif
        dataset->addFrame(cur_frame);
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
#if GAUSSIAN_LIC_ENABLE_ONLINE_METRICS
        double add_frame_seconds = secondsSince(t_start, t_end);
#endif
        if (dataset->is_keyframe_current_)
        {
#if GAUSSIAN_LIC_ENABLE_ONLINE_METRICS
            total_adding_time += add_frame_seconds;
#else
            total_adding_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
#endif
            std::cout << "\033[1;33m     Cur Frame " << dataset->all_frame_num_ - 1 << ",\033[0m";
        }
        else continue;

#if GAUSSIAN_LIC_ENABLE_ONLINE_METRICS
        double initialize_seconds = 0.0;
        double extend_seconds = 0.0;
#endif
        if (!gaussians->is_init_)
        {
            /// [3] initialize map
#if GAUSSIAN_LIC_ENABLE_ONLINE_METRICS
            /// This records CPU enqueue/setup time only. We intentionally do not
            /// add a sync here because that would perturb the online path.
            t_start = std::chrono::steady_clock::now();
#endif
            gaussians->is_init_ = true;
            gaussians_initialized = true;
            gaussians->initialize(dataset);
            gaussians->trainingSetup();
#if GAUSSIAN_LIC_ENABLE_ONLINE_METRICS
            t_end = std::chrono::steady_clock::now();
            initialize_seconds = secondsSince(t_start, t_end);
#endif
        }
        else 
        {
            /// [4] extend map
            t_start = std::chrono::steady_clock::now();
            extend(dataset, gaussians);
            torch::cuda::synchronize();
            t_end = std::chrono::steady_clock::now();
#if GAUSSIAN_LIC_ENABLE_ONLINE_METRICS
            extend_seconds = secondsSince(t_start, t_end);
            total_extending_time += extend_seconds;
#else
            total_extending_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
#endif
        }

        /// [5] optimize map
#if GAUSSIAN_LIC_ENABLE_ONLINE_METRICS
        int optimizer_iteration_before = total_iters;
#endif
        t_start = std::chrono::steady_clock::now();
        double updated_num = optimize(dataset, gaussians, total_iters);
        torch::cuda::synchronize();
        t_end = std::chrono::steady_clock::now();
#if GAUSSIAN_LIC_ENABLE_ONLINE_METRICS
        double optimize_seconds = secondsSince(t_start, t_end);
        total_mapping_time += optimize_seconds;
#else
        total_mapping_time += std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start).count();
#endif
        std::cout << std::fixed << std::setprecision(2) 
                  << "\033[1;36m Update " << updated_num / 10000 
                  << "w GS per Iter \033[0m" << std::endl;

#if GAUSSIAN_LIC_ENABLE_ONLINE_METRICS
        const int optimizer_iterations_delta = total_iters - optimizer_iteration_before;
        const auto sample_time = std::chrono::steady_clock::now();
        const double elapsed_seconds =
            online_start_set ? secondsSince(online_start, sample_time) : 0.0;
        const double processed_fps =
            elapsed_seconds > 0.0 ? dataset->all_frame_num_ / elapsed_seconds : 0.0;
        const double keyframes_per_second =
            elapsed_seconds > 0.0 ? dataset->train_cameras_.size() / elapsed_seconds : 0.0;
        const int logged_optimizer_iterations_delta =
            total_iters - last_logged_optimizer_iteration;
        const double elapsed_delta = elapsed_seconds - last_logged_elapsed_seconds;
        const double optimizer_iter_per_second =
            elapsed_delta > 0.0 ? logged_optimizer_iterations_delta / elapsed_delta : 0.0;
        const double optimize_iter_per_second =
            optimize_seconds > 0.0 ? optimizer_iterations_delta / optimize_seconds : 0.0;
        const auto gpu_memory = getTorchGpuMemoryUsage();
        const double bytes_to_mb = 1.0 / (1024.0 * 1024.0);
        const double num_gauss = static_cast<double>(gaussians->getXYZ().size(0));

        OnlineMetricsSample sample;
        sample.sample_index = static_cast<int>(online_metrics.size());
        sample.input_frame_idx = dataset->all_frame_num_ - 1;
        sample.processed_frame_count = dataset->all_frame_num_;
        sample.keyframe_count = static_cast<int>(dataset->train_cameras_.size());
        sample.test_frame_count = static_cast<int>(dataset->test_cameras_.size());
        sample.optimizer_iteration = total_iters;
        sample.optimizer_iterations_delta = logged_optimizer_iterations_delta;
        sample.elapsed_seconds = elapsed_seconds;
        sample.add_frame_seconds = add_frame_seconds;
        sample.initialize_seconds = initialize_seconds;
        sample.extend_seconds = extend_seconds;
        sample.optimize_seconds = optimize_seconds;
        sample.batch_seconds = add_frame_seconds + initialize_seconds + extend_seconds + optimize_seconds;
        sample.total_adding_time_seconds = total_adding_time;
        sample.total_extending_time_seconds = total_extending_time;
        sample.total_mapping_time_seconds = total_mapping_time;
        sample.processed_fps = processed_fps;
        sample.keyframes_per_second = keyframes_per_second;
        sample.optimizer_iter_per_second = optimizer_iter_per_second;
        sample.optimize_iter_per_second = optimize_iter_per_second;
        sample.num_gauss = num_gauss;
        sample.scene_gauss = std::max(0.0, num_gauss - static_cast<double>(gaussians->skybox_points_num_));
        sample.allocated_memory_mb = gpu_memory.allocated_current * bytes_to_mb;
        sample.allocated_peak_memory_mb = gpu_memory.allocated_peak * bytes_to_mb;
        sample.reserved_memory_mb = gpu_memory.reserved_current * bytes_to_mb;
        sample.reserved_peak_memory_mb = gpu_memory.reserved_peak * bytes_to_mb;
        sample.ram_usage_mb = getCurrentRAMUsageMB();
        online_metrics.push_back(sample);
        last_logged_optimizer_iteration = total_iters;
        last_logged_elapsed_seconds = elapsed_seconds;
#endif
    }

#if GAUSSIAN_LIC_ENABLE_ONLINE_METRICS
    if (online_start_set &&
        (online_metrics.empty() ||
         online_metrics.back().processed_frame_count != dataset->all_frame_num_))
    {
        const auto sample_time = std::chrono::steady_clock::now();
        const double elapsed_seconds = secondsSince(online_start, sample_time);
        const double processed_fps =
            elapsed_seconds > 0.0 ? dataset->all_frame_num_ / elapsed_seconds : 0.0;
        const double keyframes_per_second =
            elapsed_seconds > 0.0 ? dataset->train_cameras_.size() / elapsed_seconds : 0.0;
        const int logged_optimizer_iterations_delta =
            total_iters - last_logged_optimizer_iteration;
        const double elapsed_delta = elapsed_seconds - last_logged_elapsed_seconds;
        const double optimizer_iter_per_second =
            elapsed_delta > 0.0 ? logged_optimizer_iterations_delta / elapsed_delta : 0.0;
        const auto gpu_memory = getTorchGpuMemoryUsage();
        const double bytes_to_mb = 1.0 / (1024.0 * 1024.0);
        const double num_gauss = static_cast<double>(gaussians->getXYZ().size(0));

        OnlineMetricsSample sample;
        sample.sample_index = static_cast<int>(online_metrics.size());
        sample.input_frame_idx = dataset->all_frame_num_ - 1;
        sample.processed_frame_count = dataset->all_frame_num_;
        sample.keyframe_count = static_cast<int>(dataset->train_cameras_.size());
        sample.test_frame_count = static_cast<int>(dataset->test_cameras_.size());
        sample.optimizer_iteration = total_iters;
        sample.optimizer_iterations_delta = logged_optimizer_iterations_delta;
        sample.elapsed_seconds = elapsed_seconds;
        sample.total_adding_time_seconds = total_adding_time;
        sample.total_extending_time_seconds = total_extending_time;
        sample.total_mapping_time_seconds = total_mapping_time;
        sample.processed_fps = processed_fps;
        sample.keyframes_per_second = keyframes_per_second;
        sample.optimizer_iter_per_second = optimizer_iter_per_second;
        sample.optimize_iter_per_second = 0.0;
        sample.num_gauss = num_gauss;
        sample.scene_gauss = std::max(0.0, num_gauss - static_cast<double>(gaussians->skybox_points_num_));
        sample.allocated_memory_mb = gpu_memory.allocated_current * bytes_to_mb;
        sample.allocated_peak_memory_mb = gpu_memory.allocated_peak * bytes_to_mb;
        sample.reserved_memory_mb = gpu_memory.reserved_current * bytes_to_mb;
        sample.reserved_peak_memory_mb = gpu_memory.reserved_peak * bytes_to_mb;
        sample.ram_usage_mb = getCurrentRAMUsageMB();
        online_metrics.push_back(sample);
    }
#endif

    /// [6] evaluation
    std::cout << "\n     🎉 Runtime Statistics 🎉\n";
    std::cout << std::fixed << std::setprecision(2) << "\n        [Total Mapping Time] " << total_mapping_time << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         1) Forward " << gaussians->t_forward_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         2) Backward " << gaussians->t_backward_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         3) Step " << gaussians->t_step_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "         4) CPU2GPU " << gaussians->t_tocuda_ << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "        [Total Adding Time] " << total_adding_time << "s" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "        [Total Extending Time] " << total_extending_time << "s" << std::endl;
    std::cout << "        [Total Optimize Iterations] " << total_iters << std::endl;
    torch::NoGradGuard no_grad;
    VisualQualityMetrics metrics = evaluateVisualQuality(dataset, gaussians, result_path, lpips_path);
    printSpreadsheetRows(gaussians, dataset, total_extending_time, metrics);
    gaussians->saveMap(result_path);
    saveFrameSequence(dataset, result_path);
    saveRunConfigCopy(config_path, result_path);
#if GAUSSIAN_LIC_ENABLE_ONLINE_METRICS
    writeOnlineMetricsCsv(online_metrics, result_path);
#endif

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

    std::thread mapping_process(mapping, config_node, config_path, result_path, lpips_path);
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

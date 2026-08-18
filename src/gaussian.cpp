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

#include "gaussian.h"
#include "tensor_utils.h"
#include "loss_utils.h"

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

#include <sstream>
#include <fstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <iterator>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <limits>
#include <torch/script.h>
#include <memory>
#include <map>
#include <stdexcept>

namespace fs = std::filesystem;

struct PixelPosition 
{
    int u, v;
};

namespace
{
int frameIndexFromImageName(const std::string& image_name)
{
    const size_t underscore = image_name.find('_');
    const size_t begin = underscore == std::string::npos ? 0 : underscore + 1;
    const size_t end = image_name.find('.', begin);
    const std::string digits = image_name.substr(begin, end - begin);
    if (digits.empty())
    {
        throw std::runtime_error("Cannot parse frame index from " + image_name);
    }
    return std::stoi(digits);
}

std::string evalStem(int frame_idx)
{
    std::stringstream ss;
    ss << std::setw(6) << std::setfill('0') << frame_idx;
    return ss.str();
}

torch::Tensor asHxW(torch::Tensor tensor)
{
    tensor = tensor.detach();
    while (tensor.dim() > 2 && tensor.size(0) == 1)
    {
        tensor = tensor.squeeze(0);
    }
    if (tensor.dim() == 3 && tensor.size(2) == 1)
    {
        tensor = tensor.squeeze(2);
    }
    if (tensor.dim() != 2)
    {
        throw std::runtime_error("Expected a HxW depth/alpha tensor");
    }
    return tensor.contiguous();
}

void saveRgbPng(torch::Tensor rgb_chw, const std::string& path)
{
    rgb_chw = rgb_chw.detach().clamp(0, 1).to(torch::kCPU).contiguous();
    const int H = static_cast<int>(rgb_chw.size(1));
    const int W = static_cast<int>(rgb_chw.size(2));
    torch::Tensor rgb_hwc = rgb_chw.permute({1, 2, 0}).mul(255).clamp(0, 255).to(torch::kU8).contiguous();
    cv::Mat rgb(H, W, CV_8UC3, rgb_hwc.data_ptr<uint8_t>());
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    cv::imwrite(path, bgr);
}

void saveFloatTiff(torch::Tensor tensor, const std::string& path)
{
    torch::Tensor cpu = asHxW(tensor).to(torch::kCPU).to(torch::kFloat32).contiguous();
    cv::Mat image(static_cast<int>(cpu.size(0)),
                  static_cast<int>(cpu.size(1)),
                  CV_32FC1,
                  cpu.data_ptr<float>());
    cv::imwrite(path, image);
}

void saveDepthPreviewPng(torch::Tensor depth, const std::string& path)
{
    torch::Tensor cpu = asHxW(depth).to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor finite = torch::isfinite(cpu);
    torch::Tensor preview = torch::zeros_like(cpu);
    if (finite.sum().item<int64_t>() > 0)
    {
        torch::Tensor values = cpu.masked_select(finite);
        const float min_depth = values.min().item<float>();
        const float max_depth = values.max().item<float>();
        if (max_depth > min_depth)
        {
            preview = (cpu - min_depth) / (max_depth - min_depth) * 255.0f;
            preview = torch::where(finite, preview, torch::zeros_like(preview));
        }
    }
    cv::Mat gray(static_cast<int>(preview.size(0)),
                 static_cast<int>(preview.size(1)),
                 CV_32FC1,
                 preview.data_ptr<float>());
    gray.convertTo(gray, CV_8UC1);
    cv::Mat color;
    cv::applyColorMap(gray, color, cv::COLORMAP_JET);
    cv::imwrite(path, color);
}

std::string cameraKey(const std::shared_ptr<Camera>& cam)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(9)
       << cam->image_width_ << ":"
       << cam->image_height_ << ":"
       << cam->fx_ << ":"
       << cam->fy_ << ":"
       << cam->cx_ << ":"
       << cam->cy_;
    return ss.str();
}

/// Camera geometry at full input resolution, for evaluation only.
struct EvalGeometry
{
    int width;
    int height;
    double inverse_scale;  // multiply training intrinsics by this
};

EvalGeometry evaluationGeometry(const std::shared_ptr<Dataset>& dataset)
{
    if (dataset->source_width_ <= 0 || dataset->source_height_ <= 0)
        throw std::runtime_error("evaluationGeometry: no frame was ingested, raw geometry unknown");

    const int width = dataset->source_width_;
    const int height = dataset->source_height_ - 2 * dataset->crop_y_;
    if (height <= 0)
        throw std::runtime_error("evaluationGeometry: crop_y removes every row");

    const double x_scale = static_cast<double>(dataset->target_width_) / width;
    const double y_scale = static_cast<double>(dataset->target_height_) / height;
    if (std::abs(x_scale - y_scale) > 1e-9)
        throw std::runtime_error(
            "evaluationGeometry: training resize is not uniform (x " + std::to_string(x_scale) +
            " vs y " + std::to_string(y_scale) + "); fix crop_y/width/height in the config");
    if (x_scale <= 0.0)
        throw std::runtime_error("evaluationGeometry: non-positive training scale");

    return EvalGeometry{width, height, 1.0 / x_scale};
}

}  // namespace

std::vector<PixelPosition> selectFromDepthCompletion(const cv::Mat& depth_A, const cv::Mat& depth_B, int patch_size = 20) 
{
    CV_Assert(depth_A.size() == depth_B.size());
    CV_Assert(depth_A.type() == depth_B.type());
    
    int H = depth_A.rows;
    int W = depth_A.cols;
    std::vector<PixelPosition> result;
    result.reserve((H / patch_size) * (W / patch_size));

    for (int i = 0; i < H; i += patch_size) 
    {
        for (int j = 0; j < W; j += patch_size) 
        {
            int h_end = std::min(i + patch_size, H);
            int w_end = std::min(j + patch_size, W);
            
            bool has_valid_A = false;
            bool has_valid_B = false;
            float min_val = std::numeric_limits<float>::max();
            PixelPosition min_pos;
            
            for (int y = i; y < h_end; ++y) 
            {
                const float* ptr_A = depth_A.ptr<float>(y);
                const float* ptr_B = depth_B.ptr<float>(y);
                
                for (int x = j; x < w_end; ++x) 
                {
                    if (ptr_A[x] > 0) 
                    {
                        has_valid_A = true;
                        y = h_end;
                        break;
                    }
                    
                    if (ptr_B[x] > 0) 
                    {
                        has_valid_B = true;
                        if (ptr_B[x] < min_val) 
                        {
                            min_val = ptr_B[x];
                            min_pos = {x, y};
                        }
                    }
                }
            }
            
            if (has_valid_A || !has_valid_B) 
            {
                continue;
            }
            
            result.push_back(min_pos);
        }
    }
    
    return result;
}

void Dataset::addFrame(Frame& cur_frame)
{
    /// image
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(cur_frame.image_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat image_bgr = cv_ptr->image;
    cv::Mat image_rgb;
    cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);  // 0-255
    image_rgb.convertTo(image_rgb, CV_32FC3, 1.0f / 255.0f);  // 0-1

    /// depth
    cv_bridge::CvImagePtr dp_ptr;
    dp_ptr = cv_bridge::toCvCopy(cur_frame.depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    cv::Mat depth_map = dp_ptr->image;  // metric float32

    /// Record the size of the incoming image once, before it is cropped or
    /// resized below. For evaluationGeometry() to recover full-res intrinsics.
    if (source_width_ == 0)
    {
        source_width_ = image_rgb.cols;
        source_height_ = image_rgb.rows;
    }

    /// Crop then resize to the target training resolution defined in the config.
    /// crop_y_ (pixels off each of top AND bottom) must be applied BEFORE the
    /// 0.5x resize so that the y scale factor stays identical to the x scale
    /// factor, keeping fy exact. Without the crop, a non-integer scale (e.g.
    /// 640/1296 = 0.4938) introduces ~1.25% error in fy which misplaces 3D
    /// points throughout the scene. With the crop (1296-16 = 1280 → 640 = 0.5x)
    /// both axes scale by exactly the same factor and all intrinsics halve cleanly.
    /// INTER_NEAREST for sparse depth avoids blending metric values with zero holes.
    if (image_rgb.cols != target_width_ || image_rgb.rows != target_height_)
    {
        if (crop_y_ > 0)
        {
            int src_h = image_rgb.rows;
            cv::Rect roi(0, crop_y_, image_rgb.cols, src_h - 2 * crop_y_);
            image_rgb = image_rgb(roi).clone();
            depth_map = depth_map(roi).clone();
        }
        cv::resize(image_rgb, image_rgb, cv::Size(target_width_, target_height_), 0, 0, cv::INTER_AREA);
        cv::resize(depth_map, depth_map, cv::Size(target_width_, target_height_), 0, 0, cv::INTER_NEAREST);
    }

    /// pose
    Eigen::Quaterniond q_wc;
    Eigen::Vector3d t_wc;
    tf::quaternionMsgToEigen(cur_frame.pose_msg->pose.orientation, q_wc);
    tf::pointMsgToEigen(cur_frame.pose_msg->pose.position, t_wc);
    R_wc_.push_back(q_wc.toRotationMatrix());
    t_wc_.push_back(t_wc);

    /// point
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(*cur_frame.point_msg, *cloud);
    Eigen::Matrix3d R_cw = q_wc.toRotationMatrix().transpose();
    Eigen::Vector3d t_cw = - R_cw * t_wc;
    for (size_t i = 0; i < cloud->points.size(); i += point_stride_)
    {
        const auto& pt = cloud->points[i];
        pointcloud_.emplace_back(Eigen::Vector3d(pt.x, pt.y, pt.z));
        pointcolor_.emplace_back(Eigen::Vector3d(pt.r, pt.g, pt.b) / 255.0);
        Eigen::Vector3d pt_c = R_cw * pointcloud_.back() + t_cw;
        assert(pt_c(2) > 0);
        pointdepth_.push_back(static_cast<float>(pt_c(2)));
    }

    /// train & test
    int width = image_rgb.cols, height = image_rgb.rows;
    if ((all_frame_num_ + 1) % select_every_k_frame_ == 0)
    {
        is_keyframe_current_ = true;
        std::shared_ptr<Camera> cam = std::make_shared<Camera>();

        if (depth_completion_)
        {
            cv::Mat completed_depth;  // metric float32
            completed_depth = depth_completer_.complete(image_rgb, depth_map);

            cv::Mat mask_known = depth_map > 0;  // 0/255 uint8
            cv::Mat completed_depth_known;
            completed_depth.copyTo(completed_depth_known, mask_known);
            cv::Mat depth_difference = completed_depth_known - depth_map;
            double mean_depth_difference = cv::mean(depth_difference, mask_known)[0];

            if (std::abs(mean_depth_difference) < 0.1)
            {
                // wanted_depth：non-edge && positive
                cv::Mat depth_gradient_x, depth_gradient_y;
                cv::Sobel(completed_depth, depth_gradient_x, CV_32F, 1, 0, 3);
                cv::Sobel(completed_depth, depth_gradient_y, CV_32F, 0, 1, 3);
                cv::Mat depth_edges;
                cv::magnitude(depth_gradient_x, depth_gradient_y, depth_edges);
                double edge_threshold = 0.1;
                cv::Mat mask_not_edges = depth_edges < edge_threshold;  // 0/255 uint8
                completed_depth -= mean_depth_difference;
                cv::Mat mask = (completed_depth > 0) & mask_not_edges;  // 0/255 uint8
                cv::Mat wanted_depth;
                completed_depth.copyTo(wanted_depth, mask);

                // select
                std::vector<PixelPosition> new_positions = selectFromDepthCompletion(depth_map, wanted_depth, patch_size_);
                for (const auto& pt : new_positions) 
                {
                    int u = pt.u, v = pt.v;
                    float depth = wanted_depth.at<float>(v, u);
                    assert(depth > 0);
                    if (depth > max_depth_) continue;

                    cv::Vec3f color = image_rgb.at<cv::Vec3f>(v, u);
                    Eigen::Vector3d eigen_color(color[0], color[1], color[2]);

                    Eigen::Vector3d cam_point((u - cx_) * depth / fx_, 
                                            (v - cy_) * depth / fy_, 
                                            depth);
                    Eigen::Vector3d world_point = q_wc * cam_point + t_wc;

                    pointcloud_.emplace_back(world_point);
                    pointcolor_.emplace_back(eigen_color);
                    pointdepth_.emplace_back(static_cast<float>(depth));
                }
            }
            else
            {
                // std::cout << "[bef vs aft diff]: " << mean_depth_difference << " m" << std::endl;
            }
        }

        cam->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(image_rgb, torch::kCPU, true);
        cam->original_depth_ = tensor_utils::cvMat2TorchTensor_Float32(depth_map, torch::kCPU, true);
        
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << all_frame_num_;
        std::string formatted_str = ss.str();
        cam->image_name_ = "train_" + formatted_str + ".png";
        cam->timestamp_ = cur_frame.image_msg->header.stamp.toSec();
        cam->rgb_timestamp_ns_ = static_cast<std::int64_t>(
            cur_frame.image_msg->header.stamp.toNSec());

        cam->setIntrinsic(width, height, fx_, fy_, cx_, cy_);
        cam->setPose(q_wc.toRotationMatrix(), t_wc);

        train_cameras_.emplace_back(cam);
    }
    else
    {
        is_keyframe_current_ = false;
        std::shared_ptr<Camera> cam = std::make_shared<Camera>();

        cam->original_image_ = tensor_utils::cvMat2TorchTensor_Float32(image_rgb, torch::kCPU);
        cam->original_depth_ = tensor_utils::cvMat2TorchTensor_Float32(depth_map, torch::kCPU);

        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << all_frame_num_;
        std::string formatted_str = ss.str();
        cam->image_name_ = "test_" + formatted_str + ".png";
        cam->timestamp_ = cur_frame.image_msg->header.stamp.toSec();
        cam->rgb_timestamp_ns_ = static_cast<std::int64_t>(
            cur_frame.image_msg->header.stamp.toNSec());

        cam->setIntrinsic(width, height, fx_, fy_, cx_, cy_);
        cam->setPose(q_wc.toRotationMatrix(), t_wc);

        test_cameras_.emplace_back(cam);
    }

    all_frame_num_ += 1;
}

GaussianModel::GaussianModel(const Params& prm)
{
    sh_degree_ = prm.sh_degree;
    white_background_ = prm.white_background;
    random_background_ = prm.random_background;
    convert_SHs_python_ = prm.convert_SHs_python;
    compute_cov3D_python_ = prm.compute_cov3D_python;
    lambda_erank_ = prm.lambda_erank;
    scaling_scale_ = prm.scaling_scale;

    position_lr_ = prm.position_lr;
    feature_lr_ = prm.feature_lr;
    opacity_lr_ = prm.opacity_lr;
    scaling_lr_ = prm.scaling_lr;
    rotation_lr_ = prm.rotation_lr;
    lambda_dssim_ = prm.lambda_dssim;
    optimize_depth_ = prm.optimize_depth;
    lambda_depth_ = prm.lambda_depth;
    iteration_decay_ = prm.iteration_decay;
    max_iters_ = prm.max_iters;

    apply_exposure_ = prm.apply_exposure;
    exposure_lr_ = prm.exposure_lr;
    skybox_points_num_ = prm.skybox_points_num;
    skybox_radius_ = prm.skybox_radius;

    auto device_type = torch::kCUDA;
    GAUSSIAN_MODEL_INIT_TENSORS(device_type)

    is_init_ = false;

    t_forward_ = 0;
    t_backward_ = 0;
    t_step_ = 0;
    t_optlist_ = 0;
    t_tocuda_ = 0;
}

torch::Tensor GaussianModel::getScaling()
{
    return torch::exp(scaling_);
}

torch::Tensor GaussianModel::getRotation()
{
    return torch::nn::functional::normalize(rotation_);
}

torch::Tensor GaussianModel::getXYZ()
{
    return xyz_;
}

torch::Tensor GaussianModel::getFeaturesDc()
{
    return features_dc_;
}

torch::Tensor GaussianModel::getFeaturesRest()
{
    return features_rest_;
}

torch::Tensor GaussianModel::getOpacity()
{
    return torch::sigmoid(opacity_);
}

torch::Tensor GaussianModel::getCovariance(int scaling_modifier)
{
    // build_rotation
    auto r = this->rotation_;
    auto R = general_utils::build_rotation(r);

    // build_scaling_rotation(scaling_modifier * scaling(Activation), rotation(_))
    auto s = scaling_modifier * this->getScaling();
    auto L = torch::zeros({s.size(0), 3, 3}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    L.select(1, 0).select(1, 0).copy_(s.index({torch::indexing::Slice(), 0}));
    L.select(1, 1).select(1, 1).copy_(s.index({torch::indexing::Slice(), 1}));
    L.select(1, 2).select(1, 2).copy_(s.index({torch::indexing::Slice(), 2}));
    L = R.matmul(L); // L = R @ L

    // build_covariance_from_scaling_rotation
    auto actual_covariance = L.matmul(L.transpose(1, 2));
    // strip_symmetric
    // strip_lowerdiag
    auto symm_uncertainty = torch::zeros({actual_covariance.size(0), 6}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    symm_uncertainty.select(1, 0).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 0}));
    symm_uncertainty.select(1, 1).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 1}));
    symm_uncertainty.select(1, 2).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 2}));
    symm_uncertainty.select(1, 3).copy_(actual_covariance.index({torch::indexing::Slice(), 1, 1}));
    symm_uncertainty.select(1, 4).copy_(actual_covariance.index({torch::indexing::Slice(), 1, 2}));
    symm_uncertainty.select(1, 5).copy_(actual_covariance.index({torch::indexing::Slice(), 2, 2}));

    return symm_uncertainty;
}

torch::Tensor GaussianModel::getExposure()
{
    return exposure_;
}

void GaussianModel::initialize(const std::shared_ptr<Dataset>& dataset)
{
    /// foreground
    int num = static_cast<int>(dataset->pointcloud_.size());
    assert(num > 0);
    torch::Tensor fused_point_cloud = torch::zeros({num, 3}, torch::kFloat32).cuda();  // (n, 3)
    int deg_2 = (sh_degree_ + 1) * (sh_degree_ + 1);
    torch::Tensor features = torch::zeros({num, 3, deg_2}, torch::kFloat32).cuda();  // (n, 3, 16)
    torch::Tensor scales = torch::zeros({num}, torch::kFloat32).cuda();

    double f = (dataset->fx_ + dataset->fy_) / 2;
    for (int i = 0; i < num; ++i) 
    {
        auto& pt_w = dataset->pointcloud_[i];
        auto& color = dataset->pointcolor_[i];
        fused_point_cloud.index({i, 0}) = pt_w.x();
        fused_point_cloud.index({i, 1}) = pt_w.y();
        fused_point_cloud.index({i, 2}) = pt_w.z();
        features.index({i, 0, 0}) = RGB2SH(color.x());
        features.index({i, 1, 0}) = RGB2SH(color.y());
        features.index({i, 2, 0}) = RGB2SH(color.z());

        double d = dataset->pointdepth_[i];
        scales.index({i}) = std::log(scaling_scale_ * d / f);
    }
    scales = scales.unsqueeze(1).repeat({1, 3});  // (n, 3)
    torch::Tensor rots = torch::zeros({num, 4}, torch::kFloat32).cuda();  // (n, 4)
    rots.index({torch::indexing::Slice(), 0}) = 1;
    torch::Tensor opacities = general_utils::inverse_sigmoid(0.1f * torch::ones({num, 1}, torch::kFloat32).cuda());  // (n, 1)

    /// sky
    if (skybox_points_num_ > 0)
    {
        int num = skybox_points_num_;
        double radius = skybox_radius_;
        torch::Tensor pi = torch::acos(torch::tensor(-1.0, torch::kFloat32).cuda());
        torch::Tensor theta = 2.0 * pi * torch::rand({num}, torch::kFloat32).cuda();
        torch::Tensor phi = torch::acos(1.0 - 1.4 * torch::rand({num}, torch::kFloat32).cuda());
        torch::Tensor sky_fused_point_cloud = torch::zeros({num, 3}, torch::kFloat32).cuda();
        sky_fused_point_cloud.index({torch::indexing::Slice(), 0}) = radius * 10 * torch::cos(theta) * torch::sin(phi);
        sky_fused_point_cloud.index({torch::indexing::Slice(), 1}) = radius * 10 * torch::sin(theta) * torch::sin(phi);
        sky_fused_point_cloud.index({torch::indexing::Slice(), 2}) = radius * 10 * torch::cos(phi);

        torch::Tensor sky_features = torch::zeros({num, 3, deg_2}, torch::kFloat32).cuda();
        sky_features.index({torch::indexing::Slice(), 0, 0}) = 0.7;
        sky_features.index({torch::indexing::Slice(), 1, 0}) = 0.8;
        sky_features.index({torch::indexing::Slice(), 2, 0}) = 0.95;

        torch::Tensor point_cloud_copy = sky_fused_point_cloud.clone();
        torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
        torch::Tensor sky_scales = torch::log(torch::sqrt(dist2));
        sky_scales = sky_scales.unsqueeze(1).repeat({1, 3});
        torch::Tensor sky_rots = torch::zeros({num, 4}, torch::kFloat32).cuda();
        sky_rots.index({torch::indexing::Slice(), 0}) = 1;
        torch::Tensor sky_opacities = general_utils::inverse_sigmoid(0.7f * torch::ones({num, 1}, torch::kFloat32).cuda());

        fused_point_cloud = torch::cat({sky_fused_point_cloud, fused_point_cloud}, 0);
        features = torch::cat({sky_features, features}, 0);
        scales = torch::cat({sky_scales, scales}, 0);
        rots = torch::cat({sky_rots, rots}, 0);
        opacities = torch::cat({sky_opacities, opacities}, 0);
    }

    this->xyz_ = fused_point_cloud.requires_grad_();  // (n, 3)
    // this->xyz_ = fused_point_cloud.requires_grad_(false);  // fix xyz
    this->features_dc_ = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous().requires_grad_();  // (n, 1, 3)
    this->features_rest_ = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(1, features.size(2))}).transpose(1, 2).contiguous().requires_grad_();  // (n, 15, 3)
    this->scaling_ = scales.requires_grad_();  // (n, 3)
    this->rotation_ = rots.requires_grad_();  // (n, 4)
    this->opacity_ = opacities.requires_grad_();  // (n, 1)

    if (apply_exposure_)
    {
        torch::Tensor exposure = torch::eye(3, torch::kFloat32).cuda();
        exposure = torch::cat({exposure, torch::zeros({3, 1}, torch::kFloat32).cuda()}, 1);
        this->exposure_ = exposure.requires_grad_();  // (3, 4)
    }

    GAUSSIAN_MODEL_TENSORS_TO_VEC
    
    std::cout << std::fixed << std::setprecision(2) 
              << "\033[1;37m Init Map with " 
              << double(fused_point_cloud.size(0)) / 10000 << "w GS" 
              << ",\033[0m";

    dataset->pointcloud_.clear();
    dataset->pointcolor_.clear();
    dataset->pointdepth_.clear();
}

void GaussianModel::saveMap(const std::string& result_path)
{
    std::string pc_path = result_path + "/point_cloud.ply";

    torch::Tensor xyz = this->xyz_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();
    // torch::Tensor normals = torch::zeros_like(xyz);
    torch::Tensor f_dc = this->features_dc_.index({torch::indexing::Slice(skybox_points_num_)}).detach().transpose(1, 2).flatten(1).contiguous().cpu();
    torch::Tensor f_rest = this->features_rest_.index({torch::indexing::Slice(skybox_points_num_)}).detach().transpose(1, 2).flatten(1).contiguous().cpu();
    torch::Tensor opacities = this->opacity_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();
    torch::Tensor scale = this->scaling_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();
    torch::Tensor rotation = this->rotation_.index({torch::indexing::Slice(skybox_points_num_)}).detach().cpu();

    std::filebuf fb_binary;
    fb_binary.open(pc_path, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);

    tinyply::PlyFile result_file;

    // xyz
    result_file.add_properties_to_element(
        "vertex", {"x", "y", "z"},
        tinyply::Type::FLOAT32, xyz.size(0),
        reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // // normals
    // result_file.add_properties_to_element(
    //     "vertex", {"nx", "ny", "nz"},
    //     tinyply::Type::FLOAT32, normals.size(0),
    //     reinterpret_cast<uint8_t*>(normals.data_ptr<float>()),
    //     tinyply::Type::INVALID, 0);

    // f_dc
    std::size_t n_f_dc = this->features_dc_.size(1) * this->features_dc_.size(2);
    std::vector<std::string> property_names_f_dc(n_f_dc);
    for (int i = 0; i < n_f_dc; ++i)
        property_names_f_dc[i] = "f_dc_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_f_dc,
        tinyply::Type::FLOAT32, this->features_dc_.size(0),
        reinterpret_cast<uint8_t*>(f_dc.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // f_rest
    std::size_t n_f_rest = this->features_rest_.size(1) * this->features_rest_.size(2);
    std::vector<std::string> property_names_f_rest(n_f_rest);
    for (int i = 0; i < n_f_rest; ++i)
        property_names_f_rest[i] = "f_rest_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_f_rest,
        tinyply::Type::FLOAT32, this->features_rest_.size(0),
        reinterpret_cast<uint8_t*>(f_rest.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // opacities
    result_file.add_properties_to_element(
        "vertex", {"opacity"},
        tinyply::Type::FLOAT32, opacities.size(0),
        reinterpret_cast<uint8_t*>(opacities.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // scale
    std::size_t n_scale = scale.size(1);
    std::vector<std::string> property_names_scale(n_scale);
    for (int i = 0; i < n_scale; ++i)
        property_names_scale[i] = "scale_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_scale,
        tinyply::Type::FLOAT32, scale.size(0),
        reinterpret_cast<uint8_t*>(scale.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // rotation
    std::size_t n_rotation = rotation.size(1);
    std::vector<std::string> property_names_rotation(n_rotation);
    for (int i = 0; i < n_rotation; ++i)
        property_names_rotation[i] = "rot_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_rotation,
        tinyply::Type::FLOAT32, rotation.size(0),
        reinterpret_cast<uint8_t*>(rotation.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // Write the file
    result_file.write(outstream_binary, true);

    fb_binary.close();
}

void GaussianModel::trainingSetup()
{
    this->sparse_optimizer_.reset(new SparseGaussianAdam(Tensor_vec_xyz_, 0.0, 1e-15));
    sparse_optimizer_->param_groups()[0].options().set_lr(position_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_feature_dc_);
    sparse_optimizer_->param_groups()[1].options().set_lr(feature_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_feature_rest_);
    sparse_optimizer_->param_groups()[2].options().set_lr(feature_lr_ / 20.0);

    sparse_optimizer_->add_param_group(Tensor_vec_opacity_);
    sparse_optimizer_->param_groups()[3].options().set_lr(opacity_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_scaling_);
    sparse_optimizer_->param_groups()[4].options().set_lr(scaling_lr_);

    sparse_optimizer_->add_param_group(Tensor_vec_rotation_);
    sparse_optimizer_->param_groups()[5].options().set_lr(rotation_lr_);

    if (apply_exposure_)
    {
        this->exposure_optimizer_.reset(new torch::optim::Adam(Tensor_vec_exposure_, {}));
        exposure_optimizer_->param_groups()[0].options().set_lr(exposure_lr_);
    }
}

void GaussianModel::densificationPostfix(
    torch::Tensor& new_xyz,
    torch::Tensor& new_features_dc,
    torch::Tensor& new_features_rest,
    torch::Tensor& new_opacities,
    torch::Tensor& new_scaling,
    torch::Tensor& new_rotation)
{
    std::vector<torch::Tensor> optimizable_tensors(6);
    std::vector<torch::Tensor> tensors_dict = 
    {
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation
    };
    auto& param_groups = this->sparse_optimizer_->param_groups();
    auto& optimizer_state = this->sparse_optimizer_->get_state();

    for (int group_idx = 0; group_idx < 6; ++group_idx) 
    {
        auto& group = param_groups[group_idx];
        assert(group.params().size() == 1);
        auto& extension_tensor = tensors_dict[group_idx];
        auto& param = group.params()[0];

        auto old_param_impl = param.unsafeGetTensorImpl();

        param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
        // if (group_idx == 0) param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_(false);  // fix xyz
        // else param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();  // fix xyz
        group.params()[0] = param;

        auto new_param_impl = param.unsafeGetTensorImpl();

        auto state_it = optimizer_state.find(old_param_impl);
        if (state_it != optimizer_state.end()) 
        {
            auto stored_state = state_it->second;

            stored_state.exp_avg = torch::cat({stored_state.exp_avg.clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0);
            stored_state.exp_avg_sq = torch::cat({stored_state.exp_avg_sq.clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0);

            optimizer_state.erase(state_it);

            optimizer_state[new_param_impl] = stored_state;
        }
        else 
        {
            State new_state;
            new_state.step = 0;
            new_state.exp_avg = torch::zeros_like(param, torch::MemoryFormat::Preserve);
            new_state.exp_avg_sq = torch::zeros_like(param, torch::MemoryFormat::Preserve);
            new_state.initialized = true;

            optimizer_state[new_param_impl] = new_state;
        }

        optimizable_tensors[group_idx] = param;
    }

    this->xyz_ = optimizable_tensors[0];
    this->features_dc_ = optimizable_tensors[1];
    this->features_rest_ = optimizable_tensors[2];
    this->opacity_ = optimizable_tensors[3];
    this->scaling_ = optimizable_tensors[4];
    this->rotation_ = optimizable_tensors[5];

    GAUSSIAN_MODEL_TENSORS_TO_VEC
}

void extend(const std::shared_ptr<Dataset>& dataset, std::shared_ptr<GaussianModel>& pc)
{
    torch::NoGradGuard no_grad;
    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();
    std::shared_ptr<Camera> viewpoint_cam = dataset->train_cameras_.back();
    auto render_pkg = render(viewpoint_cam, pc, bg, pc->apply_exposure_, true);
    auto rendered_alpha = 1 - std::get<2>(render_pkg).squeeze(0);

    int n = dataset->pointcloud_.size();
    std::vector<float> float_point(n * 3);
    std::vector<float> float_color(n * 3);
    for (size_t i = 0; i < n; ++i) 
    {
        float_point[3 * i + 0] = static_cast<float>(dataset->pointcloud_[i][0]);
        float_point[3 * i + 1] = static_cast<float>(dataset->pointcloud_[i][1]);
        float_point[3 * i + 2] = static_cast<float>(dataset->pointcloud_[i][2]);
        float_color[3 * i + 0] = static_cast<float>(dataset->pointcolor_[i][0]);
        float_color[3 * i + 1] = static_cast<float>(dataset->pointcolor_[i][1]);
        float_color[3 * i + 2] = static_cast<float>(dataset->pointcolor_[i][2]);
    }
    torch::Tensor points = torch::from_blob(float_point.data(), {n, 3}).to(torch::kFloat32).cuda();
    torch::Tensor colors = torch::from_blob(float_color.data(), {n, 3}).to(torch::kFloat32).cuda();
    torch::Tensor depths_in_rsp_frame = torch::from_blob(dataset->pointdepth_.data(), {n}).to(torch::kFloat32).cuda();

    /// filter
    auto R_wc = dataset->R_wc_.back();
    auto t_wc = dataset->t_wc_.back();
    auto R_cw = R_wc.transpose();
    auto t_cw = - R_cw * t_wc;
    std::vector<float> float_R_cw(3 * 3);
    std::vector<float> float_t_cw(3);
    for (size_t i = 0; i < 3; ++i)
    {
        float_R_cw[3 * i + 0] = static_cast<float>(R_cw(i, 0));
        float_R_cw[3 * i + 1] = static_cast<float>(R_cw(i, 1));
        float_R_cw[3 * i + 2] = static_cast<float>(R_cw(i, 2));
        float_t_cw[i] = static_cast<float>(t_cw[i]);
    }
    torch::Tensor R_cw_tensor = torch::from_blob(float_R_cw.data(), {3, 3}).to(torch::kFloat32).cuda();
    torch::Tensor t_cw_tensor = torch::from_blob(float_t_cw.data(), {3, 1}).to(torch::kFloat32).cuda();
    auto points_camera = torch::matmul(points, R_cw_tensor.t()) + t_cw_tensor.view({1, 3});  // (n, 3)
    auto depths = points_camera.index({torch::indexing::Slice(), 2});  // (n)
    float fx = static_cast<float>(viewpoint_cam->fx_);
    float fy = static_cast<float>(viewpoint_cam->fy_);
    float cx = static_cast<float>(viewpoint_cam->cx_);
    float cy = static_cast<float>(viewpoint_cam->cy_);
    float focal = (fx + fy) / 2.0;
    torch::Tensor x_pixel = (points_camera.index({torch::indexing::Slice(), 0}) * fx) / depths + cx;
    torch::Tensor y_pixel = (points_camera.index({torch::indexing::Slice(), 1}) * fy) / depths + cy;
    auto pixels = torch::stack({x_pixel, y_pixel}, 1);  // (n, 2)
    pixels = pixels.floor().to(torch::kInt32);

    auto pixels_float = pixels.to(torch::kFloat32);
    auto pixels_with_depth = torch::cat({pixels_float, depths.unsqueeze(1)}, 1).to(torch::kCPU);
    auto pixels_depth_a = pixels_with_depth.accessor<float, 2>();

    std::unordered_map<std::string, std::pair<int, float>> pixel_depth_map;
    for (int i = 0; i < pixels_with_depth.size(0); ++i) {
        int x = static_cast<int>(pixels_depth_a[i][0]);
        int y = static_cast<int>(pixels_depth_a[i][1]);
        float depth = pixels_depth_a[i][2];
        
        std::string key = std::to_string(x) + "_" + std::to_string(y);
        if (!pixel_depth_map.count(key) || depth < pixel_depth_map[key].second) {
            pixel_depth_map[key] = {i, depth};
        }
    }

    std::vector<int64_t> keep_indices;
    for (const auto& item : pixel_depth_map) {
        keep_indices.push_back(item.second.first);
    }

    auto keep_indices_tensor = torch::from_blob(
        keep_indices.data(), 
        {static_cast<int64_t>(keep_indices.size())}, 
        torch::kInt64
    ).to(points.device());
    auto filtered_points = points.index_select(0, keep_indices_tensor);
    auto filtered_colors = colors.index_select(0, keep_indices_tensor);
    auto filtered_depths_in_rsp_frame = depths_in_rsp_frame.index_select(0, keep_indices_tensor);
    auto filtered_pixels = pixels.index_select(0, keep_indices_tensor);

    int H = viewpoint_cam->image_height_, W = viewpoint_cam->image_width_;
    auto filter = [H, W, &rendered_alpha](const torch::Tensor& points, 
                                        const torch::Tensor& colors, 
                                        const torch::Tensor& depths_in_rsp_frame, 
                                        const torch::Tensor& pixels) 
    {
        auto in_image = (pixels.index({torch::indexing::Slice(), 0}) >= 0) & 
                        (pixels.index({torch::indexing::Slice(), 0}) < W) &
                        (pixels.index({torch::indexing::Slice(), 1}) >= 0) & 
                        (pixels.index({torch::indexing::Slice(), 1}) < H);  // (n) bool
        
        auto positive_depth = depths_in_rsp_frame > 0;

        auto x_coords = pixels.index({torch::indexing::Slice(), 0}).clamp(0, W - 1);
        auto y_coords = pixels.index({torch::indexing::Slice(), 1}).clamp(0, H - 1);
        auto opaque = rendered_alpha.index({y_coords, x_coords}) < 0.99;  // (n) bool

        auto valid_flag = torch::logical_and(torch::logical_and(in_image, positive_depth), opaque);
        auto filtered_points = points.index({valid_flag, torch::indexing::Slice()});
        auto filtered_colors = colors.index({valid_flag, torch::indexing::Slice()});
        auto filtered_depths = depths_in_rsp_frame.index({valid_flag});
        return std::make_tuple(filtered_points, filtered_colors, filtered_depths);
    };

    // auto filtered_pkg = filter(points, colors, depths_in_rsp_frame, pixels);
    auto filtered_pkg = filter(filtered_points, filtered_colors, filtered_depths_in_rsp_frame, filtered_pixels);
    
    /// densification
    torch::Tensor fused_point_cloud = std::get<0>(filtered_pkg);  // (n, 3)
    torch::Tensor fused_color = RGB2SH(std::get<1>(filtered_pkg));
    int num = fused_point_cloud.size(0);
    int deg_2 = (pc->sh_degree_ + 1) * (pc->sh_degree_ + 1);
    torch::Tensor features = torch::zeros({num, 3, deg_2}, torch::kFloat32).cuda();  // (n, 3, 16)
    features.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}) = fused_color;
    torch::Tensor features_dc = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous();  // (n, 1, 3)
    torch::Tensor features_rest = features.index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                          torch::indexing::Slice(1, features.size(2))}).transpose(1, 2).contiguous();  // (n, 15, 3)
    torch::Tensor scales = torch::log(pc->scaling_scale_ * std::get<2>(filtered_pkg) / focal).unsqueeze(1).repeat({1, 3});  // (n, 3)
    torch::Tensor rots = torch::zeros({num, 4}, torch::kFloat32).cuda();  // (n, 4)
    rots.index({torch::indexing::Slice(), 0}) = 1;
    torch::Tensor opacities = general_utils::inverse_sigmoid(0.1f * torch::ones({num, 1}, torch::kFloat32).cuda());  // (n, 1)

    pc->densificationPostfix(fused_point_cloud, features_dc, features_rest, opacities, scales, rots);

    std::cout << std::fixed << std::setprecision(2) 
              << "\033[1;32m Insert " << double(fused_point_cloud.size(0)) / 1000 
              << "k GS" << ",\033[0m";

    dataset->pointcloud_.clear();
    dataset->pointcolor_.clear();
    dataset->pointdepth_.clear();
}

void decayOptList(int max_iters, const int train_camera_num, 
                  const std::shared_ptr<Dataset>& dataset, const std::vector<int>& all_list, std::vector<int>& opt_list)
{
    Eigen::Vector3d t0 = dataset->t_wc_[0];
    double dist = (dataset->t_wc_.back() - t0).norm();
    if (dist > 120)
    {
        max_iters /= 2;
        opt_list.clear();
        std::random_device rd;
        std::mt19937 gen(rd());
        int split = train_camera_num * 2 / 3;
        int half = max_iters / 2;
        std::sample(all_list.begin(), all_list.begin() + split,
                    std::back_inserter(opt_list), std::min(half, split), gen);
        std::sample(all_list.begin() + split, all_list.end(),
                    std::back_inserter(opt_list), std::min(half, train_camera_num - split), gen);
    }
}

double optimize(const std::shared_ptr<Dataset>& dataset, std::shared_ptr<GaussianModel>& pc, int& total_iters)
{
    pc->t_start_ = std::chrono::steady_clock::now();
    int updated_num = 0;
    std::vector<int> opt_list;
    int max_iters = pc->max_iters_;

    int train_camera_num = dataset->train_cameras_.size();
    std::vector<int> all_list(train_camera_num);
    std::iota(all_list.begin(), all_list.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    if (train_camera_num <= max_iters) 
    {
        opt_list = all_list;
    }
    else
    {
        std::sample(all_list.begin(), all_list.end(), 
                    std::back_inserter(opt_list), max_iters, gen);
    } 
    if (pc->iteration_decay_) decayOptList(max_iters, train_camera_num, dataset, all_list, opt_list);
    std::shuffle(opt_list.begin(), opt_list.end(), gen);
    torch::cuda::synchronize();
    pc->t_end_ = std::chrono::steady_clock::now();
    pc->t_optlist_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();

    pc->t_start_ = std::chrono::steady_clock::now();
    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();
    torch::cuda::synchronize();
    pc->t_end_ = std::chrono::steady_clock::now();
    pc->t_tocuda_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
    for (int idx : opt_list)
    {
        pc->t_start_ = std::chrono::steady_clock::now();
        const std::shared_ptr<Camera>& viewpoint_cam = dataset->train_cameras_[idx];
        auto gt_image = viewpoint_cam->original_image_.to(torch::kCUDA, /*non_blocking=*/true);
        auto gt_depth = viewpoint_cam->original_depth_.to(torch::kCUDA, /*non_blocking=*/true);
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_tocuda_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
        pc->t_start_ = std::chrono::steady_clock::now();
        auto render_pkg = render(viewpoint_cam, pc, bg, pc->apply_exposure_);
        auto rendered_image = std::get<0>(render_pkg);
        auto rendered_depth = std::get<1>(render_pkg);
        auto mask = (gt_depth > 0) & (rendered_depth > 0);
        auto Ll1 = loss_utils::l1_loss(rendered_image, gt_image);
        auto Ll1_depth = torch::abs(rendered_depth.masked_select(mask) - gt_depth.masked_select(mask)).mean();
        float lambda_dssim = pc->lambda_dssim_;
        float lambda_depth = pc->lambda_depth_;
        torch::Tensor ssim_value;
        torch::Tensor rendered_image_unsq = rendered_image.unsqueeze(0);
        torch::Tensor gt_image_unsq = gt_image.unsqueeze(0);
        ssim_value = loss_utils::fused_ssim(rendered_image_unsq, gt_image_unsq);
        auto loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value);
        if (pc->optimize_depth_) loss += lambda_depth * Ll1_depth;
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_forward_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
        
        pc->t_start_ = std::chrono::steady_clock::now();
        loss.backward();
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_backward_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();

        pc->t_start_ = std::chrono::steady_clock::now();
        auto visible = std::get<4>(render_pkg);
        updated_num += visible.sum().item<int>();
        pc->sparse_optimizer_->set_visibility_and_N(visible, pc->getXYZ().size(0));
        pc->sparse_optimizer_->step();
        pc->sparse_optimizer_->zero_grad(true);
        if (pc->apply_exposure_)
        {
            pc->exposure_optimizer_->step();
            pc->exposure_optimizer_->zero_grad(true);
        }
        torch::cuda::synchronize();
        pc->t_end_ = std::chrono::steady_clock::now();
        pc->t_step_ += std::chrono::duration_cast<std::chrono::duration<double>>(pc->t_end_ - pc->t_start_).count();
    }

    total_iters += opt_list.size();
    return updated_num / opt_list.size();
}

VisualQualityMetrics evaluateVisualQuality(const std::shared_ptr<Dataset>& dataset,
                           std::shared_ptr<GaussianModel>& pc,
                           const std::string& result_path,
                           const std::string& lpips_path)
{
    VisualQualityMetrics metrics;
    std::cout << "\n     🎉 Evaluate Visual Quality 🎉\n";
    std::cout << "\n        [Number of Final Gaussians] " << pc->getXYZ().size(0) << std::endl;

    if (fs::exists(result_path))
        for (auto& entry : fs::directory_iterator(result_path))
            fs::remove_all(entry.path());
    fs::create_directories(result_path);

    std::string render_dir_path = result_path + "/render";
    fs::create_directories(render_dir_path);
    std::string render_depth_dir_path = result_path + "/render_depth";
    fs::create_directories(render_depth_dir_path);
    std::string gt_dir_path = result_path + "/gt";
    fs::create_directories(gt_dir_path);
    std::string eval_gt_rgb_dir = result_path + "/eval/gt/rgb";
    std::string eval_gt_depth_dir = result_path + "/eval/gt/depth";
    std::string eval_render_rgb_dir = result_path + "/eval/renders/rgb";
    std::string eval_render_depth_dir = result_path + "/eval/renders/depth";
    std::string eval_render_alpha_dir = result_path + "/eval/renders/alpha";
    fs::create_directories(eval_gt_rgb_dir);
    fs::create_directories(eval_gt_depth_dir);
    fs::create_directories(eval_render_rgb_dir);
    fs::create_directories(eval_render_depth_dir);
    fs::create_directories(eval_render_alpha_dir);

    torch::Tensor bg;
    if (pc->white_background_) bg = torch::ones({3}, torch::kFloat32).cuda();
    else bg = torch::zeros({3}, torch::kFloat32).cuda();
    torch::jit::script::Module m_lpips;
    bool lpips_loaded = false;
    try 
    {
        m_lpips = torch::jit::load(lpips_path + "/lpips_alex.pt");
        m_lpips.to(torch::kCUDA);
        lpips_loaded = true;
    }
    catch (const c10::Error& e) 
    {
        std::cerr << "lpips model loading failed: " << e.what() << std::endl;
    }
    if (!lpips_loaded)
    {
        throw std::runtime_error("LPIPS model loading failed; cannot evaluate visual quality.");
    }

    struct SplitAccum
    {
        double psnrs = 0;
        double ssims = 0;
        double lpipss = 0;
        int rgb_n = 0;
    };

    const auto average = [](double sum, int n) {
        return n > 0 ? sum / static_cast<double>(n) : 0.0;
    };

    const auto evaluate_split =
        [&](const std::vector<std::shared_ptr<Camera>>& cameras,
            const std::string& name) -> SplitAccum
        {
            SplitAccum accum;
            for (const auto& camera : cameras)
            {
                auto render_pkg = render(camera, pc, bg, pc->apply_exposure_);
                auto rendered_image = std::get<0>(render_pkg).clamp(0, 1);
                auto rendered_depth = std::get<1>(render_pkg);
                auto rendered_final_T = std::get<2>(render_pkg);
                auto gt_image = camera->original_image_.cuda().clamp(0, 1);
                double psnr = loss_utils::psnr(rendered_image, gt_image).mean().item<double>();
                double ssim = loss_utils::ssim(rendered_image, gt_image).item<double>();
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(rendered_image.unsqueeze(0));
                inputs.push_back(gt_image.unsqueeze(0));
                double lpips = m_lpips.forward(inputs).toTensor().item<double>();
                accum.psnrs += psnr;
                accum.ssims += ssim;
                accum.lpipss += lpips;
                accum.rgb_n += 1;

                const int frame_idx = frameIndexFromImageName(camera->image_name_);
                const std::string stem = evalStem(frame_idx);
                saveRgbPng(rendered_image, render_dir_path + "/" + camera->image_name_);
                saveRgbPng(gt_image, gt_dir_path + "/" + camera->image_name_);
                saveRgbPng(rendered_image, eval_render_rgb_dir + "/" + stem + ".png");
                saveRgbPng(gt_image, eval_gt_rgb_dir + "/" + stem + ".png");
                saveFloatTiff(rendered_depth, eval_render_depth_dir + "/" + stem + ".tiff");
                saveFloatTiff(camera->original_depth_, eval_gt_depth_dir + "/" + stem + ".tiff");
                saveFloatTiff((torch::ones_like(rendered_final_T) - rendered_final_T).clamp(0, 1),
                              eval_render_alpha_dir + "/" + stem + ".tiff");
                saveDepthPreviewPng(rendered_depth, render_depth_dir_path + "/" + camera->image_name_);
            }
            std::cout << std::fixed << std::setprecision(2)
                      << "        [" << name << " PSNR] " << average(accum.psnrs, accum.rgb_n) << std::endl;
            std::cout << std::fixed << std::setprecision(3)
                      << "        [" << name << " SSIM] " << average(accum.ssims, accum.rgb_n) << std::endl;
            std::cout << std::fixed << std::setprecision(3)
                      << "        [" << name << " LPIPS] " << average(accum.lpipss, accum.rgb_n) << std::endl;
            return accum;
        };

    SplitAccum train = evaluate_split(dataset->train_cameras_, "Training View");
    metrics.train_psnr = average(train.psnrs, train.rgb_n);
    metrics.train_ssim = average(train.ssims, train.rgb_n);
    metrics.train_lpips = average(train.lpipss, train.rgb_n);

    SplitAccum test = evaluate_split(dataset->test_cameras_, "In-Sequence Novel View");
    metrics.test_psnr = average(test.psnrs, test.rgb_n);
    metrics.test_ssim = average(test.ssims, test.rgb_n);
    metrics.test_lpips = average(test.lpipss, test.rgb_n);

    return metrics;
}

void saveFrameSequence(const std::shared_ptr<Dataset>& dataset,
                       const std::string& result_path)
{
    struct CamEntry {
        std::shared_ptr<Camera> cam;
        std::string type;
        int frame_idx;
    };

    std::vector<CamEntry> all_cams;
    for (auto& c : dataset->train_cameras_)
    {
        int idx = frameIndexFromImageName(c->image_name_);
        all_cams.push_back({c, "train", idx});
    }
    for (auto& c : dataset->test_cameras_)
    {
        int idx = frameIndexFromImageName(c->image_name_);
        all_cams.push_back({c, "test", idx});
    }

    std::sort(all_cams.begin(), all_cams.end(),
              [](const CamEntry& a, const CamEntry& b) {
                  return a.frame_idx < b.frame_idx;
              });

    std::string json_path = result_path + "/cameras.json";
    std::ofstream f(json_path);
    f << std::fixed << std::setprecision(10);
    f << "[\n";

    for (size_t i = 0; i < all_cams.size(); ++i)
    {
        const auto& entry = all_cams[i];
        const auto& cam = entry.cam;

        f << "  {\n";
        f << "    \"frame_idx\": " << entry.frame_idx << ",\n";
        f << "    \"image_name\": \"" << cam->image_name_ << "\",\n";
        f << "    \"type\": \"" << entry.type << "\",\n";
        f << "    \"width\": " << cam->image_width_ << ",\n";
        f << "    \"height\": " << cam->image_height_ << ",\n";
        f << "    \"fx\": " << cam->fx_ << ",\n";
        f << "    \"fy\": " << cam->fy_ << ",\n";
        f << "    \"cx\": " << cam->cx_ << ",\n";
        f << "    \"cy\": " << cam->cy_ << ",\n";
        f << "    \"R_cw\": [[" << cam->R_cw_(0,0) << "," << cam->R_cw_(0,1) << "," << cam->R_cw_(0,2)
          << "],[" << cam->R_cw_(1,0) << "," << cam->R_cw_(1,1) << "," << cam->R_cw_(1,2)
          << "],[" << cam->R_cw_(2,0) << "," << cam->R_cw_(2,1) << "," << cam->R_cw_(2,2) << "]],\n";
        f << "    \"t_cw\": [" << cam->t_cw_(0) << "," << cam->t_cw_(1) << "," << cam->t_cw_(2) << "]\n";
        f << "  }";
        if (i + 1 < all_cams.size()) f << ",";
        f << "\n";
    }

    f << "]\n";
    f.close();

    std::cout << "[saveFrameSequence] Saved " << all_cams.size()
              << " cameras to " << json_path << std::endl;

    std::string eval_dir = result_path + "/eval";
    fs::create_directories(eval_dir);

    std::map<std::string, int> camera_ids;
    std::vector<std::shared_ptr<Camera>> unique_cameras;
    std::vector<int> entry_camera_ids;
    entry_camera_ids.reserve(all_cams.size());
    for (const auto& entry : all_cams)
    {
        const std::string key = cameraKey(entry.cam);
        auto it = camera_ids.find(key);
        if (it == camera_ids.end())
        {
            int id = static_cast<int>(unique_cameras.size());
            camera_ids[key] = id;
            unique_cameras.push_back(entry.cam);
            entry_camera_ids.push_back(id);
        }
        else
        {
            entry_camera_ids.push_back(it->second);
        }
    }

    /// The evaluation camera table is native scale-1, not the training
    /// resolution: the exported PLY is a resolution-independent 3D scene, and
    /// scoring it against native-resolution ground truth is what makes the
    /// numbers comparable with other methods. Training is unaffected.
    const EvalGeometry eval_geometry = evaluationGeometry(dataset);
    std::cout << "[saveFrameSequence] Evaluation cameras are scale-1: "
              << eval_geometry.width << "x" << eval_geometry.height << " (training "
              << dataset->target_width_ << "x" << dataset->target_height_ << ", crop_y "
              << dataset->crop_y_ << ")" << std::endl;

    std::string eval_cameras_path = eval_dir + "/cameras.json";
    std::ofstream cf(eval_cameras_path);
    cf << std::fixed << std::setprecision(10);
    cf << "[\n";
    for (size_t i = 0; i < unique_cameras.size(); ++i)
    {
        const auto& cam = unique_cameras[i];
        cf << "  {\n";
        cf << "    \"id\": " << i << ",\n";
        cf << "    \"width\": " << eval_geometry.width << ",\n";
        cf << "    \"height\": " << eval_geometry.height << ",\n";
        cf << "    \"fx\": " << cam->fx_ * eval_geometry.inverse_scale << ",\n";
        cf << "    \"fy\": " << cam->fy_ * eval_geometry.inverse_scale << ",\n";
        cf << "    \"cx\": " << cam->cx_ * eval_geometry.inverse_scale << ",\n";
        cf << "    \"cy\": " << cam->cy_ * eval_geometry.inverse_scale << ",\n";
        cf << "    \"k1\": 0.0, \"k2\": 0.0, \"p1\": 0.0, \"p2\": 0.0, \"k3\": 0.0,\n";
        cf << "    \"position\": [0.0, 0.0, 0.0],\n";
        cf << "    \"rotation\": [[1,0,0],[0,1,0],[0,0,1]]\n";
        cf << "  }";
        if (i + 1 < unique_cameras.size()) cf << ",";
        cf << "\n";
    }
    cf << "]\n";
    cf.close();

    std::string eval_manifest_path = eval_dir + "/manifest.json";
    std::ofstream mf(eval_manifest_path);
    mf << std::fixed << std::setprecision(10);
    mf << "{\n";
    mf << "  \"source\": \"Gaussian-LIC native renderer\",\n";
    mf << "  \"frame_index_convention\": \"original input frame index\",\n";
    mf << "  \"frames\": [\n";
    for (size_t i = 0; i < all_cams.size(); ++i)
    {
        const auto& entry = all_cams[i];
        const auto& cam = entry.cam;
        Eigen::Quaterniond q_cw(cam->R_cw_);
        q_cw.normalize();
        mf << "    {\n";
        mf << "      \"seq_idx\": " << entry.frame_idx << ",\n";
        mf << "      \"is_test\": " << (entry.type == "test" ? "true" : "false") << ",\n";
        mf << "      \"camera_id\": " << entry_camera_ids[i] << ",\n";
        mf << "      \"image_name\": \"" << cam->image_name_ << "\",\n";
        mf << "      \"timestamp_ns\": " << cam->rgb_timestamp_ns_ << ",\n";
        mf << "      \"rgb_timestamp_ns\": " << cam->rgb_timestamp_ns_ << ",\n";
        mf << "      \"T_CW_qwxyz_txyz\": ["
           << q_cw.w() << ", " << q_cw.x() << ", " << q_cw.y() << ", " << q_cw.z() << ", "
           << cam->t_cw_(0) << ", " << cam->t_cw_(1) << ", " << cam->t_cw_(2) << "]\n";
        mf << "    }";
        if (i + 1 < all_cams.size()) mf << ",";
        mf << "\n";
    }
    mf << "  ]\n";
    mf << "}\n";
    mf.close();

    std::cout << "[saveFrameSequence] Saved normalized eval metadata to "
              << eval_manifest_path << std::endl;
}

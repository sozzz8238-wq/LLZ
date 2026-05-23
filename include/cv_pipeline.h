#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// YOLO-Pose 的 17 个 COCO 关键点索引
enum class Keypoint : int {
    NOSE = 0, LEFT_EYE = 1, RIGHT_EYE = 2,
    LEFT_EAR = 3, RIGHT_EAR = 4,
    LEFT_SHOULDER = 5, RIGHT_SHOULDER = 6,
    LEFT_ELBOW = 7, RIGHT_ELBOW = 8,
    LEFT_WRIST = 9, RIGHT_WRIST = 10,
    LEFT_HIP = 11, RIGHT_HIP = 12,
    LEFT_KNEE = 13, RIGHT_KNEE = 14,
    LEFT_ANKLE = 15, RIGHT_ANKLE = 16
};

struct PoseResult {
    std::vector<cv::Point2f> keypoints; // 17 个关键点 (归一化坐标 0~1)
    std::vector<float> scores;          // 每个关键点的置信度
};

// ==========================================
// CV 管线: 摄像头 → RKNN推理 → 关节角度 → DTW评分
// ==========================================
class CVPipeline {
public:
    CVPipeline(const std::string& modelPath);
    ~CVPipeline();

    // 初始化 NPU 与摄像头
    bool init(int cameraIndex = 0);

    // 从摄像头抓取一帧, 返回处理后的图像 (含关键点绘制)
    cv::Mat processFrame();

    // 获取当前帧的 DTW 评分 (与指定动作模板对比)
    float getDTWScore(const std::string& templateJson, int jointIndex);

    // 获取当前帧的关节角度序列 (某一关节的历史)
    std::vector<float> getJointHistory(int jointIndex);

    // 算三点的夹角 (例如: 髋-膝-踝 → 膝关节弯曲角)
    static float calcAngle(cv::Point2f a, cv::Point2f b, cv::Point2f c);

    // 提取所有关键关节角度
    std::vector<float> extractJointAngles(const PoseResult& pose);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

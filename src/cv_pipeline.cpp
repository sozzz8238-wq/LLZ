#include "cv_pipeline.h"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <iostream>
#include <numeric>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// RKNN API (板端部署时取消注释)
// #include "rknn_api.h"

using json = nlohmann::json;

// ==========================================
// PIMPL 内部实现
// ==========================================
struct CVPipeline::Impl {
    cv::VideoCapture cap;
    cv::Mat currentFrame;
    std::vector<PoseResult> currentPoses;

    // 角度序列历史 (关节索引 → 帧序列)
    std::unordered_map<int, std::deque<float>> jointHistory;
    static constexpr int MAX_HISTORY = 100;

    // RKNN 上下文 (板端可用)
    // rknn_context ctx;
    bool npuReady = false;

    // 模型路径
    std::string modelPath;

    // 图像预处理参数
    static constexpr int MODEL_W = 640;
    static constexpr int MODEL_H = 640;

    // 骨骼连线 (COCO 标准连接对)
    static const std::vector<std::pair<int, int>> SKELETON;
};

const std::vector<std::pair<int, int>> CVPipeline::Impl::SKELETON = {
    {15,13}, {13,11}, {11,5},  // 左腿→左肩
    {16,14}, {14,12}, {12,6},  // 右腿→右肩
    {5,7},   {7,9},            // 左臂
    {6,8},   {8,10},           // 右臂
    {5,6},   {11,12}           // 肩连线 + 髋连线
};

// ==========================================
// 构造函数 & 析构函数
// ==========================================
CVPipeline::CVPipeline(const std::string& modelPath)
    : pImpl_(std::make_unique<Impl>()) {
    pImpl_->modelPath = modelPath;
}

CVPipeline::~CVPipeline() {
    if (pImpl_->cap.isOpened()) {
        pImpl_->cap.release();
    }
    // rknn_destroy(pImpl_->ctx);  // 板端释放
}

// ==========================================
// init — 初始化 NPU + 摄像头
// ==========================================
bool CVPipeline::init(int cameraIndex) {
    // 1. 初始化 NPU (板端部署时取消注释)
    // FILE* fp = fopen(pImpl_->modelPath.c_str(), "rb");
    // if (!fp) { std::cerr << "[CV] RKNN 模型文件不存在!\n"; return false; }
    // fseek(fp, 0, SEEK_END);
    // size_t modelSize = ftell(fp);
    // rewind(fp);
    // std::vector<uint8_t> modelData(modelSize);
    // fread(modelData.data(), 1, modelSize, fp);
    // fclose(fp);
    //
    // int ret = rknn_init(&pImpl_->ctx, modelData.data(), modelSize, 0, nullptr);
    // if (ret < 0) { std::cerr << "[CV] RKNN 初始化失败! ret=" << ret << "\n"; return false; }
    // pImpl_->npuReady = true;

    // 2. 打开摄像头
    pImpl_->cap.open(cameraIndex);
    if (!pImpl_->cap.isOpened()) {
        std::cerr << "[CV] 无法打开摄像头 index=" << cameraIndex << "\n";
        return false;
    }
    pImpl_->cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    pImpl_->cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    std::cout << "[CV] 初始化完成 (摄像头: " << cameraIndex << ")\n";
    return true;
}

// ==========================================
// processFrame — 核心处理管线 (每帧)
// ==========================================
cv::Mat CVPipeline::processFrame() {
    pImpl_->cap >> pImpl_->currentFrame;
    if (pImpl_->currentFrame.empty()) return {};

    cv::Mat frame = pImpl_->currentFrame.clone();

    // ——— 步骤1: 预处理 (resize, BGR→RGB, normalize) ———
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(Impl::MODEL_W, Impl::MODEL_H));

    // ——— 步骤2: NPU 推理 (板端部署时调用 rknn_run) ———
    // rknn_inputs_set(ctx, ...);
    // rknn_run(ctx, nullptr);
    // rknn_outputs_get(ctx, ...);

    // ——— 步骤3: 后处理 (NMS + 解码关键点) ———
    // PoseResult pose = decodeOutput(outputData, frame.cols, frame.rows);

    // ——— 步骤4: 计算关节角度 ———
    // auto angles = extractJointAngles(pose);
    // for (size_t i = 0; i < angles.size(); ++i) {
    //     pImpl_->jointHistory[i].push_back(angles[i]);
    //     if (pImpl_->jointHistory[i].size() > Impl::MAX_HISTORY)
    //         pImpl_->jointHistory[i].pop_front();
    // }

    // ——— 步骤5: 骨架可视化 ———
    // drawSkeleton(frame, pose);

    return frame;
}

// ==========================================
// getDTWScore — DTW 匹配评分
// ==========================================
float CVPipeline::getDTWScore(const std::string& templateJson, int jointIndex) {
    // 读取 golden template
    std::ifstream ifs(templateJson);
    if (!ifs.is_open()) return 0.0f;
    json j;
    ifs >> j;

    std::vector<float> golden = j["angle_sequence"].get<std::vector<float>>();
    std::vector<float> user(pImpl_->jointHistory[jointIndex].begin(),
                            pImpl_->jointHistory[jointIndex].end());

    if (golden.empty() || user.empty()) return 0.0f;

    // ——— DTW 算法 ———
    int n = static_cast<int>(user.size());
    int m = static_cast<int>(golden.size());

    std::vector<std::vector<float>> dtw(n + 1, std::vector<float>(m + 1, 1e9f));
    dtw[0][0] = 0.0f;

    for (int i = 1; i <= n; ++i) {
        for (int jj = 1; jj <= m; ++jj) {
            float cost = std::abs(user[i - 1] - golden[jj - 1]);
            dtw[i][jj] = cost + std::min({dtw[i - 1][jj], dtw[i][jj - 1], dtw[i - 1][jj - 1]});
        }
    }

    // 归一化评分: 路径越长 → 累积误差越大 → 除以长度
    float rawScore = dtw[n][m] / static_cast<float>(n + m);
    float normalizedScore = 1.0f / (1.0f + rawScore / 180.0f); // 180° 角度范围归一化
    return normalizedScore;
}

// ==========================================
// getJointHistory — 获取关节角度历史
// ==========================================
std::vector<float> CVPipeline::getJointHistory(int jointIndex) {
    auto& deq = pImpl_->jointHistory[jointIndex];
    return std::vector<float>(deq.begin(), deq.end());
}

// ==========================================
// calcAngle — 三点求夹角 (余弦定理)
// ==========================================
float CVPipeline::calcAngle(cv::Point2f a, cv::Point2f b, cv::Point2f c) {
    // 向量 BA 和 BC, b 是顶点
    cv::Point2f ba = a - b;
    cv::Point2f bc = c - b;
    float dot = ba.x * bc.x + ba.y * bc.y;
    float mag = std::sqrt(ba.x * ba.x + ba.y * ba.y)
              * std::sqrt(bc.x * bc.x + bc.y * bc.y);
    if (mag < 1e-6f) return 0.0f;
    float rad = std::acos(std::clamp(dot / mag, -1.0f, 1.0f));
    return static_cast<float>(rad * 180.0 / M_PI);
}

// ==========================================
// extractJointAngles — 提取所有关键关节角度
// ==========================================
std::vector<float> CVPipeline::extractJointAngles(const PoseResult& pose) {
    std::vector<float> angles;
    const auto& kp = pose.keypoints;
    if (kp.size() < 17) return angles;

    // 左膝: 左髋-左膝-左踝
    angles.push_back(calcAngle(kp[11], kp[13], kp[15]));
    // 右膝: 右髋-右膝-右踝
    angles.push_back(calcAngle(kp[12], kp[14], kp[16]));
    // 左肘: 左肩-左肘-左腕
    angles.push_back(calcAngle(kp[5], kp[7], kp[9]));
    // 右肘: 右肩-右肘-右腕
    angles.push_back(calcAngle(kp[6], kp[8], kp[10]));
    // 左髋: 左肩-左髋-左膝
    angles.push_back(calcAngle(kp[5], kp[11], kp[13]));

    return angles;
}

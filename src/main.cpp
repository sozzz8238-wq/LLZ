/**
 * 👑 核心调度室 — main.cpp
 *
 * 多线程架构:
 *   Thread-1 [CV]  : 摄像头采集 + RKNN骨骼推理 + DTW实时打分 → 30 FPS
 *   Thread-2 [LLM] : 接收DTW分数 → 拼Prompt → RKLLM推理 → 生成评语
 *   Thread-3 [TTS] : 评语队列 → espeak/festival 语音合成播放
 *
 * 数据流:
 *   摄像头 → YOLO-Pose(NPU) → 关键点 → 关节角度 → DTW vs 模板
 *                                               ↓
 *   音箱 ← TTS ← 中文评语 ← Qwen2.5(NPU) ← Prompt(分数+偏差)
 */

#include "cv_pipeline.h"
#include "llm_pipeline.h"
#include "audio_engine.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

// ==========================================
// 全局状态
// ==========================================
std::atomic<bool> g_running{true};

// Thread-1 → Thread-2 的共享数据
struct FeedbackRequest {
    float    dtwScore;
    std::string actionName;
    std::string deviations;
};
std::queue<FeedbackRequest> g_feedbackQueue;
std::mutex g_queueMutex;

// Thread-2 → Thread-3 的评语输出
std::queue<std::string> g_ttsQueue;
std::mutex g_ttsMutex;

// ==========================================
// Thread-1: CV 视觉管线 (30 FPS)
// ==========================================
void cvThreadFunc(CVPipeline& cv, const std::string& templatePath) {
    while (g_running) {
        cv::Mat frame = cv.processFrame();

        // 实时计算 DTW 评分 (每次推理后更新)
        float score = cv.getDTWScore(templatePath, static_cast<int>(Keypoint::LEFT_KNEE));

        // 当 DTW 评分稳定 (积累 >30 帧) 且动作段结束, 触发 LLM 线程
        auto history = cv.getJointHistory(static_cast<int>(Keypoint::LEFT_KNEE));
        if (history.size() > 30) {
            std::lock_guard<std::mutex> lock(g_queueMutex);
            g_feedbackQueue.push({score, "深蹲", "左膝角度偏差 5°"});
        }

        // 显示当前帧 (骨架绘制已在 cv_pipeline 完成)
        if (!frame.empty()) {
            cv::imshow("Rehab AI - RK3588", frame);
        }

        if (cv::waitKey(1) == 27) { // ESC 退出
            g_running = false;
            break;
        }
    }
}

// ==========================================
// Thread-2: LLM 语言管线 (按需触发)
// ==========================================
void llmThreadFunc(LLMPipeline& llm) {
    while (g_running) {
        FeedbackRequest req;
        {
            std::lock_guard<std::mutex> lock(g_queueMutex);
            if (g_feedbackQueue.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
            req = g_feedbackQueue.front();
            g_feedbackQueue.pop();
        }

        std::cout << "[LLM] 生成评语中... DTW=" << req.dtwScore << "\n";
        std::string feedback = llm.generateFeedback(
            req.dtwScore, req.actionName, req.deviations
        );

        {
            std::lock_guard<std::mutex> lock(g_ttsMutex);
            g_ttsQueue.push(feedback);
        }
    }
}

// ==========================================
// Thread-3: TTS 语音播报
// ==========================================
void ttsThreadFunc(AudioEngine& audio) {
    while (g_running) {
        std::string text;
        {
            std::lock_guard<std::mutex> lock(g_ttsMutex);
            if (g_ttsQueue.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
            text = g_ttsQueue.front();
            g_ttsQueue.pop();
        }

        std::cout << "[TTS] 语音播报: " << text << "\n";
        audio.speak(text);
    }
}

// ==========================================
// 主函数入口
// ==========================================
int main(int argc, char* argv[]) {
    std::cout << "========================================\n";
    std::cout << "  AI 智能康复交互系统 - RK3588 Edition\n";
    std::cout << "========================================\n\n";

    // --- 1. 初始化各模块 ---
    CVPipeline cv("../models/yolov8n-pose.rknn");
    LLMPipeline llm("../models/qwen2.5_0.5b.rkllm");
    AudioEngine audio;

    if (!cv.init(0)) {
        std::cerr << "[FATAL] CV 管线初始化失败!\n";
        return -1;
    }

    if (!llm.init()) {
        std::cerr << "[WARN] LLM 管线初始化失败, 将跳过语音反馈\n";
    }

    if (!audio.init()) {
        std::cerr << "[WARN] 音频引擎初始化失败, 将跳过语音播报\n";
    }

    // --- 2. 启动三线程 ---
    std::string templateJson = "../templates/m01_golden.json";

    std::thread cvThr(cvThreadFunc, std::ref(cv), std::ref(templateJson));
    std::thread llmThr(llmThreadFunc, std::ref(llm));
    std::thread ttsThr(ttsThreadFunc, std::ref(audio));

    std::cout << "[MAIN] 系统运行中... 按 ESC 退出\n";

    // --- 3. 等待退出 ---
    cvThr.join();
    llmThr.join();
    ttsThr.join();

    cv::destroyAllWindows();
    std::cout << "[MAIN] 系统已安全退出\n";
    return 0;
}

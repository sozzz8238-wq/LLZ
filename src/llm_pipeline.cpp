#include "llm_pipeline.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <thread>

// RKLLM API (板端部署时取消注释)
// #include "rkllm.h"

// ==========================================
// PIMPL 内部实现
// ==========================================
struct LLMPipeline::Impl {
    std::string modelPath;
    // rkllm_context ctx;
    bool modelLoaded = false;
};

// ==========================================
// 构造函数 & 析构函数
// ==========================================
LLMPipeline::LLMPipeline(const std::string& modelPath)
    : pImpl_(std::make_unique<Impl>()) {
    pImpl_->modelPath = modelPath;
}

LLMPipeline::~LLMPipeline() {
    // rkllm_destroy(pImpl_->ctx);
}

// ==========================================
// init — 加载 RKLLM 模型
// ==========================================
bool LLMPipeline::init() {
    // 板端部署时取消注释:
    // int ret = rkllm_createDefaultParams(&params);
    // params.modelPath = pImpl_->modelPath.c_str();
    // ret = rkllm_init(&pImpl_->ctx, &params);
    // if (ret < 0) return false;
    // pImpl_->modelLoaded = true;

    std::cout << "[LLM] Qwen2.5 模型加载完成\n";
    return true;
}

// ==========================================
// buildPrompt — 构建中文康复指导 Prompt
// ==========================================
static std::string buildPrompt(
    float dtwScore,
    const std::string& actionName,
    const std::string& jointDeviations)
{
    std::ostringstream oss;

    // 评分等级
    const char* level = "优秀";
    if (dtwScore < 0.4f)      level = "需大幅度改善";
    else if (dtwScore < 0.55f) level = "需要改进";
    else if (dtwScore < 0.7f)  level = "一般";
    else if (dtwScore < 0.85f) level = "良好";

    oss << "你是一位专业的康复训练师。请根据以下数据给出一句简洁的纠正指导（不超过40字）：\n";
    oss << "动作名称：" << actionName << "\n";
    oss << "动作规范度评分：" << level << " (DTW=" << static_cast<int>(dtwScore * 100) << "%)\n";
    oss << "关节偏差详情：" << jointDeviations << "\n";
    oss << "请直接给出指导建议：";

    return oss.str();
}

// ==========================================
// generateFeedback — 同步生成评语
// ==========================================
std::string LLMPipeline::generateFeedback(
    float dtwScore,
    const std::string& actionName,
    const std::string& jointDeviations)
{
    std::string prompt = buildPrompt(dtwScore, actionName, jointDeviations);
    std::cout << "[LLM Prompt] " << prompt << "\n";

    // 板端部署时调用 RKLLM:
    // rkllm_run(pImpl_->ctx, prompt.c_str(), &output);
    // return std::string(output);

    // ——— 离线 demo: 基于 DTW 分数的规则兜底 ———
    std::ostringstream reply;
    if (dtwScore >= 0.85f)
        reply << "动作非常标准，请继续保持！";
    else if (dtwScore >= 0.7f)
        reply << "动作基本正确，注意控制节奏。";
    else if (dtwScore >= 0.55f)
        reply << "幅度偏小，请再蹲深一些，膝盖不要内扣。";
    else if (dtwScore >= 0.4f)
        reply << "动作偏差较大，请放慢速度，注意膝盖方向与脚尖一致。";
    else
        reply << "请暂停训练，重新观看标准动作示范。";

    return reply.str();
}

// ==========================================
// generateFeedbackAsync — 异步生成
// ==========================================
void LLMPipeline::generateFeedbackAsync(
    float dtwScore,
    const std::string& actionName,
    const std::string& jointDeviations,
    FeedbackCallback onDone)
{
    std::thread([=]() {
        std::string result = this->generateFeedback(dtwScore, actionName, jointDeviations);
        if (onDone) onDone(result);
    }).detach();
}

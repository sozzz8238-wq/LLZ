#include "audio_engine.h"

#include <cstdlib>
#include <iostream>

// ==========================================
// PIMPL 内部实现
// ==========================================
struct AudioEngine::Impl {
    bool ttsAvailable = false;
    std::string ttsCommand;      // TTS 后端命令
    std::string ttsLanguage;     // 语言参数
};

// ==========================================
// 构造函数 & 析构函数
// ==========================================
AudioEngine::AudioEngine()
    : pImpl_(std::make_unique<Impl>()) {}

AudioEngine::~AudioEngine() {
    stop();
}

// ==========================================
// init — 检测可用 TTS 后端
// ==========================================
bool AudioEngine::init() {
    // 优先级: espeak-ng > espeak > festival > gTTS
    if (std::system("which espeak-ng > /dev/null 2>&1") == 0) {
        pImpl_->ttsCommand = "espeak-ng -v zh";
        pImpl_->ttsAvailable = true;
    } else if (std::system("which espeak > /dev/null 2>&1") == 0) {
        pImpl_->ttsCommand = "espeak -v zh";
        pImpl_->ttsAvailable = true;
    } else if (std::system("which festival > /dev/null 2>&1") == 0) {
        pImpl_->ttsCommand = "festival --tts";
        pImpl_->ttsAvailable = true;
    }

    if (pImpl_->ttsAvailable) {
        std::cout << "[TTS] 后端就绪: " << pImpl_->ttsCommand << "\n";
    } else {
        std::cerr << "[TTS] 未检测到 TTS 后端, 请安装 espeak-ng\n";
    }

    return pImpl_->ttsAvailable;
}

// ==========================================
// speak — 同步播放
// ==========================================
void AudioEngine::speak(const std::string& text) {
    if (!pImpl_->ttsAvailable || text.empty()) return;

    std::string cmd = "echo \"" + text + "\" | " + pImpl_->ttsCommand;
    std::system(cmd.c_str());
}

// ==========================================
// speakAsync — 异步播放 (非阻塞)
// ==========================================
void AudioEngine::speakAsync(const std::string& text) {
    if (!pImpl_->ttsAvailable || text.empty()) return;

    // 后台进程播放, 不等结果
    std::string cmd = "echo \"" + text + "\" | " + pImpl_->ttsCommand + " &";
    std::system(cmd.c_str());
}

// ==========================================
// isSpeaking — 是否正在播放
// ==========================================
bool AudioEngine::isSpeaking() const {
    // 简化实现: espeak 同步调用会阻塞, 所以返回时已播完
    return false;
}

// ==========================================
// stop — 停止播放
// ==========================================
void AudioEngine::stop() {
    // 杀掉可能的 TTS 进程
    std::system("pkill -9 espeak 2>/dev/null");
    std::system("pkill -9 espeak-ng 2>/dev/null");
}

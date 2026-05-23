cmake_minimum_required(VERSION 3.14)
project(Rehab_AI_RK3588 VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================
# 1. 交叉编译工具链 (板端部署时取消注释)
# =============================================
# set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
# set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
# set(CMAKE_SYSROOT /path/to/rk3588/sysroot)
# set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
# set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
# set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# =============================================
# 2. 第三方头文件
# =============================================
set(THIRDPARTY_DIR ${CMAKE_SOURCE_DIR}/3rdparty)

# =============================================
# 3. RK3588 推理库 (板端路径, 按实际SDK调整)
# =============================================
# set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/sdk/librknnrt.so)
# set(RKLLM_LIB  ${CMAKE_SOURCE_DIR}/sdk/librkllm.so)
# set(RKNN_API_DIR ${CMAKE_SOURCE_DIR}/sdk/include)

# =============================================
# 4. OpenCV (板端路径)
# =============================================
# set(OpenCV_DIR /path/to/opencv-aarch64/lib/cmake/opencv4)
find_package(OpenCV QUIET)

# =============================================
# 5. 可执行文件
# =============================================
add_executable(rehab_app
    src/main.cpp
    src/cv_pipeline.cpp
    src/llm_pipeline.cpp
    src/audio_engine.cpp
)

# =============================================
# 6. 头文件搜索路径
# =============================================
target_include_directories(rehab_app PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${THIRDPARTY_DIR}
    # ${RKNN_API_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# =============================================
# 7. 链接库
# =============================================
target_link_libraries(rehab_app PRIVATE
    # ${RKNN_RT_LIB}
    # ${RKLLM_LIB}
    ${OpenCV_LIBS}
    pthread
)

# =============================================
# 8. 编译选项
# =============================================
target_compile_options(rehab_app PRIVATE
    -Wall -Wextra
    -O2 -DNDEBUG
    # -mcpu=cortex-a76.cortex-a55  # RK3588 大小核优化
)

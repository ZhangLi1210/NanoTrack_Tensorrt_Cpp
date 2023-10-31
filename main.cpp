#include <NvInfer.h>
#include "NvInferPlugin.h"
#include <iostream>
#include<vector>
#include <fstream>
#include <cassert>
#include <string>
#include "include/Nano_track.h"

using namespace Track;

void cxy_wh_2_rect(const cv::Point& pos, const cv::Point2f& sz, cv::Rect &rect)
{
    rect.x = std::max(0, pos.x - int(sz.x / 2));
    rect.y = std::max(0, pos.y - int(sz.y / 2));
    rect.width = int(sz.x);
    rect.height = int(sz.y);
}

int main() {

    //nano_track的后处理太多了,engine运算只需要1-2ms
    //nanotrack的主干网络有两个 一个nanotrack_backbone_temp.engine用于初始化
    std::string backbone_model_init = "../engine/nanotrack_backbone_temp.engine";
    std::string backbone_model = "../engine/nanotrack_backbone_exam.engine";

    std::string head_model = "../engine/nanotrack_head.engine";

    auto nano_tracker = new NanoTrack(backbone_model_init, backbone_model, head_model);

    cv::VideoCapture cap;
    cv::Mat frame;
    //cap.open("../text_vidio/track_text_vidio.mp4");
    cap.open(0);
    cap >> frame;
    cv::putText(frame, "Select target ROI and press ENTER", cv::Point2i(20, 30),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 0, 0), 1);
    cv::Rect init_bbox = cv::selectROI("11", frame);
    std::cout << frame.size << std::endl;
    //在init中我们希望拿到一个1X48X8x8的数据 这个数据被保存在了nano_tracker对象的成员变量(this->init_input)中.
    nano_tracker->init(frame, init_bbox);
    for (;;) {
        // Read a new frame.
        cap >> frame;
        if (frame.empty())
            break;
        auto start = std::chrono::high_resolution_clock::now();
        nano_tracker->track(frame);
        // 获取结束时间点
        auto end = std::chrono::high_resolution_clock::now();

        // 计算时间差
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // 输出运行时间（毫秒）
        std::cout << "fps: " << 1000/duration.count() << std::endl;
        // Result to rect.
        cv::Rect rect;
        cxy_wh_2_rect(nano_tracker->state.target_pos, nano_tracker->state.target_sz, rect);

        // Boundary judgment.
        cv::Mat track_window;
        if (0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= frame.cols && 0 <= rect.y && 0 <= rect.height &&
            rect.y + rect.height <= frame.rows) {
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0));
        }
        // Display result.
        cv::imshow("demo", frame);
        cv::waitKey(20);
    }
    cv::destroyWindow("demo");
    cap.release();
    delete nano_tracker;
    return 0;
}


///以下是作者学习过程中的注释,可以不看
//    ////预定义
//    nvinfer1::ICudaEngine*       engine  = nullptr;
//    nvinfer1::IRuntime*          runtime = nullptr;
//    nvinfer1::IExecutionContext* context = nullptr;
//    cudaStream_t                 stream  = nullptr;
//    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
//
//    int                  num_bindings;
//    int                  num_inputs  = 0;
//    int                  num_outputs = 0;
//    std::vector<Binding> input_bindings;
//    std::vector<Binding> output_bindings;
//    std::vector<void*>   host_ptrs;
//    std::vector<void*>   device_ptrs;
//    ///预定义
//
//    ///打开trt文件
//    //打开名为"nanotrack_backbone_sim.trt"的二进制文件，这是通过std::ifstream类完成的，文件以二进制模式打开。
//    std::ifstream file("/home/lizhang/nano_track_trt_text/nanotrack_backbone_sim.trt", std::ios::binary);
//    //检查是否成功打开
//    assert(file.good());
//    //将文件指针移动到文件的末尾，以便获取文件的大小。
//    file.seekg(0, std::ios::end);
//    //获取了文件大小 即size
//    auto size = file.tellg();
//    //将文件指针移动回文件的开头。
//    file.seekg(0, std::ios::beg);
//    //分配一个名为trtModelStream的数组，其大小为前面获取的文件大小（size变量的值）。
//    char* trtModelStream = new char[size];
//    //判断是否分配成功
//    assert(trtModelStream);
//    //将文件读入trtModelStream
//    file.read(trtModelStream, size);
//    file.close();
//    std::cout<<"engine模型成功读取"<<std::endl;
//   ///打开trt文件
//
//   ///模型初始化
//   //初始化推理插件库
//    initLibNvInferPlugins(&gLogger, "");
//
//   ///步骤一:创建推理运行时(InferRuntime)对象 该对象作用如下
//   //1.该函数会创建一个 TensorRT InferRuntime对象，这个对象是 TensorRT 库的核心组成部分之一。
//   // InferRuntime对象是用于在推理阶段执行深度学习模型的实例。它提供了一种有效的方式来执行模型的前向传播操作。
//   //2. TensorRT InferRuntime对象还负责管理 GPU 资源，包括分配和释放 GPU 内存。这对于加速推理操作非常重要，因为它可以确保有效地利用 GPU 资源，同时减少 GPU 内存泄漏的风险。
//   //3. 一旦创建了 TensorRT 推理运行时对象，你可以使用它来构建、配置和执行 TensorRT 模型引擎（nvinfer1::ICudaEngine）。
//   // Engine模型引擎是一个已经优化过的深度学习模型，可以高效地在 GPU 上执行推理操作。
//    runtime = nvinfer1::createInferRuntime(gLogger);
//
//    //判断runtime是否创建成功
//    assert(runtime != nullptr);
//
//    std::cout<<"InferRuntime对象创建成功"<<std::endl;
//
//    ///步骤二: 通过构建好InferRuntime对象 反序列化加载引擎给到(nvinfer1::ICudaEngine对象)
//    engine = runtime->deserializeCudaEngine(trtModelStream, size);
//
//    //判断engine是否创建成功
//    assert(engine != nullptr);
//    std::cout<<"ICudaEngine对象创建成功"<<std::endl;
//
//    //删除模型文件,因为engine已经创建成功
//    delete[] trtModelStream;
//
//    ///步骤三:使用engine创建一个执行上下文对象(nvinfer1::IExecutionContext*)
//    //执行上下文对象作用如下(nvinfer1::IExecutionContext):
//    //1. 模型推理执行：执行上下文是用于执行深度学习模型推理操作的实例。
//    // 一旦你创建了执行上下文，你可以使用它来加载输入数据、运行模型的前向传播，然后获取输出结果。
//    // 这允许你将模型应用于实际数据，以获得推理结果。
//    //2. GPU 资源管理：执行上下文还负责管理 GPU 资源，包括内存分配和释放。
//    // 这确保了在执行推理操作时有效地利用 GPU 资源，同时降低了 GPU 内存泄漏的风险。
//    context = engine->createExecutionContext();
//
//    assert(context != nullptr);
//    std::cout<<"IExecutionContext对象创建成功"<<std::endl;
//
//    //cudaStreamCreate() 是 NVIDIA CUDA 库中的函数，用于创建一个 CUDA 流（CUDA Stream）。CUDA 流是一种并行执行 CUDA 操作的方式，它允许将多个 CUDA 操作异步执行，
//    // 从而提高GPU的利用率。每个CUDA流代表一个独立的任务队列，CUDA操作可以按照流的顺序在多个流之间并行执行。
//    cudaStreamCreate(&stream);
//
//    //engine->getNbBindings() 是用于获取 TensorRT 模型引擎（nvinfer1::ICudaEngine）绑定的输入和输出张量数量的函数。
//    // 在使用TensorRT进行深度学习模型推理时，你需要知道模型引擎绑定的输入和输出张量的数量，以便为它们分配内存并正确配置推理上下文。
//    //具体来说，engine->getNbBindings() 函数返回一个整数值，表示与该模型引擎相关的绑定张量的总数。这个值通常是输入张量的数量加上输出张量的数量
//    num_bindings = engine->getNbBindings();
//    //输出为2即输入张量的数量和输出张量的数量总数为2
//    std::cout<<"输入和输出张量的总数为:"<<num_bindings<<std::endl;
//
//    for (int i = 0; i < num_bindings; ++i) {
//        //该结构体用于保存第i个绑定的信息,
//        Binding            binding;
//        //一个结构体,用于表示张量(输入输出和中间层数据)
//        // dims.nbDims表示维度的数量1X3X255X255 时维度为4
//        // dims.d 一个整数数组，包含每个维度的大小
//        //一个形状为 (batch_size, channels, height, width) (1X3X255X255)
//        // 的四维图像张量可以表示为 nvinfer1::Dims 对象，其中 nbDims 为 4，d[0] 表示批量大小，d[1] 表示通道数，d[2] 表示高度，d[3] 表示宽度。
//        nvinfer1::Dims     dims;
//
//        // 在使用 TensorRT 进行深度学习模型推理时，每个模型引擎都有输入绑定和输出绑定。这些绑定指定了模型的输入和输出张量的属性，包括数据类型、维度等。
//        //这些信息都是可以被获取的
//        //这里是获取第i个绑定的数据类型. i=0是为输入绑定,i=1时为输出绑定
//        nvinfer1::DataType dtype = engine->getBindingDataType(i);
//        //保存第i个绑定的数据类型所对应的字节大小
//        binding.dsize            = type_to_size(dtype);
//
//        //获取第i个绑定的名称 i = 0 ,输入数据的名称  i = 1,输出数据的名称
//        std::string        name  = engine->getBindingName(i);
//        binding.name             = name;
//        std::cout<<"第"<<i<<"个绑定的名字为"<<binding.name<<std::endl;
//
//        //这个函数可以判断第i个绑定是输入绑定还是输出绑定
//        bool IsInput = engine->bindingIsInput(i);
//        if (IsInput) {
//            //如果是输入绑定,将输入绑定的数量进行保存
//            num_inputs += 1;
//            //获取一个输入的图像张量,用dims进行保存
//            dims = engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
//            //binding.size = get_size_by_dims(dims)计算该图像张量的所有元素的数量(1X3X255X255)
//            //binding.dsize保存的元素的数据类型所需要的字节数,最终即可计算所需的总内存大小
//            binding.size = get_size_by_dims(dims);
//            //将该图像张量也保存进binding里面
//            binding.dims = dims;
//            input_bindings.push_back(binding);
//
//            //context对象用于执行推理,这里获取到输入图像张量之后,使用该函数设置推理时的输入张量维度
//            context->setBindingDimensions(i, dims);
//        }
//        else {
//            //获取输出的张量维度信息
//            dims         = context->getBindingDimensions(i);
//            binding.size = get_size_by_dims(dims);
//            binding.dims = dims;
//            output_bindings.push_back(binding);
//            num_outputs += 1;
//        }
//    }

//    // 清理资源
//    context->destroy();
//    engine->destroy();
//    runtime->destroy();
//}


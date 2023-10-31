//
// Created by lizhang on 2023/10/29.
//

#ifndef NANO_TRACK_TRT_TEXT_NANO_TRACK_H
#define NANO_TRACK_TRT_TEXT_NANO_TRACK_H

#include "NvInferPlugin.h"
#include "common.h"
#include "fstream"


using namespace Track;

struct Config{

    std::string windowing = "cosine";
    std::vector<float> window;

    int stride = 16;
    float penalty_k = 0.148;
    float window_influence = 0.462;
    float lr = 0.390;
    int exemplar_size=127;
    int instance_size=255;
    int total_stride=16;
    int score_size=16;
    float context_amount = 0.5;
};

struct State {
    int im_h;
    int im_w;
    cv::Scalar channel_ave;
    cv::Point target_pos;
    cv::Point2f target_sz = {0.f, 0.f};
    float cls_score_max;
};

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

static float sz_whFun(cv::Point2f wh)
{
    float pad = (wh.x + wh.y) * 0.5f;
    float sz2 = (wh.x + pad) * (wh.y + pad);
    return std::sqrt(sz2);
}

static std::vector<float> sz_change_fun(std::vector<float> w, std::vector<float> h,float sz)
{
    int rows = int(std::sqrt(w.size()));
    int cols = int(std::sqrt(w.size()));
    std::vector<float> pad(rows * cols, 0);
    std::vector<float> sz2;
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            pad[i*cols+j] = (w[i * cols + j] + h[i * cols + j]) * 0.5f;
        }
    }
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            float t = std::sqrt((w[i * rows + j] + pad[i*rows+j]) * (h[i * rows + j] + pad[i*rows+j])) / sz;

            sz2.push_back(std::max(t,(float)1.0/t) );
        }
    }
    return sz2;
}

static std::vector<float> ratio_change_fun(std::vector<float> w, std::vector<float> h, cv::Point2f target_sz)
{
    int rows = int(std::sqrt(w.size()));
    int cols = int(std::sqrt(w.size()));
    float ratio = target_sz.x / target_sz.y;
    std::vector<float> sz2;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float t = ratio / (w[i * cols + j] / h[i * cols + j]);
            sz2.push_back(std::max(t, (float)1.0 / t));
        }
    }

    return sz2;
}

class NanoTrack {
public:
    explicit NanoTrack(const std::string& backbone_init_engine_file_path,const std::string& backbone_engine_file_path,const std::string& head_engine_file_path);
    ~NanoTrack();

    void                 make_pipe();
    void                 init(cv::Mat img, cv::Rect bbox);
    void                 infer(std::vector<void*> device_ptrs,std::vector<void*> host_ptrs,int num_inputs,int num_outputs,std::vector<Binding> output_bindings,int infer_kind);
    void                 track(cv::Mat img);
    void                 update(const cv::Mat &x_crops, cv::Point &target_pos, cv::Point2f &target_sz,  float scale_z, float &cls_score_max);
    cv::Mat              convert_score(float* out_put1);
    cv::Rect             convert_bbox(float* reg, cv::Point target_pos);

    int stride=16;
    // state  dynamic
    State state;
    // config static
    Config cfg;

    const float mean_vals[3] = { 0.485f*255.f, 0.456f*255.f, 0.406f*255.f };
    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};

    int                  num_bindings_init;  //engine所有输入和输出张量的数量
    int                  num_bindings_backbone;
    int                  num_bindings_head;

    int                  num_inputs_init  = 0; //输入张量的数量
    int                  num_outputs_init = 0; //输出张量的数量

    int                  num_inputs_backbone  = 0;
    int                  num_outputs_backbone = 0;

    int                  num_inputs_head  = 0;
    int                  num_outputs_head = 0;

    std::vector<Binding> input_bindings_init; //所有输入张量的信息(结构体Binding)存成列表
    std::vector<Binding> output_bindings_init;

    std::vector<Binding> input_bindings_backbone;
    std::vector<Binding> output_bindings_backbone;

    std::vector<Binding> input_bindings_head;
    std::vector<Binding> output_bindings_head;

    std::vector<void*>   init_host_ptrs;  //init的CPU指针
    std::vector<void*>   init_device_ptrs; //init的GPU指针

    std::vector<void*>   backbone_host_ptrs;
    std::vector<void*>   backbone_device_ptrs;

    std::vector<void*>   head_host_ptrs;
    std::vector<void*>   head_device_ptrs;

    float * init_output;
    float * backbone_output;

//
//    //分别为框的中心和框的宽高 此数据在追踪过程中被实时更新
//    cv::Point target_pos; // cx, cy
//    cv::Point2f target_sz = {0.f, 0.f}; //w,h

private:
    nvinfer1::IRuntime*          runtime = nullptr;

    nvinfer1::ICudaEngine*       engine_backbone_init  = nullptr;
    nvinfer1::IExecutionContext* context_backbone_init = nullptr;

    nvinfer1::ICudaEngine*       engine_backbone  = nullptr;
    nvinfer1::IExecutionContext* context_backbone = nullptr;

    nvinfer1::ICudaEngine*       engine_head  = nullptr;
    nvinfer1::IExecutionContext* context_head = nullptr;

    cudaStream_t                 stream  = nullptr;
    Logger                       gLogger{nvinfer1::ILogger::Severity::kERROR};
    void create_grids();
    void create_window();
    cv::Mat get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz,cv::Scalar channel_ave);

    std::vector<float> grid_to_search_x;
    std::vector<float> grid_to_search_y;
    std::vector<float> window;
//    cv::Scalar avg_chans;
//    float s_x;
//    float scale_z;

    float * input_blob;
//    cv::Mat x_crop;

};

NanoTrack::NanoTrack(const std::string& backbone_init_engine_file_path,const std::string& backbone_engine_file_path,const std::string& head_engine_file_path)
{
    this->input_blob = new float[3*255*255];

    ///打开trt文件
    //打开engine二进制文件，这是通过std::ifstream类完成的，文件以二进制模式打开。
    std::ifstream file_backbone_init(backbone_init_engine_file_path, std::ios::binary);
    std::ifstream file_backbone(backbone_engine_file_path, std::ios::binary);
    std::ifstream file_head(head_engine_file_path, std::ios::binary);
    //检查是否成功打开
    assert(file_backbone_init.good());
    assert(file_backbone.good());
    assert(file_head.good());
    //将文件指针移动到文件的末尾，以便获取文件的大小。
    file_backbone_init.seekg(0, std::ios::end);
    file_backbone.seekg(0, std::ios::end);
    file_head.seekg(0, std::ios::end);
    //获取了文件大小 即size
    auto size_backbone_init = file_backbone_init.tellg();
    auto size_backbone = file_backbone.tellg();
    auto size_head = file_head.tellg();
    //将文件指针移动回文件的开头。
    file_backbone_init.seekg(0, std::ios::beg);
    file_backbone.seekg(0, std::ios::beg);
    file_head.seekg(0, std::ios::beg);
    //分配一个名为trtModelStream的数组，其大小为前面获取的文件大小（size变量的值）。
    char* trtModelStream_init = new char[size_backbone_init];
    char* trtModelStream_backbone = new char[size_backbone];
    char* trtModelStream_head = new char[size_head];
    //判断是否分配成功
    assert(trtModelStream_init);
    assert(trtModelStream_backbone);
    assert(trtModelStream_head);
    //将文件读入trtModelStream
    file_backbone_init.read(trtModelStream_init,  size_backbone_init);
    file_backbone.read(trtModelStream_backbone,  size_backbone);
    file_head.read(trtModelStream_head,  size_head);

    file_backbone_init.close();
    file_backbone.close();
    file_head.close();
    ///打开trt文件

    //初始化推理插件库
    initLibNvInferPlugins(&this->gLogger, "");

    ///步骤一:创建推理运行时(InferRuntime)对象 该对象作用如下
    //1.该函数会创建一个 TensorRT InferRuntime对象，这个对象是 TensorRT 库的核心组成部分之一。
    // InferRuntime对象是用于在推理阶段执行深度学习模型的实例。它提供了一种有效的方式来执行模型的前向传播操作。
    //2. TensorRT InferRuntime对象还负责管理 GPU 资源，包括分配和释放 GPU 内存。这对于加速推理操作非常重要，因为它可以确保有效地利用 GPU 资源，同时减少 GPU 内存泄漏的风险。
    //3. 一旦创建了 TensorRT 推理运行时对象，你可以使用它来构建、配置和执行 TensorRT 模型引擎（nvinfer1::ICudaEngine）。
    // Engine模型引擎是一个已经优化过的深度学习模型，可以高效地在 GPU 上执行推理操作。
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    ///步骤二: 通过构建好InferRuntime对象 反序列化加载引擎给到(nvinfer1::ICudaEngine对象)
    this->engine_backbone_init = this->runtime->deserializeCudaEngine(trtModelStream_init, size_backbone_init);
    assert(this->engine_backbone_init != nullptr);

    this->engine_backbone = this->runtime->deserializeCudaEngine(trtModelStream_backbone, size_backbone);
    assert(this->engine_backbone != nullptr);

    this->engine_head = this->runtime->deserializeCudaEngine(trtModelStream_head, size_head);
    assert(this->engine_backbone_init != nullptr);

    delete[] trtModelStream_head;
    delete[] trtModelStream_init;
    delete[] trtModelStream_backbone;

    ///步骤三:使用engine创建一个执行上下文对象(nvinfer1::IExecutionContext*)
    //执行上下文对象作用如下(nvinfer1::IExecutionContext):
    //1. 模型推理执行：执行上下文是用于执行深度学习模型推理操作的实例。
    // 一旦你创建了执行上下文，你可以使用它来加载输入数据、运行模型的前向传播，然后获取输出结果。
    // 这允许你将模型应用于实际数据，以获得推理结果。
    //2. GPU 资源管理：执行上下文还负责管理 GPU 资源，包括内存分配和释放。
    // 这确保了在执行推理操作时有效地利用 GPU 资源，同时降低了 GPU 内存泄漏的风险。
    this->context_backbone_init = this->engine_backbone_init->createExecutionContext();
    assert(this->context_backbone_init != nullptr);

    this->context_backbone = this->engine_backbone->createExecutionContext();
    assert(this->context_backbone != nullptr);

    this->context_head = this->engine_head->createExecutionContext();
    assert(this->context_head != nullptr);

    //cudaStreamCreate() 是 NVIDIA CUDA 库中的函数，用于创建一个 CUDA 流（CUDA Stream）。CUDA 流是一种并行执行 CUDA 操作的方式，它允许将多个 CUDA 操作异步执行，
    // 从而提高GPU的利用率。每个CUDA流代表一个独立的任务队列，CUDA操作可以按照流的顺序在多个流之间并行执行。
    cudaStreamCreate(&this->stream);

    //engine->getNbBindings() 是用于获取 TensorRT 模型引擎（nvinfer1::ICudaEngine）绑定的输入和输出张量数量的函数。
    // 在使用TensorRT进行深度学习模型推理时，你需要知道模型引擎绑定的输入和输出张量的数量，以便为它们分配内存并正确配置推理上下文。
    //具体来说，engine->getNbBindings() 函数返回一个整数值，表示与该模型引擎相关的绑定张量的总数。这个值通常是输入张量的数量加上输出张量的数量
    this->num_bindings_init = this->engine_backbone_init->getNbBindings();
    this->num_bindings_backbone = this->engine_backbone->getNbBindings();
    this->num_bindings_head = this->engine_head->getNbBindings();

    ///为初始化模型赋予初值
    for (int i = 0; i < this->num_bindings_init; ++i) {
        //该结构体用于保存第i个输入或输出绑定的信息
        Binding            binding;

        //一个结构体,用于表示张量(输入输出和中间层数据)
        // dims.nbDims表示维度的数量1X3X255X255 时维度为4
        // dims.d 一个整数数组，包含每个维度的大小
        //一个形状为 (batch_size, channels, height, width) (1X3X255X255)
        // 的四维图像张量可以表示为 nvinfer1::Dims 对象，其中 nbDims 为 4，d[0] 表示批量大小，d[1] 表示通道数，d[2] 表示高度，d[3] 表示宽度。
        nvinfer1::Dims     dims;

        // 在使用 TensorRT 进行深度学习模型推理时，每个模型引擎都有输入绑定和输出绑定。这些绑定指定了模型的输入和输出张量的属性，包括数据类型、维度等。
        //这些信息都是可以被获取的
        //这里是获取第i个绑定的数据类型. i=0是为输入绑定,i=1时为输出绑定
        nvinfer1::DataType dtype = this->engine_backbone_init->getBindingDataType(i);
        //保存第i个绑定的张量的数据类型所对应的字节大小
        binding.dsize            = type_to_size(dtype);

        //获取第i个绑定的名称 i = 0 ,输入数据的名称  i = 1,输出数据的名称
        std::string        name  = this->engine_backbone_init->getBindingName(i);
        binding.name             = name;

        //这个函数可以判断第i个绑定是输入绑定还是输出绑定
        bool IsInput = engine_backbone_init->bindingIsInput(i);
        if (IsInput) {
            //如果是输入绑定,将输入绑定的数量加1
            this->num_inputs_init += 1;
            //获取一个输入的图像张量,用dims进行保存
            dims         = this->engine_backbone_init->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            //binding.size = get_size_by_dims(dims)计算该图像张量的所有元素的数量(1X3X255X255)
            //binding.dsize保存的元素的数据类型所需要的字节数,最终即可计算所需的总内存大小
            binding.size = get_size_by_dims(dims);
            //将该图像张量也保存进binding里面
            binding.dims = dims;
            this->input_bindings_init.push_back(binding);
            // set max opt shape
            // context对象用于执行推理,这里获取到输入图像张量之后,使用该函数设置推理时的输入张量维度
            this->context_backbone_init->setBindingDimensions(i, dims);
        }
        else {
            //获取输出的张量维度信息
            dims         = this->context_backbone_init->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings_init.push_back(binding);
            this->num_outputs_init += 1;
        }
    }

    ///为backbone模型赋予初值
    for (int i = 0; i < this->num_bindings_backbone; ++i) {
        Binding            binding;
        nvinfer1::Dims     dims;
        nvinfer1::DataType dtype = this->engine_backbone->getBindingDataType(i);
        binding.dsize            = type_to_size(dtype);
        std::string        name  = this->engine_backbone->getBindingName(i);
        binding.name             = name;
        bool IsInput = engine_backbone->bindingIsInput(i);
        if (IsInput) {
            this->num_inputs_backbone += 1;
            dims         = this->engine_backbone->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings_backbone.push_back(binding);
            this->context_backbone->setBindingDimensions(i, dims);
        }
        else {
            //获取输出的张量维度信息
            dims         = this->context_backbone->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings_backbone.push_back(binding);
            this->num_outputs_backbone += 1;
        }
    }

    ///为head模型赋予初值
    for (int i = 0; i < this->num_bindings_head; ++i) {
        Binding            binding;
        nvinfer1::Dims     dims;
        nvinfer1::DataType dtype = this->engine_head->getBindingDataType(i);
        binding.dsize            = type_to_size(dtype);
        std::string        name  = this->engine_head->getBindingName(i);
        binding.name             = name;
        bool IsInput = engine_head->bindingIsInput(i);
        if (IsInput) {
            this->num_inputs_head += 1;
            dims         = this->engine_head->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings_head.push_back(binding);
            this->context_head->setBindingDimensions(i, dims);
        }
        else {
            //获取输出的张量维度信息
            dims         = this->context_head->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings_head.push_back(binding);
            this->num_outputs_head += 1;
        }
    }
//    std::cout<<"初始化backbone模型输入张量数量为"<<this->num_inputs_init<<std::endl;
//    std::cout<<"初始化backbone模型输出张量数量为"<<this->num_outputs_init<<std::endl;
//
//    std::cout<<"backbone模型输入张量数量为"<<this->num_inputs_backbone<<std::endl;
//    std::cout<<"backbone模型输出张量数量为"<<this->num_outputs_backbone<<std::endl;
//
//    std::cout<<"head模型输入张量数量为"<<this->num_inputs_head<<std::endl;
//    std::cout<<"head模型输出张量数量为"<<this->num_outputs_head<<std::endl;

    //这里已经为每一个模型输入和输出在GPU和CPU上都分配了内存和指针
    this->make_pipe();
}


NanoTrack::~NanoTrack()
{
    this->context_backbone->destroy();
    this->engine_backbone->destroy();
    this->context_head->destroy();
    this->engine_head->destroy();

    this->runtime->destroy();
    delete[] input_blob;

//    CHECK(cudaFree(init_output));

    cudaStreamDestroy(this->stream);

    for (auto& ptr : this->backbone_device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->backbone_host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }

    for (auto& ptr : this->head_device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->head_host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}


void NanoTrack::infer(std::vector<void*> device_ptrs,std::vector<void*> host_ptrs,int num_inputs,int num_outputs,std::vector<Binding> output_bindings,int infer_kind)
{
    // this->context->enqueueV2：使用TensorRT执行上下文 this->context 对象执行推理操作。
    // 这是实际的推理步骤，它将输入数据传递给模型并执行模型的前向传播。
    // 具体来说，enqueueV2 接受以下参数：
    // this->device_ptrs.data()：包含了输入和输出数据的GPU内存指针数组。这些指针指向了经过分配的GPU内存中的输入数据和输出数据。
    // this->stream：CUDA流，用于异步执行推理操作。
    // nullptr：这里为了简化没有传递其他回调函数。
    switch (infer_kind) {
        //在init中前向传播
        case 0:
            this->context_backbone_init->enqueueV2(device_ptrs.data(), this->stream, nullptr);
            break;
        //在backbone中前向传播
        case 1:
            this->context_backbone->enqueueV2(device_ptrs.data(), this->stream, nullptr);
            break;
        //在head中前向传播
        case 2:
            this->context_head->enqueueV2(device_ptrs.data(), this->stream, nullptr);
            break;
        default:
            std::cout<<"infer_kind set error"<<std::endl;
            return;
    }

    ///循环处理输出数据
    for (int i = 0; i < num_outputs; i++) {
        //对于每一个输出绑定,计算输出张量的大小,该大小为张量所有元素个数乘以每个元素所占用的字节数
        size_t osize = output_bindings[i].size * output_bindings[i].dsize;

        //使用 cudaMemcpyAsync 将模型的输出数据从GPU内存（this->device_ptrs[i + this->num_inputs]，其中 i 是当前输出绑定的索引）
        // 异步复制到CPU内存（this->host_ptrs[i]，其中 i 是当前输出绑定的索引）。
        //这个步骤用于将模型的输出数据从GPU内存传输到CPU内存，以便进一步处理和分析。
        CHECK(cudaMemcpyAsync(
                host_ptrs[i], device_ptrs[i + num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    //等待CUDA流中的所有操作完成。这是必要的，以确保在使用模型的输出数据之前，所有数据都已正确地从GPU传输到CPU。
    cudaStreamSynchronize(this->stream);
}

void NanoTrack::make_pipe()
{
    ///为三个模型的每个绑定分配内存和指针 初始化模型使用完毕后,可以先将其释放
    ///device_ptrs中包含了输入和输出的GPU内存指针

    //对于输入绑定，使用 cudaMallocAsync 分配GPU内存，以便存储输入数据。
    // 它会为每个输入绑定分配一个相应大小的GPU内存块，并将指针添加到 this->device_ptrs 向量中。
    for (auto& bindings : this->input_bindings_init) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->init_device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : this->input_bindings_backbone) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->backbone_device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : this->input_bindings_head) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->head_device_ptrs.push_back(d_ptr);
    }

    //对于输出绑定，函数分配两个内存块：一个在GPU上分配，一个在CPU上分配。
    // cudaMallocAsync 用于分配GPU内存，
    // cudaHostAlloc 用于在CPU上分配内存。
    // 然后，将这两个内存块的指针添加到 this->device_ptrs 和 this->host_ptrs 向量中。
    for (auto& bindings : this->output_bindings_init) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->init_device_ptrs.push_back(d_ptr);
        this->init_host_ptrs.push_back(h_ptr);
    }

    for (auto& bindings : this->output_bindings_backbone) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->backbone_device_ptrs.push_back(d_ptr);
        this->backbone_host_ptrs.push_back(h_ptr);
    }

    for (auto& bindings : this->output_bindings_head) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->head_device_ptrs.push_back(d_ptr);
        this->head_host_ptrs.push_back(h_ptr);
    }
}

void NanoTrack::init(cv::Mat img, cv::Rect bbox) {
    this->create_window();
    this->create_grids();

    cv::Point target_pos; // cx, cy
    cv::Point2f target_sz = {0.f, 0.f}; //w,h

    target_pos.x = bbox.x + bbox.width / 2;
    target_pos.y = bbox.y + bbox.height / 2;
    target_sz.x = bbox.width;
    target_sz.y = bbox.height;

    float wc_z = target_sz.x + cfg.context_amount * (target_sz.x + target_sz.y);
    float hc_z = target_sz.y + cfg.context_amount * (target_sz.x + target_sz.y);
    float s_z = round(sqrt(wc_z * hc_z));

    cv::Scalar avg_chans = cv::mean(img);
    cv::Mat z_crop;

    z_crop = get_subwindow_tracking(img, target_pos, cfg.exemplar_size, int(s_z), avg_chans); //cv::Mat BGR order

    float *input_blob = new float[3 * 127 * 127];
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 127; h++) {
            for (int w = 0; w < 127; w++) {
                input_blob[c * 127 * 127 + h * 127 + w] =
                        z_crop.at<cv::Vec3b>(h, w)[c];
            }
        }
    }

    //将127x127的图像数据传入init_backbone_engine进行推理
    //第一步将数据复制到给初始化模型分配的GPU内存中,init_device_ptrs[0]是输入数据的GPU内存指针,init_device_ptrs[1]是输出数据的GPU内存指针
    cudaMemcpy(init_device_ptrs[0], input_blob, input_bindings_init[0].size * input_bindings_init[0].dsize,
               cudaMemcpyHostToDevice);

    //第二步进行前向推理
    this->infer(init_device_ptrs, init_host_ptrs, num_inputs_init, num_outputs_init, output_bindings_init, 0);
    std::cout << "11" << std::endl;

    //推理之后,数据已经被复制到cpu指针指向的内存中
    this->init_output = static_cast<float *>(this->init_host_ptrs[0]); //1x48x8x8
    ///初始模型的out_put已经获取 将初始化模型的都free掉
    this->context_backbone_init->destroy();
    this->engine_backbone_init->destroy();

    //free掉初始化模型占用的GPU内存和CPU内存
    for (auto &ptr: this->init_device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto &ptr: this->init_host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }

    delete[] input_blob;

    this->state.channel_ave = avg_chans;
    this->state.im_h = img.rows;
    this->state.im_w = img.cols;
    this->state.target_pos = target_pos;
    this->state.target_sz = target_sz;
}



void NanoTrack::track(cv::Mat im)
{

    cv::Point target_pos = this->state.target_pos;
    cv::Point2f target_sz = this->state.target_sz;

    float hc_z = target_sz.y + cfg.context_amount * (target_sz.x + target_sz.y);
    float wc_z = target_sz.x + cfg.context_amount * (target_sz.x + target_sz.y);
    float s_z = sqrt(wc_z * hc_z);
    float scale_z = cfg.exemplar_size / s_z;

    float d_search = (cfg.instance_size - cfg.exemplar_size) / 2;
    float pad = d_search / scale_z;
    float s_x = s_z + 2*pad;

    cv::Mat x_crop;
    x_crop  = get_subwindow_tracking(im, target_pos, cfg.instance_size, int(s_x),state.channel_ave);

    // update
    target_sz.x = target_sz.x * scale_z;
    target_sz.y = target_sz.y * scale_z;

    float cls_score_max;

    this->update(x_crop, target_pos, target_sz, scale_z, cls_score_max);

    target_pos.x = std::max(0, std::min(state.im_w, target_pos.x));
    target_pos.y = std::max(0, std::min(state.im_h, target_pos.y));
    target_sz.x = float(std::max(10, std::min(state.im_w, int(target_sz.x))));
    target_sz.y = float(std::max(10, std::min(state.im_h, int(target_sz.y))));

    state.target_pos = target_pos;
    state.target_sz = target_sz;
}

void NanoTrack::update(const cv::Mat &x_crops, cv::Point &target_pos, cv::Point2f &target_sz,  float scale_z, float &cls_score_max)
{
//    auto start1 = std::chrono::high_resolution_clock::now();
    //这个程序段占用了大约一半的时间
    //这段程序进行优化 遍历给值的时间从3-4ms降低到了1-2ms
//    for (int c = 0; c < 3; c++) {
//        for (int h = 0; h < 255; h++) {
//            for (int w = 0; w < 255; w++) {
//                input_blob[c * 255 * 255 + h * 255 + w] =
//                        x_crops.at<cv::Vec3b>(h, w)[c];
//            }
//        }
//    }
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 255; h++) {
            const cv::Vec3b* row_ptr = x_crops.ptr<cv::Vec3b>(h);
            float* blob_ptr = input_blob + (c * 255 * 255) + (h * 255);

            for (int w = 0; w < 255; w++) {
                blob_ptr[w] = static_cast<float>(row_ptr[w][c]);
            }
        }
    }
//    auto end1 = std::chrono::high_resolution_clock::now();
//
//    // 计算时间差
//    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
//
//    // 输出运行时间（毫秒）
//    std::cout << "遍历给值的运行时间: " << duration1.count() << " 毫秒" << std::endl;

    //auto start = std::chrono::high_resolution_clock::now();
    ///将图像数据传入backbone中
    //第一步将数据复制到给backbone模型分配的GPU内存中,backbone_device_ptrs[0]是输入数据的GPU内存指针,backbone_device_ptrs[0]是输出数据的GPU内存指针
    cudaMemcpy(backbone_device_ptrs[0], input_blob, input_bindings_backbone[0].size * input_bindings_backbone[0].dsize, cudaMemcpyHostToDevice);
    //第二步进行前向推理
    this->infer(backbone_device_ptrs,backbone_host_ptrs,num_inputs_backbone,num_outputs_backbone,output_bindings_backbone,1);
    //推理之后,数据已经被复制到cpu指针指向的内存中
    this->backbone_output = static_cast<float*>(this->backbone_host_ptrs[0]); //1x48x16x16
    ///此时已经得到了两个输出 this->backbone_output(1x48x16x16,input_2) this->init_output(1x48x8x8,input_1)  this->backbone_output 随追踪不断更新
    ///这两个输出即为head_engine的输入 将这两个数据传输给head_engine
    //第一步将数据复制到给head模型分配的GPU内存中,backbone_device_ptrs[0]是输入数据的GPU内存指针,backbone_device_ptrs[0]是输出数据的GPU内存指针
    cudaMemcpy(head_device_ptrs[0],this->init_output , input_bindings_head[0].size * input_bindings_head[0].dsize, cudaMemcpyHostToDevice);

    cudaMemcpy(head_device_ptrs[1],this->backbone_output , input_bindings_head[1].size * input_bindings_head[1].dsize, cudaMemcpyHostToDevice);

    //第二步进行前向推理
    this->infer(head_device_ptrs,head_host_ptrs,num_inputs_head,num_outputs_head,output_bindings_head,2);
    //推理之后,数据已经被复制到cpu指针指向的内存中
    float* out_put1  = static_cast<float*>(this->head_host_ptrs[0]); //1X2x16x16  cls 保存了概率信息
    float* out_put2 = static_cast<float*>(this->head_host_ptrs[1]); //1X4x16x16   reg 保存了偏移量信息

    //auto end = std::chrono::high_resolution_clock::now();

    // 计算时间差
    //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 输出运行时间（毫秒）
    //std::cout << "engine运行时间: " << duration.count() << " 毫秒" << std::endl;


    auto start3 = std::chrono::high_resolution_clock::now();
    std::vector<float> cls_score_sigmoid;

    // float* cls_score_data = (float*)cls_score.data;
    //float* cls_score_data = cls_score.channel(1);

    cls_score_sigmoid.clear();

    int cols = 16;
    int rows = 16;

    for (int i = 0; i < 256; i++)   //
    {
        //cls_score_sigmoid.push_back(sigmoid(cls_score_data[i]));
        //cls_score.channel(1)对应output1第二个通道
        cls_score_sigmoid.push_back(sigmoid(out_put1[i+256]));
//        out_put1
    }

    std::vector<float> pred_x1(256, 0), pred_y1(256, 0), pred_x2(256, 0), pred_y2(256, 0);

    //bbox_pred.channel(i) 分比别对应output2的4个通道
//    float* bbox_pred_data1 = bbox_pred.channel(0);
//    float* bbox_pred_data2 = bbox_pred.channel(1);
//    float* bbox_pred_data3 = bbox_pred.channel(2);
//    float* bbox_pred_data4 = bbox_pred.channel(3);

    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {

            pred_x1[i*cols + j] = this->grid_to_search_x[i*cols + j] - out_put2[i*cols + j];
            pred_y1[i*cols + j] = this->grid_to_search_y[i*cols + j] - out_put2[i*cols + j+256];
            pred_x2[i*cols + j] = this->grid_to_search_x[i*cols + j] + out_put2[i*cols + j+512];
            pred_y2[i*cols + j] = this->grid_to_search_y[i*cols + j] + out_put2[i*cols + j+768];
        }
    }

    // size penalty
    std::vector<float> w(256, 0), h(256, 0);
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {
            w[i*cols + j] = pred_x2[i*cols + j] - pred_x1[i*cols + j];
            h[i*rows + j] = pred_y2[i*rows + j] - pred_y1[i*cols + j];
        }
    }

    float sz_wh = sz_whFun(target_sz);
    std::vector<float> s_c = sz_change_fun(w, h, sz_wh);
    std::vector<float> r_c = ratio_change_fun(w, h, target_sz);

    std::vector<float> penalty(rows*cols,0);
    for (int i = 0; i < rows * cols; i++)
    {
        penalty[i] = std::exp(-1 * (s_c[i] * r_c[i]-1) * cfg.penalty_k);
    }

    // window penalty
    std::vector<float> pscore(rows*cols,0);
    int r_max = 0, c_max = 0;
    float maxScore = 0;

    for (int i = 0; i < rows * cols; i++)
    {
        pscore[i] = (penalty[i] * cls_score_sigmoid[i]) * (1 - cfg.window_influence) + this->window[i] * cfg.window_influence;
        if (pscore[i] > maxScore)
        {
            // get max
            maxScore = pscore[i];
            r_max = std::floor(i / rows);
            c_max = ((float)i / rows - r_max) * rows;
        }
    }

    // to real size
    float pred_x1_real = pred_x1[r_max * cols + c_max]; // pred_x1[r_max, c_max]
    float pred_y1_real = pred_y1[r_max * cols + c_max];
    float pred_x2_real = pred_x2[r_max * cols + c_max];
    float pred_y2_real = pred_y2[r_max * cols + c_max];

    float pred_xs = (pred_x1_real + pred_x2_real) / 2;
    float pred_ys = (pred_y1_real + pred_y2_real) / 2;
    float pred_w = pred_x2_real - pred_x1_real;
    float pred_h = pred_y2_real - pred_y1_real;

    float diff_xs = pred_xs - cfg.instance_size / 2;
    float diff_ys = pred_ys - cfg.instance_size / 2;

    diff_xs /= scale_z;
    diff_ys /= scale_z;
    pred_w /= scale_z;
    pred_h /= scale_z;

    target_sz.x = target_sz.x / scale_z;
    target_sz.y = target_sz.y / scale_z;

    // size learning rate
    float lr = penalty[r_max * cols + c_max] * cls_score_sigmoid[r_max * cols + c_max] * cfg.lr;

    // size rate
    auto res_xs = float (target_pos.x + diff_xs);
    auto res_ys = float (target_pos.y + diff_ys);
    float res_w = pred_w * lr + (1 - lr) * target_sz.x;
    float res_h = pred_h * lr + (1 - lr) * target_sz.y;

    target_pos.x = int(res_xs);
    target_pos.y = int(res_ys);

    target_sz.x = target_sz.x * (1 - lr) + lr * res_w;
    target_sz.y = target_sz.y * (1 - lr) + lr * res_h;

    cls_score_max = cls_score_sigmoid[r_max * cols + c_max];
    //auto end3 = std::chrono::high_resolution_clock::now();

    // 计算时间差
    //auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

}

cv::Mat NanoTrack::convert_score(float* out_put1){
    // 假设 score_data 是一个 float* 指针，指向模型输出的数组
    float* score_data = out_put1;

    int num_pixels = 16; // 假设 width 和 height 是图像的宽度和高度

    // 将 score_data 重新组织为 cv::Mat，假设通道数是 2
    cv::Mat score_mat(16, 16, CV_32FC2, score_data);

    // 调整数据的通道次序，以匹配 Python 代码中的 permute(1, 2, 3, 0)
    cv::Mat score_transposed;
    cv::transpose(score_mat, score_transposed);

    // 使用 reshape 以匹配 Python 代码中的 view(2, -1).permute(1, 0)
    cv::Mat score_reshaped = score_transposed.reshape(1, 2 * num_pixels);

    // 计算 softmax
    cv::Mat score_softmax;
    cv::Mat score_softmax_result;

    cv::exp(score_reshaped, score_softmax);
    cv::reduce(score_softmax, score_softmax_result, 1, cv::REDUCE_SUM, CV_32F);

    // 获取第 1 个类别的概率
    cv::Mat score_class_1 = score_softmax_result.col(1);
    return score_class_1;
}

cv::Rect NanoTrack::convert_bbox(float* reg, cv::Point target_pos){


}

// 生成每一个格点的坐标
void NanoTrack::create_window()
{
    int score_size= cfg.score_size;
    std::vector<float> hanning(score_size,0);
    this->window.resize(score_size*score_size, 0);

    for (int i = 0; i < score_size; i++)
    {
        float w = 0.5f - 0.5f * std::cos(2 * 3.1415926535898f * i / (score_size - 1));
        hanning[i] = w;
    }
    for (int i = 0; i < score_size; i++)
    {
        for (int j = 0; j < score_size; j++)
        {
            this->window[i*score_size+j] = hanning[i] * hanning[j];
        }
    }
}

// 生成每一个格点的坐标
void NanoTrack::create_grids()
{
    /*
    each element of feature map on input search image
    :return: H*W*2 (position for each element)
    */
    int sz = cfg.score_size;   //16x16

    this->grid_to_search_x.resize(sz * sz, 0);
    this->grid_to_search_y.resize(sz * sz, 0);

    for (int i = 0; i < sz; i++)
    {
        for (int j = 0; j < sz; j++)
        {
            this->grid_to_search_x[i*sz+j] = j*cfg.total_stride;
            this->grid_to_search_y[i*sz+j] = i*cfg.total_stride;
        }
    }
}

cv::Mat NanoTrack::get_subwindow_tracking(cv::Mat im, cv::Point2f pos, int model_sz, int original_sz,cv::Scalar channel_ave)
{
    float c = (float)(original_sz + 1) / 2;
    int context_xmin = std::round(pos.x - c);
    int context_xmax = context_xmin + original_sz - 1;
    int context_ymin = std::round(pos.y - c);
    int context_ymax = context_ymin + original_sz - 1;

    int left_pad = int(std::max(0, -context_xmin));
    int top_pad = int(std::max(0, -context_ymin));
    int right_pad = int(std::max(0, context_xmax - im.cols + 1));
    int bottom_pad = int(std::max(0, context_ymax - im.rows + 1));

    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;
    cv::Mat im_path_original;

    if (top_pad > 0 || left_pad > 0 || right_pad > 0 || bottom_pad > 0)
    {
        cv::Mat te_im = cv::Mat::zeros(im.rows + top_pad + bottom_pad, im.cols + left_pad + right_pad, CV_8UC3);

        cv::copyMakeBorder(im, te_im, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, channel_ave);
        im_path_original = te_im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
    }
    else
        im_path_original = im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));

    cv::Mat im_path;
    cv::resize(im_path_original, im_path, cv::Size(model_sz, model_sz));

    return im_path;
}



//void NanoTrack::infer()
//{
//    // this->context->enqueueV2：使用TensorRT执行上下文 this->context 对象执行推理操作。
//    // 这是实际的推理步骤，它将输入数据传递给模型并执行模型的前向传播。
//    // 具体来说，enqueueV2 接受以下参数：
//    // this->device_ptrs.data()：包含了输入数据的GPU内存指针数组。这些指针指向了经过分配的GPU内存中的输入数据。
//    // this->stream：CUDA流，用于异步执行推理操作。
//    // nullptr：这里为了简化没有传递其他回调函数。
//    this->context_backbone_init->enqueueV2(this->device_ptrs.data(), this->stream, nullptr);
//
//    ///循环处理输出数据
//    //对于NanoTrack的主干网络,输出数据只有一个
//    for (int i = 0; i < this->num_outputs; i++) {
//        //对于每一个输出绑定,计算输出张量的大小,该大小为张量所有元素个数乘以每个元素所占用的字节数
//        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
//
//        //使用 cudaMemcpyAsync 将模型的输出数据从GPU内存（this->device_ptrs[i + this->num_inputs]，其中 i 是当前输出绑定的索引）
//        // 异步复制到CPU内存（this->host_ptrs[i]，其中 i 是当前输出绑定的索引）。
//        //这个步骤用于将模型的输出数据从GPU内存传输到CPU内存，以便进一步处理和分析。
//        CHECK(cudaMemcpyAsync(
//                this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
//    }
//
//    //等待CUDA流中的所有操作完成。这是必要的，以确保在使用模型的输出数据之前，所有数据都已正确地从GPU传输到CPU。
//    cudaStreamSynchronize(this->stream);
//}


#endif //NANO_TRACK_TRT_TEXT_NANO_TRACK_H

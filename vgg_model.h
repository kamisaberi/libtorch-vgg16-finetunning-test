#pragma once
#include <torch/torch.h>

// This struct is the entire VGG16 features block.
// It doesn't need to be nested.
struct VGGFeaturesImpl : torch::nn::Module {
    torch::nn::Sequential features_seq;
    VGGFeaturesImpl() {
        torch::nn::Sequential features;
        int64_t in_channels = 3;
        // Helper lambda to reduce code duplication
        auto add_block = [&](int64_t out_channels, int num_convs) {
            for (int i = 0; i < num_convs; ++i) {
                features->push_back(torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)));
                features->push_back(torch::nn::BatchNorm2d(out_channels));
                features->push_back(torch::nn::ReLU(true));
                in_channels = out_channels;
            }
            features->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
        };

        add_block(64, 2);
        add_block(128, 2);
        add_block(256, 3);
        add_block(512, 3);
        add_block(512, 3);

        features_seq = features;
        register_module("features_seq", features_seq);
    }

    torch::Tensor forward(torch::Tensor x) {
        return features_seq->forward(x);
    }
};
TORCH_MODULE(VGGFeatures);


struct Food101VGG16Impl : torch::nn::Module {
    VGGFeatures features{nullptr}; // This will be the module named "features"
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Sequential classifier{nullptr};

    Food101VGG16Impl(int64_t num_classes = 101) {
        features = VGGFeatures();
        // The python state dict keys start with "features.LAYER_NUM.PARAM"
        // So we must register our VGGFeatures module with the name "features"
        register_module("features", features->features_seq);

        avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(7));
        register_module("avgpool", avgpool);

        // The python state dict keys are "classifier.LAYER_NUM.PARAM"
        classifier = torch::nn::Sequential(
            torch::nn::Linear(512 * 7 * 7, 4096),
            torch::nn::ReLU(true),
            torch::nn::Dropout(0.5),
            torch::nn::Linear(4096, 4096),
            torch::nn::ReLU(true),
            torch::nn::Dropout(0.5),
            torch::nn::Linear(4096, num_classes)
        );
        register_module("classifier", classifier);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = features->forward(x);
        x = avgpool->forward(x);
        x = torch::flatten(x, 1);
        x = classifier->forward(x);
        return x;
    }
};
TORCH_MODULE(Food101VGG16);
#include "vgg_model.h"
#include "Food101Dataset.h"
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <chrono>

using WeightMap = std::map<std::string, torch::Tensor>;

WeightMap load_weights_from_scratch(const std::string& path) {
    std::cout << "Loading weights from scratch from: " << path << std::endl;
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open weights file: " + path);
    }

    WeightMap weight_map;

    while (file.peek() != EOF) {
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        file.read(&name[0], name_len);
        uint64_t num_elements;
        file.read(reinterpret_cast<char*>(&num_elements), sizeof(num_elements));
        size_t data_size_bytes = num_elements * sizeof(float);
        std::vector<float> data(num_elements);
        file.read(reinterpret_cast<char*>(data.data()), data_size_bytes);
        torch::Tensor tensor = torch::from_blob(data.data(), {static_cast<long>(num_elements)}, torch::kFloat32).clone();
        weight_map[name] = tensor;
    }
    std::cout << "Finished loading " << weight_map.size() << " tensors from file." << std::endl;
    return weight_map;
}


int main() {
    torch::Device device(torch::kCUDA);
    Food101VGG16 model(101);
    model->to(device);

    const std::string WEIGHTS_PATH = "../vgg16_weights.bin";

    try {
        WeightMap loaded_weights = load_weights_from_scratch(WEIGHTS_PATH);

        torch::NoGradGuard no_grad;
        int count = 0;
        for (auto& pair : model->named_parameters()) {
            const std::string& name = pair.key();
            torch::Tensor& param = pair.value();
            if (loaded_weights.count(name)) {
                param.copy_(loaded_weights[name].reshape(param.sizes()));
                count++;
            }
        }
        for (auto& pair : model->named_buffers()) {
            const std::string& name = pair.key();
            torch::Tensor& buffer = pair.value();
            if (loaded_weights.count(name)) {
                buffer.copy_(loaded_weights[name].reshape(buffer.sizes()));
            }
        }
        std::cout << "Manually copied " << count << " parameters into the model." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    // --- 4. Freeze Layers and Prepare Optimizer ---
    // NO MORE MISTAKES. Access the public member 'features' directly.
    for (auto& param : model->features->parameters()) {
        param.set_requires_grad(false);
    }

    std::vector<torch::Tensor> params_to_update;
    // Access the public member 'classifier' directly.
    for (auto& param : model->classifier->parameters()) {
        params_to_update.push_back(param);
    }
    torch::optim::Adam optimizer(params_to_update, torch::optim::AdamOptions(0.001));

    // --- 5. Data Loader ---
    auto train_dataset = Food101Dataset("../food-101", "train").map(torch::data::transforms::Stack<>());
    auto data_loader_options = torch::data::DataLoaderOptions().batch_size(64).workers(8);
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), data_loader_options);

    // --- 6. Fine-Tuning Loop ---
    std::cout << "\nStarting C++ fine-tuning from scratch..." << std::endl;
    for (int epoch = 0; epoch < 3; ++epoch) {
        model->train();
        double running_loss = 0.0;
        for (auto& batch : *train_loader) {
            auto inputs = batch.data.to(device);
            auto labels = batch.target.to(device);
            optimizer.zero_grad();
            auto outputs = model->forward(inputs);
            auto loss = torch::cross_entropy_loss(outputs, labels);
            loss.backward();
            optimizer.step();
            running_loss += loss.item().toDouble() * inputs.size(0);
        }
        std::cout << "Epoch " << epoch + 1 << " Loss: " << running_loss / train_dataset.size().value() << std::endl;
    }

    std::cout << "\nFine-tuning complete." << std::endl;
    torch::save(model, "food101_vgg16_finetuned_model.pt");

    return 0;
}
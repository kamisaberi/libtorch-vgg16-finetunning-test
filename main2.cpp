#include "vgg_model.h"
// #include "Food101Dataset.h"
#include <torch/torch.h>
#include <xtorch/xtorch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <chrono>

using WeightMap = std::map<std::string, torch::Tensor>;

WeightMap load_weights_from_scratch(const std::string& path) {
    std::cout << "Loading weights from robust binary file: " << path << std::endl;
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
        uint8_t type_code;
        file.read(reinterpret_cast<char*>(&type_code), sizeof(type_code));
        uint64_t num_elements;
        file.read(reinterpret_cast<char*>(&num_elements), sizeof(num_elements));
        torch::Tensor tensor;
        switch (type_code) {
            case 0: {
                std::vector<float> data(num_elements);
                file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(float));
                tensor = torch::from_blob(data.data(), {static_cast<long>(num_elements)}, torch::kFloat32).clone();
                break;
            }
            case 1: {
                std::vector<int64_t> data(num_elements);
                file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(int64_t));
                tensor = torch::from_blob(data.data(), {static_cast<long>(num_elements)}, torch::kInt64).clone();
                break;
            }
            default:
                throw std::runtime_error("Unknown data type code in weights file: " + std::to_string(type_code));
        }
        weight_map[name] = tensor;
    }
    std::cout << "Finished loading " << weight_map.size() << " tensors from file." << std::endl;
    return weight_map;
}


int main() {
    const int NUM_CLASSES = 101;
    const int NUM_EPOCHS = 3;
    const int BATCH_SIZE = 64;
    const double LEARNING_RATE = 0.001;

    torch::Device device(torch::kCUDA);
    Food101VGG16 model(101);
    model->to(device);

    const std::string WEIGHTS_PATH = "../vgg16_weights.bin";

    try {
        WeightMap loaded_weights = load_weights_from_scratch(WEIGHTS_PATH);

        torch::NoGradGuard no_grad;
        int copied_params = 0;
        int skipped_params = 0;

        for (auto& pair : model->named_parameters()) {
            const std::string& name = pair.key();
            torch::Tensor& param = pair.value();

            if (loaded_weights.count(name)) {
                const torch::Tensor& loaded_tensor = loaded_weights.at(name);

                // ================== THE CRITICAL FIX IS HERE ==================
                // ONLY copy the weights if the number of elements is the same.
                if (param.numel() == loaded_tensor.numel()) {
                    param.copy_(loaded_tensor.reshape(param.sizes()));
                    copied_params++;
                } else {
                    // This will happen for the final classifier layer, which is EXPECTED.
                    std::cout << "Skipping parameter '" << name << "' due to shape mismatch." << std::endl;
                    skipped_params++;
                }
                // =============================================================

            }
        }
        for (auto& pair : model->named_buffers()) {
            const std::string& name = pair.key();
            torch::Tensor& buffer = pair.value();
            if (loaded_weights.count(name)) {
                 // Buffers should always match in size.
                buffer.copy_(loaded_weights.at(name).reshape(buffer.sizes()));
            }
        }
        std::cout << "Manually copied " << copied_params << " parameters and skipped " << skipped_params << "." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    // --- The rest of the code is unchanged and correct ---
    for (auto& param : model->features->parameters()) {
        param.set_requires_grad(false);
    }

    std::vector<torch::Tensor> params_to_update;
    for (auto& param : model->classifier->parameters()) {
        params_to_update.push_back(param);
    }
    torch::optim::Adam optimizer(params_to_update, torch::optim::AdamOptions(0.001));



    // auto train_dataset = Food101Dataset("../food-101", "train").map(torch::data::transforms::Stack<>());
    // auto data_loader_options = torch::data::DataLoaderOptions().batch_size(64).workers(8);
    // auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    //     std::move(train_dataset), data_loader_options);

    std::vector<std::shared_ptr<xt::Module>> transform_list;
    transform_list.push_back(std::make_shared<xt::transforms::image::Resize>(std::vector<int64_t>{224, 224}));
    transform_list.push_back(
        std::make_shared<xt::transforms::general::Normalize>(std::vector<float>{0.5, 0.5, 0.5},
                                                             std::vector<float>{0.5, 0.5, 0.5}));
    auto compose = std::make_unique<xt::transforms::Compose>(transform_list);
    auto dataset = xt::datasets::Food101("/home/kami/Documents/datasets/", xt::datasets::DataMode::TRAIN, false,true,
                                         std::move(compose));
    xt::dataloaders::ExtendedDataLoader data_loader(dataset, BATCH_SIZE, true, 32, 20);



    std::cout << "\nStarting C++ fine-tuning from scratch..." << std::endl;
    for (int epoch = 0; epoch < 3; ++epoch) {
        model->train();
        double running_loss = 0.0;
        for (auto& batch : data_loader) {
            auto inputs = batch.first.to(device);
            auto labels = batch.second.to(device);
            optimizer.zero_grad();
            auto outputs = model->forward(inputs);
            auto loss = torch::cross_entropy_loss(outputs, labels);
            loss.backward();
            optimizer.step();
            running_loss += loss.item().toDouble() * inputs.size(0);
        }
        // std::cout << "Epoch " << epoch + 1 << " Loss: " << running_loss / train_dataset.size().value() << std::endl;
    }

    std::cout << "\nFine-tuning complete." << std::endl;
    torch::save(model, "food101_vgg16_finetuned_model.pt");

    return 0;
}
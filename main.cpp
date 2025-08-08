#include "vgg_model.h"
// #include "Food101Dataset.h"
#include <torch/torch.h>
#include <xtorch/xtorch.h>
#include <iostream>
#include <chrono>

int main() {
    // --- 1. Configuration ---
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using device: CUDA" << std::endl;
    }

    // const std::string DATA_ROOT = "../food-101";
    const std::string WEIGHTS_PATH = "vgg16_bn_weights.pt";
    const int NUM_CLASSES = 101;
    const int NUM_EPOCHS = 3;
    const int BATCH_SIZE = 64;
    const double LEARNING_RATE = 0.001;

    // --- 2. Create Native C++ Model ---
    Food101VGG16 model(NUM_CLASSES);
    model->to(device);
    std::cout << "Native C++ VGG16 model created and moved to device." << std::endl;

    // --- 3. Load Pre-trained Weights (Modern, Non-Strict Method) ---
    // This is the cleanest way and should be supported by a modern LibTorch version.
    // try {
    //     torch::serialize::InputArchive archive;
    //     archive.load_from(WEIGHTS_PATH);
    //     model->load(archive); // Using the `strict` flag on the model's load method
    //     std::cout << "Successfully loaded pre-trained weights." << std::endl;
    // } catch (const c10::Error& e) {
    //     std::cerr << "Error loading weights: " << e.msg() << std::endl;
    //     return -1;
    // }


    try {
        std::cout << "1\n";
        torch::serialize::InputArchive archive;
        // This loads the raw data from the .pt file

        archive.load_from(WEIGHTS_PATH);

        std::cout << "2\n";
        // Get all named parameters from our C++ model
        auto params = model->named_parameters(true /*recurse*/);
        // Get all named buffers (for batch norm running_mean, etc.)
        auto buffers = model->named_buffers(true /*recurse*/);

        std::cout << "3\n";
        // Manually iterate and load each parameter/buffer by name
        for (auto& val : params) {
            archive.try_read(val.key(), val.value());
        }
        std::cout << "4\n";
        for (auto& val : buffers) {
            archive.try_read(val.key(), val.value());
        }
        std::cout << "5\n";
        std::cout << "Successfully loaded weights and buffers from state_dict file." << std::endl;

    } catch (const c10::Error& e) {
        std::cerr << "Error loading weights: " << e.msg() << std::endl;
        return -1;
    }

    // --- 4. Freeze Layers and Prepare Optimizer ---
    for (auto& param : model->features->parameters()) {
        param.set_requires_grad(false);
    }

    std::vector<torch::Tensor> params_to_update;
    for (auto& param : model->classifier->parameters()) {
        params_to_update.push_back(param);
    }
    torch::optim::Adam optimizer(params_to_update, torch::optim::AdamOptions(LEARNING_RATE));




    std::vector<std::shared_ptr<xt::Module>> transform_list;
    transform_list.push_back(std::make_shared<xt::transforms::image::Resize>(std::vector<int64_t>{224, 224}));
    transform_list.push_back(
        std::make_shared<xt::transforms::general::Normalize>(std::vector<float>{0.5, 0.5, 0.5},
                                                             std::vector<float>{0.5, 0.5, 0.5}));
    auto compose = std::make_unique<xt::transforms::Compose>(transform_list);
    auto dataset = xt::datasets::Food101("/home/kami/Documents/datasets/", xt::datasets::DataMode::TRAIN, false,true,
                                         std::move(compose));
    xt::dataloaders::ExtendedDataLoader data_loader(dataset, BATCH_SIZE, true, 32, 20);




    // // --- 5. Create Data Loader ---
    // auto train_dataset = Food101Dataset(DATA_ROOT, "train").map(torch::data::transforms::Stack<>());
    // size_t dataset_size = train_dataset.size().value();
    // auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    //     std::move(train_dataset),
    //     torch::data::DataLoaderOptions().batch_size(BATCH_SIZE).workers(8));
    // std::cout << "Training data loader created with " << dataset_size << " images." << std::endl;

    // --- 6. Fine-Tuning Loop ---
    std::cout << "\nStarting C++ fine-tuning with native model..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        model->train();
        double running_loss = 0.0;
        int64_t num_correct = 0;
        int64_t batch_idx = 0;

        for (auto& batch : data_loader) {
            auto inputs = batch.first.to(device, true);
            auto labels = batch.second.to(device, true);

            // Use torch::NoGradGuard for clarity when not needed
            {
                torch::autograd::AutoGradMode guard(false); // An alternative way to disable grads
                // For inference part if any
            }

            optimizer.zero_grad();
            auto outputs = model->forward(inputs);
            auto loss = torch::cross_entropy_loss(outputs, labels);

            loss.backward();
            optimizer.step();

            running_loss += loss.item<double>() * inputs.size(0);
            auto preds = torch::argmax(outputs, 1);
            num_correct += torch::sum(preds == labels).item<int64_t>();

            if (++batch_idx % 100 == 0) {
                 std::cout << "  Epoch [" << epoch + 1 << "/" << NUM_EPOCHS
                          << "], Loss: " << loss.item<double>() << std::endl;
            }
        }

        // double epoch_loss = running_loss / dataset_size;
        // double epoch_acc = static_cast<double>(num_correct) / dataset_size;
        // std::cout << "Epoch " << epoch + 1 << " Summary -> Loss: " << epoch_loss << " | Accuracy: " << epoch_acc << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "\nFine-tuning complete in " << elapsed.count() << " seconds." << std::endl;

    torch::save(model, "food101_vgg16_finetuned_model.pt");

    return 0;
}
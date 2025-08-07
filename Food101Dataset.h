#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <fstream>
#include <iostream>

// Include libjpeg-turbo header for high-performance decoding
#include <jpeglib.h>
#include <csetjmp>

// Error handling for libjpeg-turbo
struct my_error_mgr { struct jpeg_error_mgr pub; jmp_buf setjmp_buffer; };
void my_error_exit(j_common_ptr cinfo) {
    my_error_mgr* myerr = (my_error_mgr*) cinfo->err;
    longjmp(myerr->setjmp_buffer, 1);
}

// High-performance JPEG reader using libjpeg-turbo
// Decodes directly into a pre-allocated cv::Mat
bool read_jpeg_into_mat(const std::string& filename, cv::Mat& output_mat) {
    struct jpeg_decompress_struct cinfo;
    struct my_error_mgr jerr;

    FILE* infile = fopen(filename.c_str(), "rb");
    if (!infile) { return false; }

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return false;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);

    // VGG expects BGR, so we can ask libjpeg-turbo to do the conversion!
    cinfo.out_color_space = JCS_EXT_BGR;

    jpeg_start_decompress(&cinfo);

    output_mat.create(cinfo.output_height, cinfo.output_width, CV_8UC3);

    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW row_pointer = (JSAMPROW)output_mat.ptr(cinfo.output_scanline);
        jpeg_read_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return true;
}

class Food101Dataset : public torch::data::Dataset<Food101Dataset> {
private:
    std::vector<std::tuple<std::string, int64_t>> image_info_;
    torch::Tensor mean_, std_;

public:
    explicit Food101Dataset(const std::string& root_path, const std::string& split = "train") {
        // ... (Parsing logic from previous example remains the same) ...
        std::map<std::string, int64_t> class_to_idx;
        std::ifstream classes_file(root_path + "/meta/classes.txt");
        std::string class_name;
        int64_t idx = 0;
        while (std::getline(classes_file, class_name)) { class_to_idx[class_name] = idx++; }

        std::ifstream split_file(root_path + "/meta/" + split + ".txt");
        std::string line;
        while (std::getline(split_file, line)) {
            size_t slash_pos = line.find('/');
            std::string image_path = root_path + "/images/" + line + ".jpg";
            int64_t label = class_to_idx[line.substr(0, slash_pos)];
            image_info_.emplace_back(image_path, label);
        }

        // VGG uses these normalization constants
        mean_ = torch::tensor({0.485, 0.456, 0.406}).view({3, 1, 1});
        std_ = torch::tensor({0.229, 0.224, 0.225}).view({3, 1, 1});
    }

    torch::data::Example<> get(size_t index) override {
        std::string image_path = std::get<0>(image_info_[index]);
        int64_t label = std::get<1>(image_info_[index]);

        cv::Mat image;
        // Use high-performance libjpeg-turbo reader
        if (!read_jpeg_into_mat(image_path, image)) {
            std::cerr << "Error reading image: " << image_path << std::endl;
            // Return a dummy tensor on failure
            return {torch::zeros({3, 224, 224}), torch::tensor(-1L)};
        }

        // Transforms
        cv::resize(image, image, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
        // Note: No cvtColor needed, libjpeg-turbo gave us BGR.

        auto image_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kByte);
        image_tensor = image_tensor.permute({2, 0, 1}); // HWC to CHW
        image_tensor = image_tensor.to(torch::kFloat32).div(255.0);
        image_tensor = image_tensor.sub_(mean_).div_(std_);

        return {image_tensor, torch::tensor(label, torch::kInt64)};
    }

    torch::optional<size_t> size() const override {
        return image_info_.size();
    }
};
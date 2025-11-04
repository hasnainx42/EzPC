#include <torch/torch.h>
#include <torch/script.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace secure {

// -----------------------------------------------------------------------------
// Lightweight mock encryption utilities.
// In an actual BOLT deployment this would wrap homomorphic encryption or MPC
// transports so that the router never observes raw embeddings in the clear.
// -----------------------------------------------------------------------------
struct EncryptedVector {
    std::vector<float> masked_values;
    float mask;
};

class Encryptor {
  public:
    Encryptor() : rng_(Seed()) {}

    EncryptedVector EncryptVector(const std::vector<float> &input) {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        float mask = dist(rng_);
        std::vector<float> masked(input.size());
        std::transform(input.begin(), input.end(), masked.begin(),
                       [mask](float value) { return value + mask; });
        return {std::move(masked), mask};
    }

    std::vector<float> DecryptVector(const EncryptedVector &enc) {
        std::vector<float> plain(enc.masked_values.size());
        std::transform(enc.masked_values.begin(), enc.masked_values.end(),
                       plain.begin(),
                       [mask = enc.mask](float value) { return value - mask; });
        return plain;
    }

  private:
    static std::mt19937 Seed() {
        const auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
        return std::mt19937(static_cast<uint32_t>(now.count()));
    }

    std::mt19937 rng_;
};

} // namespace secure

using secure::EncryptedVector;

namespace {

secure::Encryptor &GetEncryptor() {
    static secure::Encryptor encryptor;
    return encryptor;
}

} // namespace

EncryptedVector EncryptVector(const std::vector<float> &input) {
    return GetEncryptor().EncryptVector(input);
}

std::vector<float> DecryptVector(const EncryptedVector &enc) {
    return GetEncryptor().DecryptVector(enc);
}

// -----------------------------------------------------------------------------
// Secure routing primitives.
// -----------------------------------------------------------------------------

// In production this tensor would never exist in plaintext on the host CPU.
// Instead the encrypted embeddings would be consumed by BOLT's secure linear
// layers. Here we simply simulate that behaviour to keep the sample compact.
torch::Tensor SecureForward(const torch::Tensor &llm_vec,
                            const torch::Tensor &query_vec) {
    TORCH_CHECK(llm_vec.sizes() == query_vec.sizes(),
                "SecureForward expects equally shaped tensors");

    // --- Begin simulated BOLT secure computation block ----------------------
    // The interaction tensor below is formed from element-wise products and
    // differences. In a BOLT deployment each operation would be implemented as
    // privacy-preserving linear / non-linear layers. These comments highlight
    // where those implementations would be plugged in.
    torch::Tensor product = llm_vec * query_vec;             // Secure mul
    torch::Tensor distance = (llm_vec - query_vec).pow(2);   // Secure sub + square
    torch::Tensor combined = torch::cat({llm_vec, query_vec, product, distance},
                                        llm_vec.dim() - 1);
    // Optional normalisation to stabilise inference.
    combined = torch::nn::functional::normalize(
        combined,
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(combined.dim() - 1));
    // --- End simulated BOLT secure computation block ------------------------

    return combined;
}

std::string RouteSecurely(const torch::Tensor &llm_vec,
                          const torch::Tensor &query_vec,
                          const std::string &model_path = "mirt_bert.snapshot") {
    torch::Tensor fused = SecureForward(llm_vec, query_vec);

    // Attempt to load the pretrained router. If unavailable, fall back to a
    // deterministic heuristic so the demo can run end-to-end.
    torch::jit::script::Module router_module;
    bool model_loaded = false;
    try {
        router_module = torch::jit::load(model_path);
        router_module.eval();
        model_loaded = true;
    } catch (const c10::Error &e) {
        std::cerr << "[secure-router] Warning: Unable to load " << model_path
                  << ". Falling back to heuristic routing.\n";
    }

    torch::Tensor logits;
    if (model_loaded) {
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(fused);
        try {
            logits = router_module.forward(inputs).toTensor();
        } catch (const c10::Error &e) {
            std::cerr << "[secure-router] Warning: Forward pass failed (" << e.what()
                      << "). Falling back to heuristic routing.\n";
            model_loaded = false;
        }
    }

    if (!model_loaded) {
        // Simple cosine-similarity heuristic between llm and query vectors.
        auto norm_options = torch::nn::functional::NormalizeFuncOptions().p(2).dim(llm_vec.dim() - 1);
        torch::Tensor norm_llm = torch::nn::functional::normalize(llm_vec, norm_options);
        torch::Tensor norm_query = torch::nn::functional::normalize(query_vec, norm_options);
        logits = (norm_llm * norm_query).sum(llm_vec.dim() - 1, true);
    }

    // Pick the highest scoring backend index. Only the ID is shared back.
    auto max_result = logits.squeeze().argmax();
    int64_t route_id = max_result.item<int64_t>();
    return "llm_backend_" + std::to_string(route_id);
}

// -----------------------------------------------------------------------------
// I/O helpers (used by the optional test harness below).
// -----------------------------------------------------------------------------
std::vector<float> ReadEmbedding(const std::string &path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to open embedding file: " + path);
    }

    std::vector<float> values;
    std::string line;
    while (std::getline(input, line)) {
        std::stringstream ss(line);
        float value;
        while (ss >> value) {
            values.push_back(value);
        }
    }
    return values;
}

torch::Tensor VectorToTensor(const std::vector<float> &values) {
    torch::Tensor tensor = torch::from_blob(const_cast<float *>(values.data()),
                                            {(int64_t)1, (int64_t)values.size()},
                                            torch::TensorOptions().dtype(torch::kFloat));
    return tensor.clone();
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <query_embedding.txt> <llm_embedding.txt> [--model path]" << std::endl;
        return 1;
    }

    std::string query_path = argv[1];
    std::string llm_path = argv[2];
    std::string model_path = "mirt_bert.snapshot";

    if (argc == 5) {
        std::string flag = argv[3];
        if (flag != "--model") {
            std::cerr << "Unknown flag: " << flag << std::endl;
            return 1;
        }
        model_path = argv[4];
    }

    try {
        std::vector<float> query_embedding = ReadEmbedding(query_path);
        std::vector<float> llm_embedding = ReadEmbedding(llm_path);

        if (query_embedding.size() != llm_embedding.size()) {
            throw std::runtime_error("Embeddings must be the same dimensionality.");
        }

        EncryptedVector enc_query = EncryptVector(query_embedding);
        EncryptedVector enc_llm = EncryptVector(llm_embedding);

        // The server only sees encrypted vectors. Within the secure enclave,
        // they are decrypted and consumed by secure layers. This split mirrors
        // BOLT's client/server trust model.
        std::vector<float> dec_query = DecryptVector(enc_query);
        std::vector<float> dec_llm = DecryptVector(enc_llm);

        torch::Tensor query_tensor = VectorToTensor(dec_query);
        torch::Tensor llm_tensor = VectorToTensor(dec_llm);

        std::string route = RouteSecurely(llm_tensor, query_tensor, model_path);
        std::cout << "Selected secure route: " << route << std::endl;
    } catch (const std::exception &ex) {
        std::cerr << "[secure-router] Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}


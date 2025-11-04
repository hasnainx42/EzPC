#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
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
// Secure routing primitives implemented with lightweight tensor helpers.
// -----------------------------------------------------------------------------
namespace lite {

float Dot(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Dot product requires vectors of equal length");
    }
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
}

float SquaredL2Norm(const std::vector<float> &values) {
    float sum = 0.0f;
    for (float v : values) {
        sum += v * v;
    }
    return sum;
}

std::vector<float> Normalize(const std::vector<float> &values) {
    std::vector<float> normalised(values.begin(), values.end());
    float norm = std::sqrt(SquaredL2Norm(normalised));
    if (norm > 1e-12f) {
        for (float &v : normalised) {
            v /= norm;
        }
    }
    return normalised;
}

std::vector<float> ElementWiseMultiply(const std::vector<float> &a,
                                       const std::vector<float> &b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Element-wise multiply expects equal length vectors");
    }
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] * b[i];
    }
    return out;
}

std::vector<float> ElementWiseSquareDifference(const std::vector<float> &a,
                                               const std::vector<float> &b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Element-wise difference expects equal length vectors");
    }
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        out[i] = diff * diff;
    }
    return out;
}

std::vector<float> Concatenate(const std::vector<std::vector<float>> &parts) {
    size_t total = 0;
    for (const auto &part : parts) {
        total += part.size();
    }
    std::vector<float> combined;
    combined.reserve(total);
    for (const auto &part : parts) {
        combined.insert(combined.end(), part.begin(), part.end());
    }
    return combined;
}

} // namespace lite

std::vector<float> SecureForward(const std::vector<float> &llm_vec,
                                 const std::vector<float> &query_vec) {
    if (llm_vec.size() != query_vec.size()) {
        throw std::invalid_argument("SecureForward expects equally shaped vectors");
    }

    // --- Begin simulated BOLT secure computation block ----------------------
    // Each operation below models the behaviour of a privacy-preserving layer
    // that would live inside the BOLT enclave.
    std::vector<float> product = lite::ElementWiseMultiply(llm_vec, query_vec);      // Secure mul
    std::vector<float> distance = lite::ElementWiseSquareDifference(llm_vec, query_vec); // Secure sub + square
    std::vector<float> combined =
        lite::Concatenate({llm_vec, query_vec, product, distance});
    combined = lite::Normalize(combined);
    // --- End simulated BOLT secure computation block ------------------------

    return combined;
}

std::vector<float> LoadModelSignature(const std::string &model_path, bool &loaded) {
    std::ifstream model(model_path, std::ios::binary);
    if (!model) {
        loaded = false;
        return {};
    }

    std::vector<unsigned char> buffer(4096);
    model.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
    std::streamsize read = model.gcount();
    if (read <= 0) {
        loaded = false;
        return {};
    }

    std::vector<float> signature(4, 0.0f);
    for (std::streamsize i = 0; i < read; ++i) {
        signature[i % signature.size()] += static_cast<float>(buffer[static_cast<size_t>(i)]) / 255.0f;
    }

    loaded = true;
    return signature;
}

std::vector<float> ComputeModelLogits(const std::vector<float> &fused,
                                      const std::vector<float> &signature) {
    if (signature.empty()) {
        return {};
    }

    std::vector<float> logits(signature.size(), 0.0f);
    for (size_t i = 0; i < fused.size(); ++i) {
        logits[i % logits.size()] += fused[i] * signature[i % signature.size()];
    }
    return logits;
}

std::vector<float> ComputeHeuristicLogits(const std::vector<float> &llm_vec,
                                          const std::vector<float> &query_vec) {
    float dot = lite::Dot(llm_vec, query_vec);
    float query_energy = lite::SquaredL2Norm(query_vec);
    float llm_energy = lite::SquaredL2Norm(llm_vec);

    // Produce four interpretable scores.
    std::vector<float> logits(4);
    logits[0] = dot;
    logits[1] = -dot;
    logits[2] = query_energy;
    logits[3] = llm_energy;
    return logits;
}

int64_t ArgMaxIndex(const std::vector<float> &values) {
    if (values.empty()) {
        return 0;
    }
    return static_cast<int64_t>(std::distance(values.begin(),
                                              std::max_element(values.begin(), values.end())));
}

std::string RouteSecurely(const std::vector<float> &llm_vec,
                          const std::vector<float> &query_vec,
                          const std::string &model_path = "mirt_bert.snapshot") {
    std::vector<float> fused = SecureForward(llm_vec, query_vec);

    bool model_loaded = false;
    std::vector<float> signature = LoadModelSignature(model_path, model_loaded);

    std::vector<float> logits;
    if (model_loaded) {
        logits = ComputeModelLogits(fused, signature);
    }

    if (logits.empty()) {
        logits = ComputeHeuristicLogits(llm_vec, query_vec);
    }

    int64_t route_id = ArgMaxIndex(logits);
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
        // they are decrypted and consumed by secure layers. This mirrors
        // BOLT's client/server trust model.
        std::vector<float> dec_query = DecryptVector(enc_query);
        std::vector<float> dec_llm = DecryptVector(enc_llm);

        std::string route = RouteSecurely(dec_llm, dec_query, model_path);
        std::cout << "Selected secure route: " << route << std::endl;
    } catch (const std::exception &ex) {
        std::cerr << "[secure-router] Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}


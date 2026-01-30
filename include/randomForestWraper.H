#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;
class RandomForestWrapper : public torch::nn::Module 
{
public:
    RandomForestWrapper(const std::string& json_file) {
        // Load the JSON file
        std::ifstream file(json_file);
        json j;
        file >> j;

        // Parse the JSON data
        for (const auto& tree : j) {
            trees.push_back(tree);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        // Implement the prediction logic
        std::cout<<x.size(1)<<std::endl;
        std::vector<float> predictions(x.size(1), 0.0);
        std::cout<<"loop"<<std::endl;
        for (const auto& tree : trees) {
            // Apply each tree to the input
            // This is a placeholder for the actual tree prediction logic
            tree["values"][0][0];
            std::cout<<"loop2"<<std::endl;
            
            for (int i = 0; i < x.size(1); ++i) {
                predictions[i] = predictions[i]+ tree["values"][0][0].get<float>(); // Simplified example
            }
        }
        return torch::tensor(predictions);
    }

private:
    std::vector<json> trees;
};
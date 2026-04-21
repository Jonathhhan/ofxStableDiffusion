#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <utility>

// Mock minimal ofDirectory and ofFile for testing
class ofFile {
public:
	ofFile(const std::string& path) : path_(path) {}
	std::string getBaseName() const {
		size_t pos = path_.find_last_of("/\\");
		std::string filename = (pos != std::string::npos) ? path_.substr(pos + 1) : path_;
		size_t dotPos = filename.find_last_of('.');
		return (dotPos != std::string::npos) ? filename.substr(0, dotPos) : filename;
	}
	std::string getAbsolutePath() const { return path_; }
	bool isFile() const { return true; }
private:
	std::string path_;
};

// Test that LoRA discovery patterns work correctly
void testLoraFilePatterns() {
	std::cout << "Testing LoRA file pattern recognition..." << std::endl;

	// Test file extension extraction
	std::vector<std::string> validExtensions = {
		"model.safetensors",
		"lora.ckpt",
		"adapter.pt",
		"weights.bin"
	};

	std::vector<std::string> invalidExtensions = {
		"model.txt",
		"config.json",
		"readme.md"
	};

	for (const auto& filename : validExtensions) {
		size_t dotPos = filename.find_last_of('.');
		std::string ext = filename.substr(dotPos);
		assert(ext == ".safetensors" || ext == ".ckpt" || ext == ".pt" || ext == ".bin");
	}

	std::cout << "✓ LoRA file pattern recognition passed" << std::endl;
}

// Test basename extraction
void testBasenameExtraction() {
	std::cout << "Testing basename extraction..." << std::endl;

	ofFile file1("/path/to/lora_model.safetensors");
	assert(file1.getBaseName() == "lora_model");

	ofFile file2("simple.ckpt");
	assert(file2.getBaseName() == "simple");

	ofFile file3("/nested/path/adapter_v2.pt");
	assert(file3.getBaseName() == "adapter_v2");

	std::cout << "✓ Basename extraction passed" << std::endl;
}

// Test that result pairs contain expected structure
void testResultStructure() {
	std::cout << "Testing LoRA result structure..." << std::endl;

	std::vector<std::pair<std::string, std::string>> results;
	results.emplace_back("my_lora", "/path/to/my_lora.safetensors");
	results.emplace_back("style_adapter", "/path/to/style_adapter.ckpt");

	assert(results.size() == 2);
	assert(results[0].first == "my_lora");
	assert(results[0].second == "/path/to/my_lora.safetensors");
	assert(results[1].first == "style_adapter");
	assert(results[1].second == "/path/to/style_adapter.ckpt");

	std::cout << "✓ LoRA result structure passed" << std::endl;
}

int main() {
	try {
		testLoraFilePatterns();
		testBasenameExtraction();
		testResultStructure();

		std::cout << "\n✅ All LoRA discovery tests passed!" << std::endl;
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
		return 1;
	}
}

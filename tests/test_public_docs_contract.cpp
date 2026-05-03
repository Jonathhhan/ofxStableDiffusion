#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string readFile(const std::filesystem::path& path) {
	std::ifstream input(path);
	if (!input.is_open()) {
		std::cerr << "Failed to open " << path << std::endl;
		std::exit(1);
	}
	return std::string(
		std::istreambuf_iterator<char>(input),
		std::istreambuf_iterator<char>());
}

void expectContains(
	const std::string& content,
	const std::string& needle,
	const std::filesystem::path& path) {
	if (content.find(needle) == std::string::npos) {
		std::cerr << "Expected to find '" << needle << "' in " << path << std::endl;
		std::exit(1);
	}
}

void expectNotContains(
	const std::string& content,
	const std::string& needle,
	const std::filesystem::path& path) {
	if (content.find(needle) != std::string::npos) {
		std::cerr << "Did not expect to find '" << needle << "' in " << path << std::endl;
		std::exit(1);
	}
}

} // namespace

int main() {
	const std::filesystem::path repoRoot =
		std::filesystem::path(__FILE__).parent_path().parent_path();

	const std::filesystem::path readmePath = repoRoot / "README.md";
	const std::filesystem::path apiRefPath = repoRoot / "docs" / "API_REFERENCE.md";
	const std::filesystem::path troubleshootingPath = repoRoot / "docs" / "TROUBLESHOOTING.md";
	const std::filesystem::path basicReadmePath =
		repoRoot / "examples" / "basic_generation" / "README.md";
	const std::filesystem::path basicAppPath =
		repoRoot / "examples" / "basic_generation" / "src" / "ofApp.cpp";
	const std::filesystem::path cancellationReadmePath =
		repoRoot / "examples" / "cancellation_example" / "README.md";
	const std::filesystem::path cancellationAppPath =
		repoRoot / "examples" / "cancellation_example" / "src" / "ofApp.cpp";

	const std::string readme = readFile(readmePath);
	const std::string apiReference = readFile(apiRefPath);
	const std::string troubleshooting = readFile(troubleshootingPath);
	const std::string basicReadme = readFile(basicReadmePath);
	const std::string basicApp = readFile(basicAppPath);
	const std::string cancellationReadme = readFile(cancellationReadmePath);
	const std::string cancellationApp = readFile(cancellationAppPath);

	expectContains(readme, "## Feature Readiness", readmePath);
	expectContains(readme, "## Threading Contract", readmePath);
	expectContains(readme, "Recommended starting point", readmePath);
	expectContains(apiReference, "`weightType` - Weight precision type", apiRefPath);
	expectContains(apiReference, "placeholder results", apiRefPath);
	expectContains(apiReference, "dispatch their callbacks from the thread that calls `update()`", apiRefPath);
	expectContains(troubleshooting, "**Safe from any thread:**", troubleshootingPath);
	expectContains(basicReadme, "worker thread", basicReadmePath);
	expectContains(cancellationReadme, "worker thread", cancellationReadmePath);
	expectContains(basicApp, "settings.weightType = SD_TYPE_F16", basicAppPath);
	expectContains(cancellationApp, "settings.weightType = SD_TYPE_F16", cancellationAppPath);

	expectNotContains(apiReference, "`wType` - Weight precision type", apiRefPath);
	expectNotContains(troubleshooting, "settings.wType", troubleshootingPath);
	expectNotContains(troubleshooting, "Not Thread-Safe (main thread only)", troubleshootingPath);
	expectNotContains(readme, "All public API methods should be called from the main thread", readmePath);
	expectNotContains(readme, "single-threaded use from the main thread", readmePath);
	expectNotContains(basicReadme, "settings.wType", basicReadmePath);
	expectNotContains(cancellationApp, "settings.wType", cancellationAppPath);

	const std::vector<std::filesystem::path> requiredExampleFiles = {
		repoRoot / "examples" / "basic_generation" / "Makefile",
		repoRoot / "examples" / "basic_generation" / "config.make",
		repoRoot / "examples" / "basic_generation" / "src" / "main.cpp",
		repoRoot / "examples" / "cancellation_example" / "Makefile",
		repoRoot / "examples" / "cancellation_example" / "config.make",
		repoRoot / "examples" / "cancellation_example" / "src" / "main.cpp"
	};
	for (const auto& path : requiredExampleFiles) {
		if (!std::filesystem::exists(path)) {
			std::cerr << "Missing expected example project file: " << path << std::endl;
			return 1;
		}
	}

	return 0;
}

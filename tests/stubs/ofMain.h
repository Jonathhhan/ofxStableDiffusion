#pragma once

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

enum ofImageType {
	OF_IMAGE_GRAYSCALE = 1,
	OF_IMAGE_COLOR = 3,
	OF_IMAGE_COLOR_ALPHA = 4
};

class ofPixels {
public:
	bool isAllocated() const {
		return !storage.empty();
	}

	std::size_t getWidth() const {
		return width;
	}

	std::size_t getHeight() const {
		return height;
	}

	std::size_t getNumChannels() const {
		return channels;
	}

	unsigned char* getData() {
		return storage.empty() ? nullptr : storage.data();
	}

	const unsigned char* getData() const {
		return storage.empty() ? nullptr : storage.data();
	}

	void setFromPixels(const unsigned char* data, int pixelWidth, int pixelHeight, ofImageType type) {
		width = pixelWidth > 0 ? static_cast<std::size_t>(pixelWidth) : 0;
		height = pixelHeight > 0 ? static_cast<std::size_t>(pixelHeight) : 0;
		channels = static_cast<std::size_t>(std::max(0, static_cast<int>(type)));
		const std::size_t byteCount = width * height * channels;
		storage.resize(byteCount);
		if (data && byteCount > 0) {
			std::copy(data, data + byteCount, storage.begin());
		}
	}

	void setFromPixels(const ofPixels& pixels) {
		storage = pixels.storage;
		width = pixels.width;
		height = pixels.height;
		channels = pixels.channels;
	}

	void clear() {
		storage.clear();
		width = 0;
		height = 0;
		channels = 0;
	}

private:
	std::size_t width = 0;
	std::size_t height = 0;
	std::size_t channels = 0;
	std::vector<unsigned char> storage;
};

class ofImage {
public:
	bool isAllocated() const {
		return pixels_.isAllocated();
	}

	void setFromPixels(const ofPixels& pixels) {
		pixels_.setFromPixels(pixels);
	}

	bool load(const std::string&) {
		return false;
	}

	bool save(const std::string& path) const {
		std::ofstream output(path, std::ios::binary);
		if (!output.is_open()) {
			return false;
		}
		output << "stub-image";
		return output.good();
	}

	void resize(int width, int height) {
		if (!pixels_.isAllocated()) {
			return;
		}
		ofPixels resized;
		const int channels = static_cast<int>(pixels_.getNumChannels());
		resized.setFromPixels(
			pixels_.getData(),
			width,
			height,
			channels == OF_IMAGE_COLOR_ALPHA ? OF_IMAGE_COLOR_ALPHA :
				(channels == OF_IMAGE_GRAYSCALE ? OF_IMAGE_GRAYSCALE : OF_IMAGE_COLOR));
		pixels_ = resized;
	}

	float getWidth() const {
		return static_cast<float>(pixels_.getWidth());
	}

	float getHeight() const {
		return static_cast<float>(pixels_.getHeight());
	}

	ofPixels& getPixels() {
		return pixels_;
	}

	const ofPixels& getPixels() const {
		return pixels_;
	}

	void draw(float, float, float, float) const {
	}

private:
	ofPixels pixels_;
};

class ofTexture {};
class ofFbo {};

class ofBaseApp {
public:
	virtual ~ofBaseApp() = default;
};

class ofThread {
public:
	virtual ~ofThread() = default;
	virtual void threadedFunction() {}

	bool isThreadRunning() const {
		return false;
	}

	void startThread() {}
	void stopThread() {}
	void waitForThread(bool = true) {}
	bool lock() { return true; }
	void unlock() {}
};

class ofLogStream {
public:
	template <typename T>
	ofLogStream& operator<<(const T&) {
		return *this;
	}
};

#ifndef OF_MAIN_STUB_CUSTOM_LOG_FUNCTIONS
inline ofLogStream ofLogNotice(const std::string& = "") { return {}; }
inline ofLogStream ofLogWarning(const std::string& = "") { return {}; }
inline ofLogStream ofLogError(const std::string& = "") { return {}; }
inline ofLogStream ofLogVerbose(const std::string& = "") { return {}; }
#endif

class ofJson {
public:
	ofJson() = default;
	ofJson(const char*) {}
	ofJson(const std::string&) {}
	ofJson(bool) {}
	ofJson(int) {}
	ofJson(unsigned int) {}
	ofJson(long long) {}
	ofJson(unsigned long long) {}
	ofJson(float) {}
	ofJson(double) {}
	ofJson(std::initializer_list<std::pair<const char*, ofJson>>) {}

	class Proxy {
	public:
		Proxy& operator[](const std::string&) {
			return *this;
		}

		template <typename T>
		Proxy& operator=(const T&) {
			return *this;
		}

		template <typename T>
		void push_back(const T&) {
		}
	};

	Proxy operator[](const std::string&) {
		return {};
	}

	static ofJson array() {
		return {};
	}

	void push_back(const ofJson&) {
	}

	template <typename T>
	void push_back(const T&) {
	}

	std::string dump(int = -1) const {
		return "{}";
	}

	static ofJson parse(const std::string&) {
		return {};
	}
};

inline bool ofSaveJson(const std::string& path, const ofJson& json) {
	std::ofstream output(path);
	if (!output.is_open()) {
		return false;
	}
	output << json.dump();
	return true;
}

inline bool ofSavePrettyJson(const std::string& path, const ofJson& json) {
	std::ofstream output(path);
	if (!output.is_open()) {
		return false;
	}
	output << json.dump(2);
	return true;
}

#ifndef OF_MAIN_STUB_CUSTOM_TIME_FUNCTIONS
inline std::uint64_t ofGetElapsedTimeMicros() {
	using namespace std::chrono;
	return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

inline std::uint64_t ofGetElapsedTimeMillis() {
	using namespace std::chrono;
	return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}
#endif

inline std::string ofTrim(const std::string& value) {
	const auto isSpaceChar = [](unsigned char c) { return std::isspace(c) != 0; };
	auto first = std::find_if_not(value.begin(), value.end(), isSpaceChar);
	auto last = std::find_if_not(value.rbegin(), value.rend(), isSpaceChar).base();
	if (first >= last) {
		return {};
	}
	return std::string(first, last);
}

inline std::string ofToUpper(std::string value) {
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
		return static_cast<char>(std::toupper(c));
	});
	return value;
}

inline std::string ofToDataPath(const std::string& value, bool = false) {
	return value;
}

template <typename T>
inline std::string ofToString(const T& value) {
	std::ostringstream stream;
	stream << value;
	return stream.str();
}

inline std::string ofToString(float value, int precision) {
	std::ostringstream stream;
	stream << std::fixed << std::setprecision(precision) << value;
	return stream.str();
}

inline std::string ofGetTimestampString() {
	return "19700101-000000";
}

inline int ofGetWidth() {
	return 1280;
}

inline int ofGetHeight() {
	return 720;
}

inline void ofBackground(int) {}
inline void ofSetColor(int, int = 255, int = 255, int = 255) {}
inline void ofDrawBitmapString(const std::string&, float, float) {}

template <typename T>
inline T ofClamp(T value, T minValue, T maxValue) {
	return std::max(minValue, std::min(value, maxValue));
}

class ofFile {
public:
	ofFile() = default;
	explicit ofFile(const std::string& path)
		: path_(path) {
	}

	std::string getExtension() const {
		return std::filesystem::path(path_).extension().string();
	}

	static bool doesFileExist(const std::string& path) {
		return std::filesystem::exists(path);
	}

private:
	std::string path_;
};

class ofDirectory {
public:
	ofDirectory() = default;
	explicit ofDirectory(const std::string& path)
		: path_(path) {
	}

	bool exists() const {
		return std::filesystem::exists(path_);
	}

private:
	std::string path_;
};

enum ofWindowModeType {
	OF_WINDOW = 0
};

struct ofGLFWWindowSettings {
	int width = 0;
	int height = 0;
	bool visible = true;
	ofWindowModeType windowMode = OF_WINDOW;

	void setSize(int w, int h) {
		width = w;
		height = h;
	}
};

using ofAppBaseWindow = int;
using ofAppBaseWindowPtr = std::shared_ptr<ofAppBaseWindow>;

inline ofAppBaseWindowPtr ofCreateWindow(const ofGLFWWindowSettings&) {
	return std::make_shared<ofAppBaseWindow>(0);
}

inline void ofRunApp(const ofAppBaseWindowPtr&, const std::shared_ptr<ofBaseApp>&) {}
inline void ofRunMainLoop() {}

#include "ofxStableDiffusionNativeVideoExport.h"

namespace ofxStableDiffusionNativeVideoExport {

bool isWebmExportAvailable() {
	return false;
}

bool saveWebm(const std::string& path, const ofxStableDiffusionVideoClip& clip, int quality) {
	ofLogWarning("ofxStableDiffusion")
		<< "WebM export is unavailable because the addon no longer compiles against "
		<< "vendored stable-diffusion.cpp source files. Stage a public export helper "
		<< "through libs/stable-diffusion/include and libs/stable-diffusion/lib to "
		<< "re-enable this path.";
	static_cast<void>(path);
	static_cast<void>(clip);
	static_cast<void>(quality);
	return false;
}

} // namespace ofxStableDiffusionNativeVideoExport

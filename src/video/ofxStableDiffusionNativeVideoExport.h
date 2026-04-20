#pragma once

#include "../core/ofxStableDiffusionTypes.h"

namespace ofxStableDiffusionNativeVideoExport {

bool isWebmExportAvailable();
bool saveWebm(const std::string& path, const ofxStableDiffusionVideoClip& clip, int quality = 90);

} // namespace ofxStableDiffusionNativeVideoExport

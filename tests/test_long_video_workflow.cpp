#include "video/ofxStableDiffusionLongVideoWorkflow.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

bool expect(bool condition, const std::string& message) {
	if (condition) {
		return true;
	}
	std::cerr << "FAIL: " << message << std::endl;
	return false;
}

} // namespace

int main() {
	bool ok = true;

	{
		ofxStableDiffusionLongVideoManifest manifest;
		const auto validation = ofxStableDiffusionLongVideoWorkflow::validate(manifest);
		ok &= expect(!validation.ok, "empty long video manifest is invalid");
	}

	{
		ofxStableDiffusionLongVideoManifest manifest;
		manifest.projectName = "Orbital Drift";
		manifest.conceptText = "A lonely orbital habitat above a storm planet";
		manifest.continuityBible = "Preserve station geometry, cloud color, and slow camera drift.";
		manifest.outputDirectory = "renders/orbital-drift";
		manifest.preset = ofxStableDiffusionLongVideoPreset::Balanced;
		manifest.chunks = {
			{
				"chunk-1",
				"Opening",
				"Orbiting station reveal, cinematic, slow parallax",
				"flicker, drift, broken geometry",
				"Establish the station and storm world",
				"Start with a stable reveal and end on a forward drift.",
				640,
				384,
				49,
				12,
				20,
				6.5f,
				0.65f,
				100,
				false,
				"chunk-1"
			},
			{
				"chunk-2",
				"Escalation",
				"Follow the station as lightning rises through the clouds",
				"flicker, identity drift, abrupt reset",
				"Escalate the weather and motion",
				"Reuse the previous last frame and keep camera direction coherent.",
				640,
				384,
				49,
				12,
				20,
				6.5f,
				0.65f,
				101,
				true,
				"chunk-2"
			}
		};

		const auto validation = ofxStableDiffusionLongVideoWorkflow::validate(manifest);
		ok &= expect(validation.ok, "configured long video manifest validates");

		const auto request =
			ofxStableDiffusionLongVideoWorkflow::buildChunkRequest(manifest, manifest.chunks.front());
		ok &= expect(request.prompt.find("Orbiting station reveal") != std::string::npos, "chunk request keeps prompt");
		ok &= expect(request.frameCount == 49, "chunk request keeps frame count");
		ok &= expect(request.seed == 100, "chunk request keeps seed");

		const std::string chunkDir =
			ofxStableDiffusionLongVideoWorkflow::buildChunkOutputDirectory(manifest, manifest.chunks[1]);
		ok &= expect(chunkDir.find("renders/orbital-drift") != std::string::npos, "chunk output directory includes root");
		ok &= expect(chunkDir.find("chunk-2") != std::string::npos, "chunk output directory includes chunk id");

		const std::string manifestJson =
			ofxStableDiffusionLongVideoWorkflow::buildManifestJson(manifest);
		ok &= expect(manifestJson.find("\"project_type\": \"long_video_render_manifest\"") != std::string::npos, "manifest json has project type");
		ok &= expect(manifestJson.find("\"project_name\": \"Orbital Drift\"") != std::string::npos, "manifest json has project name");
		ok &= expect(manifestJson.find("\"use_previous_last_frame\": true") != std::string::npos, "manifest json records continuity handoff");

		std::vector<ofxStableDiffusionLongVideoChunkResult> chunkResults = {
			{"chunk-1", true, "", "renders/orbital-drift/chunk-1", "renders/orbital-drift/chunk-1/metadata.json", "", 100, 49},
			{"chunk-2", true, "", "renders/orbital-drift/chunk-2", "renders/orbital-drift/chunk-2/metadata.json", "", 101, 49}
		};
		const std::string playlistJson =
			ofxStableDiffusionLongVideoWorkflow::buildPlaylistManifestJson(manifest, chunkResults);
		ok &= expect(playlistJson.find("\"chunk_id\": \"chunk-2\"") != std::string::npos, "playlist json includes second chunk");
		ok &= expect(playlistJson.find("\"actual_seed\": 101") != std::string::npos, "playlist json includes seed");
	}

	return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

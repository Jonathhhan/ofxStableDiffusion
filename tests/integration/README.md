# Integration Tests

Integration tests verify the addon works correctly with the actual stable-diffusion.cpp library. Unlike unit tests, these require:

- A working stable-diffusion.cpp library installation
- Model files (SD 1.5, SDXL, etc.)
- Sufficient system resources (VRAM/RAM)

## Running Integration Tests

### Prerequisites

1. **Build stable-diffusion.cpp:**
   ```bash
   # From addon root
   ./scripts/build-stable-diffusion.sh
   ```

2. **Download test models:**
   ```bash
   mkdir -p tests/integration/models
   # Download a small model for testing (e.g., SD 1.5 or SD-Turbo)
   # Place in tests/integration/models/
   ```

### Build and Run

```bash
# Configure
cmake -S tests/integration -B tests/integration/build -DMODEL_PATH=/path/to/model.safetensors

# Build
cmake --build tests/integration/build --config Release

# Run all integration tests
ctest --test-dir tests/integration/build -C Release --output-on-failure

# Run specific test
tests/integration/build/test_basic_generation
```

## Test Categories

### Basic Generation Tests
- **test_basic_generation** - Text-to-image generation with default settings
- **test_cancellation** - Cancellation during long generation

### Advanced Tests
- **test_img2img** - Image-to-image transformation
- **test_video_generation** - Video frame generation
- **test_upscaling** - ESRGAN upscaling

### Stress Tests
- **test_memory_limits** - Memory handling under load
- **test_batch_generation** - Multiple sequential generations
- **test_model_switching** - Switch between models

## Environment Variables

- `SD_MODEL_PATH` - Path to SD model file (required)
- `SD_VAE_PATH` - Path to VAE file (optional)
- `SD_ESRGAN_PATH` - Path to ESRGAN upscaler (optional)
- `INTEGRATION_TEST_TIMEOUT` - Test timeout in seconds (default: 300)
- `INTEGRATION_TEST_SKIP` - Skip integration tests if set

## CI/CD

Integration tests are optional in CI due to resource requirements. They run when:
- Manually triggered via workflow dispatch
- On release branches
- With `[test:integration]` in commit message

## Writing Integration Tests

Integration tests should:

1. **Check for required resources:**
   ```cpp
   if (!hasModel()) {
       std::cout << "SKIPPED: Model not available" << std::endl;
       return 0;
   }
   ```

2. **Use reasonable timeouts:**
   ```cpp
   // Use fast settings for CI
   request.sampleSteps = 5;  // Not 50
   request.width = 512;      // Not 1024
   ```

3. **Clean up resources:**
   ```cpp
   // Ensure cleanup even on failure
   sd.clearModelCache();
   ```

4. **Verify results:**
   ```cpp
   assert(images.size() == 1);
   assert(images[0].pixels.getWidth() == 512);
   assert(images[0].pixels.getHeight() == 512);
   ```

## Performance Benchmarks

Integration tests can also measure performance:

```bash
# Run with profiling enabled
cmake -S tests/integration -B tests/integration/build -DENABLE_PROFILING=ON
tests/integration/build/test_basic_generation --benchmark
```

Results are saved to `tests/integration/benchmarks/`.

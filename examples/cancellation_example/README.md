# Cancellation Example

Demonstrates how to cancel long-running generation operations gracefully.

## Features

- Start a long-running generation (50 steps, 768x768)
- Cancel during generation
- Thread-safe cancellation from any thread
- Check cancellation status
- Handle cancelled operations

## Usage

1. Run the example
2. Press **SPACE** to start a long generation
3. Press **C** during generation to cancel
4. Observe graceful cancellation after current step

## Cancellation API

### Request Cancellation

```cpp
if (sd.isGenerating()) {
    bool requested = sd.requestCancellation();
    // Returns true if cancellation was requested
}
```

This is **thread-safe** - you can call it from UI threads, button callbacks, etc.

### Check Cancellation Status

```cpp
// Check if cancellation is pending
if (sd.isCancellationRequested()) {
    // Cancellation requested, waiting for current step
}

// Check if last operation was cancelled
if (sd.wasCancelled()) {
    // Previous generation was cancelled
}
```

### Handle Cancellation

```cpp
if (!sd.isGenerating()) {
    if (sd.wasCancelled()) {
        ofLogNotice() << "User cancelled";
    } else {
        auto error = sd.getLastErrorInfo();
        if (error.code == ofxStableDiffusionErrorCode::Cancelled) {
            ofLogNotice() << "Cancelled: " << error.message;
        }
    }
}
```

## Timing Expectations

- Cancellation happens **after the current step completes**
- Each step may take 1-5 seconds depending on:
  - Image dimensions
  - Model complexity
  - Hardware speed
- Typical cancellation delay: 1-5 seconds

## Thread Safety

These methods are thread-safe:

```cpp
sd.isGenerating()              // Safe
sd.requestCancellation()       // Safe
sd.isCancellationRequested()   // Safe
sd.wasCancelled()              // Safe
```

## Use Cases

- **Long operations**: Cancel 50+ step generations
- **User interaction**: Cancel button in UI
- **Timeouts**: Cancel if taking too long
- **Queue management**: Cancel queued items
- **Resource cleanup**: Cancel before app shutdown

## Example: UI Cancel Button

```cpp
void onCancelButtonPressed() {
    if (sd.isGenerating()) {
        sd.requestCancellation();
        showSpinner("Cancelling...");
    }
}

void update() {
    if (!sd.isGenerating()) {
        hideSpinner();
        if (sd.wasCancelled()) {
            showNotification("Operation cancelled");
        }
    }
}
```

## See Also

- [API Reference](../../docs/API_REFERENCE.md#cancellation) - Cancellation API details
- [Migration Guide](../../docs/MIGRATION_GUIDE.md#using-new-features) - v1.3.0 cancellation feature

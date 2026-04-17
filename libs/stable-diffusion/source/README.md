# Vendored Source

Place a known-compatible `stable-diffusion.cpp` source checkout in this folder.

Expected contents:

- `CMakeLists.txt`
- the native source files for the library
- `stable-diffusion.h` somewhere in the tree

Once vendored, rebuild the addon-native library with:

```powershell
scripts\build-stable-diffusion.ps1
```

The build script stages the resulting `stable-diffusion.dll`, `stable-diffusion.lib`,
and `stable-diffusion.h` back into the addon's `libs/stable-diffusion/` layout.

This addon intentionally keeps the native diffusion stack standalone instead of
sharing the `ggml` build from `ofxGgml`, so each addon can pin and evolve its
native dependencies independently.

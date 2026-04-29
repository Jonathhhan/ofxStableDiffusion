# Test Stubs

This directory holds lightweight native-symbol stubs for helper-only tests.

Do not add broad stable-diffusion.cpp replacements here. Only stub symbols that
are needed by value-type constructors or helper code used by unit tests that do
not create a native Stable Diffusion context. Keep stub defaults aligned with
the upstream native initializer, and wire each stub only into the specific test
targets that need it.

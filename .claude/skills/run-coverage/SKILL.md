---
name: run-coverage
description: Runs code coverage checker on the chosen device backend. Use when the user asks to run the coverage, or to check the code coverage.
---

# Checking the code coverage

Run every instrumented test suite in the workspace against a single chosen device backend, then make sure everything is within required limits.

Makefile is the source of truth for the gated values, and the code coverage setup.

## Step 1: choose the device

`TEST_DEVICE` names the backend feature to compile with, and holds **only** the backend
name: `cuda`, `metal`, `vulkan` or `rocm`. Leave it unset for CPU, since there is no
`cpu` feature.

Ask which device to use when the conversation has not already established one.

## Step 2: run the suites

Pass the same device to every target, so llama.cpp is compiled once and reused across
all suites instead of being rebuilt for a different feature set. Run exactly:

```bash
make coverage TEST_DEVICE=cuda
```

For CPU, omit the assignment entirely:

```bash
make coverage
```

## Step 4: report

After all suites finish, sum up the results in an actionable report. Make sure all code coverage gates are met.



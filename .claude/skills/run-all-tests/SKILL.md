---
name: run-all-tests
description: Runs every test suite in the workspace on the chosen device backend. Use when the user asks to run the tests, run all the tests, run the full test suite, or check that everything still passes.
---

# Running all tests

Run every test suite in the workspace against a single chosen device backend.

## Step 1: choose the device

`TEST_DEVICE` names the backend feature to compile with, and holds **only** the backend
name: `cuda`, `metal`, `vulkan` or `rocm`. Leave it unset for CPU, since there is no
`cpu` feature.

Ask which device to use when the conversation has not already established one.

## Step 2: run the suites

Pass the same device to every target, so llama.cpp is compiled once and reused across
all suites instead of being rebuilt for a different feature set. Run exactly:

```bash
make test.llms TEST_DEVICE=cuda
```

For CPU, omit the assignment entirely:

```bash
make test.llms
```

## Step 3: rules during the run

- **Per-test 30 s budget.** Flag any individual test that exceeds 30 s wall-clock. That is a real bug — production or test — not flakiness.

## Step 4: report

After all suites finish, sum up the results in an actionable report.


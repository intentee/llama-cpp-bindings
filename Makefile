.DELETE_ON_ERROR:

TEST_DEVICE ?=

DEVICE_FEATURE = $(if $(TEST_DEVICE),--features $(TEST_DEVICE),)

CARGO_HOME ?= $(HOME)/.cargo

CLANG_TIDY ?= clang-tidy

COMPILE_COMMANDS = target/compile_commands.json

CPPCHECK = lint-tools/cppcheck-src/cppcheck

CPPCHECK_COMMIT = f726f98ed2c3277780ee133ec5338f6352d7b43b

CPPCHECK_SOURCE = https://github.com/danmar/cppcheck.git

HEISENBERG_CONFIGURATION = x86_64-unknown-linux-gnu

RUST_SYSROOT = $(shell rustc --print sysroot)

WRAPPER_SOURCES_RESPONSE_FILE = target/wrapper_sources.rsp

WRAPPER_SOURCES_CRATE = llama-cpp-wrapper-sources

WRAPPER_SOURCES_CRATE_FILES = \
	$(WRAPPER_SOURCES_CRATE)/Cargo.toml \
	$(wildcard $(WRAPPER_SOURCES_CRATE)/src/*.rs)

EMIT_WRAPPER_BUILD_INPUTS = cargo run --quiet --package $(WRAPPER_SOURCES_CRATE) -- \
	$(CURDIR)/llama-cpp-bindings-sys $(COMPILE_COMMANDS) $(WRAPPER_SOURCES_RESPONSE_FILE)

LLAMA_CPP_AND_GSL_SUPPRESSIONS = \
	--suppress='*:*llama-cpp-bindings-sys/llama.cpp/*' \
	--suppress='*:*llama-cpp-bindings-sys/GSL/*'

lint-tools/clang-tidy-venv/bin/clang-tidy: requirements-lint.txt
	python3 -m venv lint-tools/clang-tidy-venv
	lint-tools/clang-tidy-venv/bin/pip install --quiet --requirement requirements-lint.txt

lint-tools/cppcheck-src/Makefile:
	git init --quiet lint-tools/cppcheck-src
	git -C lint-tools/cppcheck-src fetch --quiet --depth 1 $(CPPCHECK_SOURCE) $(CPPCHECK_COMMIT)
	git -C lint-tools/cppcheck-src checkout --quiet FETCH_HEAD

$(CPPCHECK): lint-tools/cppcheck-src/Makefile
	$(MAKE) --directory=lint-tools/cppcheck-src cppcheck \
		FILESDIR=$(CURDIR)/lint-tools/cppcheck-src

$(COMPILE_COMMANDS): $(WRAPPER_SOURCES_CRATE_FILES)
	$(EMIT_WRAPPER_BUILD_INPUTS)

$(WRAPPER_SOURCES_RESPONSE_FILE): $(WRAPPER_SOURCES_CRATE_FILES)
	$(EMIT_WRAPPER_BUILD_INPUTS)

.PHONY: clean.cmake
clean.cmake:
	cargo clean --package llama-cpp-bindings-sys

.PHONY: clippy
clippy:
	cargo clippy --workspace --all-targets $(DEVICE_FEATURE) -- -D warnings

.PHONY: coverage
coverage:
	cargo llvm-cov clean --workspace
	cargo llvm-cov --no-report --no-fail-fast --workspace $(DEVICE_FEATURE)
	cargo llvm-cov report --json --output-path target/llvm-cov.json
	cargo llvm-cov report --lcov --output-path target/lcov.info
	cargo llvm-cov report
	heisenberg coverage-check \
		--workspace $(CURDIR) \
		--cargo-home $(CARGO_HOME) \
		--sysroot $(RUST_SYSROOT) \
		--coverage-export $(CURDIR)/target/llvm-cov.json \
		--configuration $(HEISENBERG_CONFIGURATION)

.PHONY: coverage-clean
coverage-clean:
	cargo llvm-cov clean --workspace
	rm -rf target/llvm-cov-target
	rm -f target/llvm-cov.json target/lcov.info

.PHONY: coverage-report
coverage-report:
	cargo llvm-cov report --html

.PHONY: fmt
fmt:
	cargo fmt --all

.PHONY: fmt.check
fmt.check:
	cargo fmt --all --check

.PHONY: lint.cpp
lint.cpp: lint.cpp.clang-tidy lint.cpp.cppcheck

.PHONY: lint.cpp.clang-tidy
lint.cpp.clang-tidy: $(COMPILE_COMMANDS) $(WRAPPER_SOURCES_RESPONSE_FILE)
	$(CLANG_TIDY) -p $(dir $(COMPILE_COMMANDS)) @$(WRAPPER_SOURCES_RESPONSE_FILE)

.PHONY: lint.cpp.cppcheck
lint.cpp.cppcheck: $(COMPILE_COMMANDS) $(CPPCHECK)
	$(CPPCHECK) --project=$(COMPILE_COMMANDS) --enable=all --inconclusive \
		--check-level=exhaustive --error-exitcode=1 \
		$(LLAMA_CPP_AND_GSL_SUPPRESSIONS) \
		--suppress=missingIncludeSystem

.PHONY: test
test: test.llms

.PHONY: test.harness
test.harness: clippy
	cargo test -p llama-cpp-test-harness-macros -p llama-cpp-test-harness \
		-p llama-cpp-test-model-provisioning $(DEVICE_FEATURE)

.PHONY: test.llms
test.llms: clippy test.harness test.unit
	cargo test --no-fail-fast -p llama-cpp-bindings-tests $(DEVICE_FEATURE)

.PHONY: test.unit
test.unit: clippy
	cargo test -p llama-cpp-backend -p llama-cpp-backend-tests -p llama-cpp-bindings \
		-p llama-cpp-bindings-build -p llama-cpp-bindings-sys -p llama-cpp-bindings-types -p llama-cpp-capturing-logger \
		-p llama-cpp-chat -p llama-cpp-context -p llama-cpp-context-memory -p llama-cpp-context-state \
		-p llama-cpp-error-chain -p llama-cpp-error-recorder -p llama-cpp-gbnf -p llama-cpp-gguf \
		-p llama-cpp-log-decoder -p llama-cpp-model -p llama-cpp-mtmd -p llama-cpp-params \
		-p llama-cpp-path-c-string -p llama-cpp-sampler -p llama-cpp-token-usage \
		-p llama-cpp-wrapper-sources \
		$(DEVICE_FEATURE)

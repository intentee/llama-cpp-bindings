.DELETE_ON_ERROR:

TEST_DEVICE ?=

DEVICE_FEATURE = $(if $(TEST_DEVICE),--features $(TEST_DEVICE),)

CLANG_TIDY ?= clang-tidy

COMPILE_COMMANDS = target/compile_commands.json

CPPCHECK = target/cppcheck-src/cppcheck

CPPCHECK_COMMIT = f726f98ed2c3277780ee133ec5338f6352d7b43b

CPPCHECK_SOURCE = https://github.com/danmar/cppcheck.git

WRAPPER_SOURCES_RESPONSE_FILE = target/wrapper_sources.rsp

WRAPPER_SOURCES_CRATE = llama-cpp-wrapper-sources

WRAPPER_SOURCES_CRATE_FILES = \
	$(WRAPPER_SOURCES_CRATE)/src/compile_commands_file.rs \
	$(WRAPPER_SOURCES_CRATE)/src/cpp_standard.rs \
	$(WRAPPER_SOURCES_CRATE)/src/wrapper_include_dirs.rs \
	$(WRAPPER_SOURCES_CRATE)/src/wrapper_source_paths.rs \
	$(WRAPPER_SOURCES_CRATE)/src/wrapper_sources.rs \
	$(WRAPPER_SOURCES_CRATE)/src/wrapper_sources_response_file.rs

EMIT_WRAPPER_BUILD_INPUTS = cargo run --quiet --package $(WRAPPER_SOURCES_CRATE) -- \
	$(CURDIR)/llama-cpp-bindings-sys $(COMPILE_COMMANDS) $(WRAPPER_SOURCES_RESPONSE_FILE)

VENDORED_SUPPRESSIONS = \
	--suppress='*:*llama-cpp-bindings-sys/llama.cpp/*' \
	--suppress='*:*llama-cpp-bindings-sys/GSL/*'

node_modules: package-lock.json
	npm ci
	touch node_modules

package-lock.json: package.json
	npm install --package-lock-only

$(COMPILE_COMMANDS): $(WRAPPER_SOURCES_CRATE_FILES)
	$(EMIT_WRAPPER_BUILD_INPUTS)

target/cppcheck-src/Makefile:
	git init --quiet target/cppcheck-src
	git -C target/cppcheck-src fetch --quiet --depth 1 $(CPPCHECK_SOURCE) $(CPPCHECK_COMMIT)
	git -C target/cppcheck-src checkout --quiet FETCH_HEAD

$(CPPCHECK): target/cppcheck-src/Makefile
	$(MAKE) --directory=target/cppcheck-src cppcheck \
		FILESDIR=$(CURDIR)/target/cppcheck-src

target/lint-venv/bin/clang-tidy: requirements-lint.txt
	python3 -m venv target/lint-venv
	target/lint-venv/bin/pip install --quiet --requirement requirements-lint.txt

$(WRAPPER_SOURCES_RESPONSE_FILE): $(WRAPPER_SOURCES_CRATE_FILES)
	$(EMIT_WRAPPER_BUILD_INPUTS)

.PHONY: clean.cmake
clean.cmake:
	cargo clean --package llama-cpp-bindings-sys

.PHONY: clippy
clippy:
	cargo clippy --workspace --all-targets $(DEVICE_FEATURE) -- -D warnings

.PHONY: coverage
coverage: node_modules
	cargo llvm-cov clean --workspace
	cargo llvm-cov --no-report --no-fail-fast --workspace $(DEVICE_FEATURE)
	cargo llvm-cov report --json --output-path target/llvm-cov.json
	cargo llvm-cov report --lcov --output-path target/lcov.info
	cargo llvm-cov report
	./node_modules/.bin/rust-coverage-check target/llvm-cov.json \
		--workspace-root $(CURDIR) \
		--gated llama-cpp-bindings=98 \
		--gated llama-cpp-bindings-types=100 \
		--gated llama-cpp-error-recorder=100 \
		--gated llama-cpp-ffi-status=100 \
		--gated llama-cpp-gbnf=100 \
		--gated llama-cpp-log-decoder=100

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
		$(VENDORED_SUPPRESSIONS) \
		--suppress=missingIncludeSystem

.PHONY: test
test: test.llms

.PHONY: test.harness
test.harness: clippy
	cargo test -p llama-cpp-test-harness-macros -p llama-cpp-test-harness $(DEVICE_FEATURE)

.PHONY: test.llms
test.llms: clippy test.harness test.unit
	cargo test --no-fail-fast -p llama-cpp-bindings-tests $(DEVICE_FEATURE)

.PHONY: test.unit
test.unit: clippy
	cargo test -p llama-cpp-bindings -p llama-cpp-bindings-build -p llama-cpp-gbnf \
		-p llama-cpp-log-decoder -p llama-cpp-wrapper-sources $(DEVICE_FEATURE)

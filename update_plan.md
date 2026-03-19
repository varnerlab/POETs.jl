# POETs.jl Modernization Plan

## Current State Assessment

**Good News:**
- Package still loads and runs in Julia 1.12.5
- Core algorithm implementation is sound (2 exported functions: `estimate_ensemble`, `rank_function`)
- Real academic value with published papers
- MIT licensed (in source headers)

**Critical Issues Found:**
- **Tests are broken** - Type error (`rank_cutoff=4` expects Float64, not Int64)
- **Extremely minimal Project.toml** - No dependencies, compat bounds, or metadata
- **Outdated documentation** - Uses Julia 0.x syntax (`Pkg.clone()`)
- **No modern package infrastructure** - Missing CI/CD, proper tests, docs
- **Last significant update ~2018**

---

## Phase 1: Critical Fixes (Week 1-2)

### 1.1 Fix Project.toml Structure
- Add `[compat]` section with Julia version bounds (>=1.6)
- Add `[deps]` section if any dependencies are identified
- Include proper metadata: description, homepage, repository
- Add test dependencies under `[extras]` and `[targets]`

### 1.2 Fix Type System Issues
- Update test files to use proper types (`4.0` instead of `4` for Float64 parameters)
- Review all function signatures for overly strict type annotations
- Fix the `rank_cutoff` parameter type issue in test calls

### 1.3 Create Proper Test Suite
- Restructure `test/runtests.jl` to use `Test` stdlib properly
- Add `@test` macros and proper test organization
- Ensure tests run successfully with `Pkg.test()`

---

## Phase 2: Documentation & Usability (Week 3-4)

### 2.1 Modernize README
- Replace deprecated `Pkg.clone()` with modern `Pkg.add(url=...)` syntax
- Add clear usage examples with modern Julia syntax
- Include performance characteristics and algorithm overview
- Add badges for CI status, version, license

### 2.2 Add Formal Documentation
- Create `docs/` directory with Documenter.jl setup
- Document the core algorithms with mathematical background
- Provide comprehensive API documentation
- Include the biochemical sample as a tutorial

### 2.3 Add License File
- Extract MIT license from source headers into proper `LICENSE` file
- Ensure license is recognized by GitHub and package registries

---

## Phase 3: Modern Infrastructure (Week 5-6)

### 3.1 Continuous Integration
- Add GitHub Actions workflow for testing across Julia versions
- Include testing on multiple platforms (Linux, macOS, Windows)
- Set up code coverage reporting with Codecov

### 3.2 Package Registration Preparation
- Ensure compatibility with General Registry standards
- Add appropriate keywords and categories in Project.toml
- Prepare for Julia package registry submission

### 3.3 Code Quality Improvements
- Add type annotations where beneficial for performance
- Review algorithm implementation for modern Julia best practices
- Consider using `Random.jl` for reproducible random number generation

---

## Phase 4: Enhancement & Validation (Week 7-8)

### 4.1 Sample Code Modernization
- Review biochemical sample for compatibility with modern Julia
- Update any deprecated syntax in sample code
- Ensure sample runs successfully and provides expected output

### 4.2 Performance Optimization
- Profile the core algorithm for potential bottlenecks
- Consider using `StaticArrays.jl` for small fixed-size arrays
- Review memory allocation patterns

### 4.3 Extended Testing
- Add property-based tests for algorithm correctness
- Include benchmark tests for performance regression detection
- Add integration tests with sample problems

---

## Key Files for Implementation

| File | Action |
|------|--------|
| `Project.toml` | Add dependencies, compat bounds, and metadata |
| `test/test_binh_korn.jl` | Fix type errors and modernize test structure |
| `test/runtests.jl` | Restructure to use proper Test framework |
| `README.md` | Update installation instructions and add modern examples |
| `src/Main.jl` | Review for type annotation improvements |
| `LICENSE` | Create from embedded source headers |

## Risk Assessment

- **Low Risk**: Core algorithm is mathematically sound and well-implemented
- **Medium Risk**: Academic code may have undocumented assumptions
- **Mitigation**: Maintain backward compatibility, test against known benchmarks, document breaking changes

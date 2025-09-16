# Pydala Codebase Refactoring Plan

## Overview
This document outlines a comprehensive refactoring plan for the Pydala codebase, focusing on improving code quality, reducing complexity, and enhancing maintainability while ensuring full backward compatibility.

## Analysis Summary

The codebase analysis revealed several areas for improvement:
- **Code Quality**: Large blocks of commented code, bare exception clauses
- **Complexity**: Methods with high cyclomatic complexity, duplicate code
- **Type Safety**: Missing type hints, insufficient input validation
- **Maintainability**: Magic numbers, scattered configuration

## Priority 1: Code Cleanup and Bug Fixes

### 1. Remove Commented Code in misc.py
- **Location**: `pydala/helpers/misc.py:15-370`
- **Issue**: ~355 lines of commented code (90% of file)
- **Action**: Remove all commented code blocks
- **Impact**:
  - Reduces file size from ~400 lines to ~45 lines
  - Improves code readability and navigation
  - Eliminates confusion about deprecated functionality

### 2. Fix Critical Bug in catalog.py
- **Location**: `pydala/catalog.py:63-66`
- **Current Code**:
  ```python
  # Update the catalog with itself?
  catalog_dict.update(self.to_dict())  # This line is suspicious
  ```
- **Issue**: Redundant self-update potentially causing data corruption
- **Fix**: Remove or properly document this update logic

### 3. Fix Bare Exception Clause
- **Location**: `pydala/io.py:49`
- **Current Code**:
  ```python
  except Exception:
      pass
  ```
- **Issue**: Masks all errors without logging
- **Fix**:
  ```python
  except (OSError, IOError) as e:
      logger.warning(f"Failed to process {file_path}: {e}")
  ```

## Priority 2: Reduce Complexity

### 4. Simplify _get_sort_by Method
- **Location**: `pydala/table.py:30-93`
- **Issue**: 63-line method with high cyclomatic complexity
- **Action**: Extract helper functions:
  - `_parse_string_sort(value: str) -> SortKey`
  - `_parse_callable_sort(value: Callable) -> SortKey`
  - `_validate_sort_key(key: SortKey) -> None`
- **Benefits**:
  - Improves testability
  - Reduces cognitive complexity
  - Enables better error messages

### 5. Remove Duplicate Scanner Method
- **Location**: `pydala/table.py`
- **Issue**: `to_scanner()` and `scanner()` are identical (lines 748-755, 757-764)
- **Action**:
  - Keep `scanner()` as primary method
  - Mark `to_scanner()` as deprecated with warning
  - Update documentation

### 6. Extract Scanner Parameters
- **Issue**: Scanner configuration duplicated across 10+ methods
- **Action**: Create `ScannerConfig` dataclass:
  ```python
  @dataclass
  class ScannerConfig:
      batch_size: int = 131072
      buffer_size: int = 65536
      prefetch: int = 2
      num_threads: int = 4
  ```
- **Files to Update**: `table.py`, `dataset.py`, `scanner.py`
- **Benefits**:
  - Centralized configuration
  - Easier parameter management
  - Consistent defaults across the codebase

## Priority 3: Improve Type Safety

### 7. Add Type Hints
- **Focus Areas**:
  - All public methods in core modules
  - Complex methods in `table.py`, `dataset.py`
  - Helper functions in `misc.py`
- **Example**:
  ```python
  def head(
      self,
      n: int = 5,
      columns: Optional[List[str]] = None
  ) -> "Table":
      ...
  ```

### 8. Add Input Validation
- **Approach**:
  - Add parameter validation at public API boundaries
  - Use clear, descriptive error messages
  - Validate types, ranges, and constraints
- **Example**:
  ```python
  if not isinstance(n, int) or n < 0:
      raise ValueError("n must be a non-negative integer")
  ```

## Priority 4: Maintainability Improvements

### 9. Create Constants File
- **New File**: `pydala/constants.py`
- **Content**:
  ```python
  # Performance tuning
  DEFAULT_BATCH_SIZE = 131072
  DEFAULT_BUFFER_SIZE = 65536
  DEFAULT_PREFETCH_COUNT = 2
  DEFAULT_THREAD_COUNT = 4

  # Validation
  MAX_COLUMN_NAME_LENGTH = 255
  MIN_PARTITION_SIZE = 1024
  ```
- **Benefits**:
  - Single source of truth
  - Easier tuning
  - Better documentation

### 10. Fix Test Infrastructure
- **Location**: `tests/test_table.py`
- **Issues**:
  - Missing imports
  - Incomplete test coverage
  - Outdated test cases
- **Action**:
  - Fix import statements
  - Add tests for refactored methods
  - Ensure 90%+ code coverage

## Implementation Strategy

### Phase 1: Code Cleanup (Days 1-2)
1. Remove commented code
2. Fix critical bugs
3. Update exception handling

### Phase 2: Complexity Reduction (Days 3-5)
1. Extract helper methods
2. Remove duplicate code
3. Create configuration classes

### Phase 3: Type Safety (Days 6-7)
1. Add type hints
2. Add input validation
3. Update documentation

### Phase 4: Maintainability (Days 8-9)
1. Create constants file
2. Update tests
3. Performance optimization

### Phase 5: Testing and Documentation (Day 10)
1. Run full test suite
2. Update API documentation
3. Create migration guide

## Backward Compatibility Guarantees

1. **API Stability**: No breaking changes to public APIs
2. **Method Signatures**: All existing parameters remain supported
3. **Return Types**: No changes to return types
4. **Error Handling**: Same exceptions thrown for same conditions
5. **Deprecation Path**: Deprecated methods will warn but continue to work

## Risk Assessment

### Low Risk
- Removing commented code
- Adding type hints
- Creating constants file

### Medium Risk
- Extracting helper methods
- Adding input validation
- Fixing exception handling

### High Risk
- Bug fixes in core logic
- Removing duplicate methods
- Performance optimizations

## Success Metrics

1. **Code Quality**:
   - Reduce cyclomatic complexity by 40%
   - Eliminate all commented code
   - Fix all linting issues

2. **Maintainability**:
   - Achieve 90%+ test coverage
   - Add type hints to 100% of public methods
   - Reduce code duplication by 30%

3. **Performance**:
   - No performance regression
   - 10% improvement in memory usage for large datasets

## Rollback Plan

1. Git tags will be created before each major change
2. Each commit will be atomic and revertable
3. Continuous integration will catch regressions early
4. Feature flags for performance optimizations

## Next Steps

1. Create feature branch for refactoring
2. Set up continuous integration
3. Begin with Phase 1 (Code Cleanup)
4. Regular progress updates to stakeholders
5. Code reviews for each change

This plan provides a structured approach to improving the Pydala codebase while ensuring stability and backward compatibility throughout the process.
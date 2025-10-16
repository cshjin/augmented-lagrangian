# Changelog

All notable changes to the augmented-lagrangian package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of the augmented-lagrangian package

## [0.1.1] - 2025-10-16

### Changed
- Renamed `mu_increase_factor` parameter to `rho` for clarity in penalty parameter.
- Flexible handling of constraint functions returning multiple constraints.
- Added `max_inner_iterations` and `backend` parameters to `solve` method for better control over optimization process.

### Added
- Support for optional PyTorch backend for optimization.
- Example scripts demonstrating usage with both SciPy and PyTorch backends.
- Additional tests for new features and backends.


## [0.1.0] - 2025-10-15

### Added
- `AugmentedLagrangian` class for constrained optimization
- Support for single and multiple equality constraints
- Flexible API for defining objective and constraint functions
- Comprehensive test suite with pytest
- Example scripts demonstrating usage
- Complete documentation and README
- MIT license
- PyPI package configuration with pyproject.toml

### Features
- Augmented Lagrangian method implementation
- Automatic Lagrange multiplier updates
- Penalty parameter adaptation
- Convergence monitoring and history tracking
- Integration with SciPy's L-BFGS-B optimizer
- Verbose output option for debugging
- Configurable algorithm parameters

### Documentation
- Comprehensive README with examples
- API reference documentation
- Mathematical background explanation
- Installation and usage instructions
- Contributing guidelines
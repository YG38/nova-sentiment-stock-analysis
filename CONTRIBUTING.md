# Contributing to Stock Sentiment Analysis

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing to the Stock Sentiment Analysis project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/your-username/stock-sentiment-analysis.git
   cd stock-sentiment-analysis
   ```
3. **Set up** the development environment (see main README for details).
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. **Create/update** your feature or bugfix.
2. **Test** your changes (see [Testing](#testing)).
3. **Commit** your changes with a descriptive message:
   ```bash
   git commit -m "Add feature: your feature description"
   ```
4. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Create a Pull Request** from your fork to the main repository.

## Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code.
- Use type hints for better code clarity.
- Document all public functions and classes with docstrings.
- Keep lines under 88 characters (Black's default).

## Testing

1. **Run tests** before submitting changes:
   ```bash
   pytest tests/
   ```
2. **Check code coverage**:
   ```bash
   pytest --cov=src tests/
   ```
3. **Lint** your code:
   ```bash
   flake8 src/
   black --check src/
   isort --check-only src/
   mypy src/
   ```

## Pull Request Process

1. Ensure your code passes all tests and linting checks.
2. Update the documentation if necessary.
3. Ensure your branch is up to date with the main branch.
4. Submit your pull request with a clear description of the changes.
5. Address any review comments.

## Reporting Issues

When reporting issues, please include:

1. A clear, descriptive title.
2. Steps to reproduce the issue.
3. Expected vs. actual behavior.
4. Any relevant error messages or logs.
5. Your environment (OS, Python version, etc.).

---

Thank you for your contribution!

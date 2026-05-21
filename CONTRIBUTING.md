# Contributing to Orchestra

## Running Tests

```bash
pytest tests/
```

## Running Tests with Coverage

```bash
pytest --cov=orchestra
```

## Running Type Checks

```bash
mypy orchestra/
```

## PR Requirements

Before submitting a pull request, ensure the following:

- All tests pass.
- Code coverage does not drop.
- No `except Exception: pass` bare except blocks are introduced.

## Commit Message Format

This project follows the [Conventional Commits](https://www.conventionalcommits.org/) specification. Commit messages should be structured as:

```
type(scope): description
```

Allowed types: `feat`, `fix`, `test`, `refactor`, `docs`, `chore`.

Example:

```
feat(storage): add SQLite backend for conversation history
```

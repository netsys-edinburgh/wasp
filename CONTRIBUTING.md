# Contributing to Wasp

Wasp is the companion code for the Wasp paper and builds on Morphling. We
welcome bug reports, features, docs, and code.

## Development environment

```bash
pip install pre-commit
pre-commit install --install-hooks
```

## Testing

Wasp's runnable example and coordinator paths require the Morphling runtime, so
those tests run inside the Wasp Docker image (which is `FROM` the Morphling
image):

```bash
docker build -t wasp:latest .
docker run --rm --gpus all --ulimit memlock=-1 wasp:latest \
    python3 -m pytest tests -v
```

The pure-Python placement solver tests run without Docker:

```bash
pip install -e ".[dev,plot]"
python3 -m pytest tests/test_placement_smoke.py -v
```

## Commit messages

Follow the [Angular Commit Format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#-commit-message-format):
`<type>: <summary>`, where type is one of `build`, `ci`, `docs`, `feat`, `fix`,
`perf`, `refactor`, `test`, `chore`.

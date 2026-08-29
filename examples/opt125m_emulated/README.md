# OPT-125M on emulated devices

Runs Wasp's coordinator over Morphling-emulated devices with the license-free
heuristic placement backend.

```bash
# 1) Build the Morphling base image (in the morphling checkout):
docker build -t device-emulator:wasp-api .

# 2) Build the Wasp image (in this repo):
docker build -t wasp:latest .

# 3) Run the example:
docker run --rm --gpus all --ulimit memlock=-1 wasp:latest run --tiny
# or, inside the container: bash examples/opt125m_emulated/run.sh
```

Expected: the run completes the configured steps and reports a finite per-rank
loss. This exercises the full path — placement, coordinator dispatch, emulated
device execution, and aggregation — end to end.

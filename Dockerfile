# Wasp builds on the Morphling runtime image so the native Torch extension and
# its CUDA/MKL/libevent/protobuf toolchain (and the morphling.api surface,
# tagged wasp-api-v2) are already present.
#
# Build the base from the Morphling checkout first:
#   docker build -t device-emulator:wasp-api .        # in the morphling repo
# then build Wasp:
#   docker build -t wasp:latest .                     # in this repo
# Override the base with --build-arg MORPHLING_IMAGE=... if you publish one.
ARG MORPHLING_IMAGE=device-emulator:wasp-api
FROM ${MORPHLING_IMAGE}

WORKDIR /opt/wasp
COPY . /opt/wasp

# Morphling (+ torch, transformers, numpy) is already installed in the base
# image; add Wasp's remaining pure-Python deps and install Wasp itself without
# reinstalling Morphling from git.
RUN python3 -m pip install --no-cache-dir "networkx>=3.0" "scipy>=1.10" "tqdm>=4.60" \
 && python3 -m pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["wasp"]

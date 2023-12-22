#!/bin/bash

# This script is set up to be run only from linux or apple silicon mac hosts
#
# USAGE: ./build_release_package.sh <platform>
#
# It takes one parameter to indicate which platform to build a package for: mac, windows or linux
#
# Cross compiling to x86_64 mac is done just with the toolchain
#       rustup target add x86_64-apple-darwin
#       cargo build --release --target=x86_64-apple-darwin
#
# Notes:
#   * On Windows, I used scoop to install zip

# Set the release tag
RELEASE_TAG="v0.2.0"

# Set the output directory for archives
OUTPUT_DIR="target"

# Additional files to include
SUPPORT_FILES="README.md config.yaml LICENSE characters/Vox.yaml"

# Set the name of the output archives
LINUX_ARCHIVE_NAME="sentientcore-linux_CUDA_x86_64_$RELEASE_TAG.tar.gz"
WINDOWS_ARCHIVE_NAME="sentientcore-windows_CUDA_x86_64_$RELEASE_TAG.zip"
MAC_ARM_ARCHIVE_NAME="sentientcore-mac_METAL_aarch64_$RELEASE_TAG.tar.gz"
MAC_INTEL_ARCHIVE_NAME="sentientcore-mac_METAL_x86_64_$RELEASE_TAG.tar.gz"

# Check if platform argument is provided
if [ $# -ne 1 ]; then
  echo "Please provide a platform argument: 'linux', 'windows' or 'mac'"
  exit 1
fi

# Read the platform argument
PLATFORM="$1"


if [ "$PLATFORM" == "linux" ]; then
    cargo build --release
    rm -f "$OUTPUT_DIR/$LINUX_ARCHIVE_NAME" 
    tar -czf "$OUTPUT_DIR/$LINUX_ARCHIVE_NAME" $SUPPORT_FILES -C target/release sentient_core 
elif [ "$PLATFORM" == "windows" ]; then
    cargo build --release --no-default-features --features cuda
    rm -f "$OUTPUT_DIR/$WINDOWS_ARCHIVE_NAME"
    zip -rj "$OUTPUT_DIR/$WINDOWS_ARCHIVE_NAME" "target/release/sentient_core.exe" $SUPPORT_FILES
elif [ "$PLATFORM" == "mac" ]; then
    cargo build --release --no-default-features --features metal
    cargo build --release --no-default-features --features metal --target=x86_64-apple-darwin
    rm -f "$MAC_ARM_ARCHIVE_NAME" "$MAC_INTEL_ARCHIVE_NAME"
    tar -czf "$OUTPUT_DIR/$MAC_ARM_ARCHIVE_NAME" $SUPPORT_FILES -C target/release sentient_core 
    tar -czf "$OUTPUT_DIR/$MAC_INTEL_ARCHIVE_NAME" $SUPPORT_FILES -C target/x86_64-apple-darwin/release sentient_core 
else
  echo "Invalid platform argument: '$PLATFORM'"
  echo "Please provide a platform argument: 'linux', 'windows' or 'mac'"
  exit 1
fi




echo "Archives created successfully."

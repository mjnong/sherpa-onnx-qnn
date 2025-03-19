#!/bin/bash

# Exit on error
set -e

# Define colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Define variables
ONNX_VERSION="1.21.0"
DOWNLOAD_URL="https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android-qnn/${ONNX_VERSION}/onnxruntime-android-qnn-${ONNX_VERSION}.aar"
AAR_FILE="onnxruntime-android-qnn-${ONNX_VERSION}.aar"
ZIP_FILE="onnxruntime-android-qnn-${ONNX_VERSION}.zip"

# Function to print colored messages
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Required command '$1' not found. Please install it first."
        exit 1
    fi
}

# Function to clean up on error
cleanup() {
    if [ $? -ne 0 ]; then
        log_error "An error occurred during execution!"
        # Clean up any partial downloads or unzipped files
        rm -f "${AAR_FILE}" "${ZIP_FILE}"
        exit 1
    fi
}

# Set up error handling
trap cleanup EXIT

# Print script header
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  ONNX Runtime QNN Downloader${NC}"
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Version: ${ONNX_VERSION}${NC}"
echo ""

# Check required commands
log_info "Checking required commands..."
check_command wget
check_command unzip
log_success "All required commands are available"

# Create build directory if it doesn't exist
BUILD_DIR="build-android-arm64-v8a"
if [ ! -d "$BUILD_DIR" ]; then
    log_info "Creating build directory: ${BUILD_DIR}"
    mkdir -p "$BUILD_DIR"
fi
cd "$BUILD_DIR"

# Create download directory if it doesn't exist
DOWNLOAD_DIR="${ONNX_VERSION}"
if [ ! -d "$DOWNLOAD_DIR" ]; then
    log_info "Creating download directory: ${DOWNLOAD_DIR}"
    mkdir -p "$DOWNLOAD_DIR"
fi
cd "$DOWNLOAD_DIR"

# Download the AAR file
log_info "Downloading ONNX Runtime QNN AAR file..."
if [ -f "${AAR_FILE}" ]; then
    log_warning "File ${AAR_FILE} already exists. Removing..."
    rm -f "${AAR_FILE}"
fi

wget --progress=bar:force:noscroll "${DOWNLOAD_URL}" -O "${AAR_FILE}" 2>&1
log_success "Download completed successfully"

# Rename and unzip
log_info "Preparing to extract files..."
cp "${AAR_FILE}" "${ZIP_FILE}"
log_info "Extracting files..."
unzip -o "${ZIP_FILE}" > /dev/null 2>&1

# Clean up temporary files
log_info "Cleaning up temporary files..."
rm -f "${ZIP_FILE}"

# Verify extraction
if [ -d "jni" ]; then
    log_success "Files extracted successfully to ${DOWNLOAD_DIR}"
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}Download and extraction complete!${NC}"
    echo -e "${GREEN}Files are located in: $(pwd)${NC}"
    echo -e "${GREEN}================================${NC}"
else
    log_error "Extraction failed or files are missing!"
    exit 1
fi
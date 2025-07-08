# DiffusionDrive Docker Setup Guide

This guide provides instructions for setting up and running DiffusionDrive using Docker containers.

## Recent Updates

### Version Changes (Latest)
- **PyTorch**: Updated from 2.4.1 to 2.7.1
- **TorchVision**: Updated from 0.19.1 to 0.22.1
- **PyTorch Lightning**: Updated from 1.9.4 to 2.4.0
- **Python**: Constrained to >=3.9,<=3.10
- **Security**: Updated multiple packages to fix vulnerabilities
- **Docker**: Genericized configuration for public release
- **Compatibility**: Fixed PyTorch 2.7.1 compatibility in WarmupCosLR scheduler

For detailed changes, see [CHANGELOG.md](../CHANGELOG.md).

## Prerequisites

- Docker installed on your system
- NVIDIA GPU drivers installed (for GPU support)
- NVIDIA Container Toolkit (nvidia-docker2) for GPU support
- At least 50GB of free disk space

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd DiffusionDrive
```

2. Build the Docker image:
```bash
cd docker
./build.sh
```

3. Run the container:
```bash
./run.sh
```

4. Install navsim package (inside container):
```bash
cd /workspace
pip install -e .
```

## Important: Customization Required

**You MUST modify the Docker setup scripts to match your environment:**

1. **User ID/Group ID**: Edit `build.sh` to match your system's user/group IDs:
   ```bash
   --build-arg USERID=$(id -u) \
   --build-arg GROUPID=$(id -g) \
   ```

2. **Data Directory**: Edit `run.sh` to point to your actual data location:
   ```bash
   export DATA_DIR=/path/to/your/nuplan/data
   ```

3. **Workspace Directory**: If not running from the repository root:
   ```bash
   export WORKSPACE_DIR=/path/to/DiffusionDrive
   ```

## Docker Image Details

### Base Image
- **CUDA Version**: 12.6.3 (supports H100 GPUs)
- **Ubuntu Version**: 22.04
- **Python Version**: 3.10

### Pre-installed Dependencies
The Docker image includes all required dependencies:
- PyTorch 2.7.1 with CUDA support
- NuPlan DevKit v1.2
- All Python packages from requirements.txt
- Development tools (git, vim, htop, etc.)

## Build Configuration

### Building with Custom Parameters

You can customize the build process by modifying the build arguments:

```bash
docker build \
    --build-arg CUDA_VER=12.6.3 \
    --build-arg UBUNTU_VER=22.04 \
    --build-arg PYTHON_VER=3.10 \
    --build-arg USERNAME=user \
    --build-arg USERID=$(id -u) \
    --build-arg GROUPID=$(id -g) \
    -t diffusiondrive:latest \
    -f dockerfile \
    .
```

### Build Arguments
- `CUDA_VER`: CUDA version (default: 12.6.3, minimum 11.8 for H100)
- `UBUNTU_VER`: Ubuntu version (default: 22.04)
- `PYTHON_VER`: Python version (default: 3.10)
- `USERNAME`: Container user name (default: user)
- `USERID`: User ID (should match your host user ID)
- `GROUPID`: Group ID (should match your host group ID)

## Running the Container

### Basic Usage

Run the container with GPU support:
```bash
./run.sh
```

### Advanced Usage with Environment Variables

You can customize the container runtime using environment variables:

```bash
# Set custom workspace directory
export WORKSPACE_DIR=/path/to/your/DiffusionDrive

# Set custom data directory (IMPORTANT: Set this to your NuPlan data location)
export DATA_DIR=/path/to/your/nuplan/data

# Set custom container name
export CONTAINER_NAME=my-diffusiondrive

# Run the container
./run.sh
```

### Volume Mounts

The run script automatically mounts the following directories:

1. **Workspace Directory**: 
   - Host: `${WORKSPACE_DIR}` or current directory
   - Container: `/workspace`

2. **Data Directory**:
   - Host: `${DATA_DIR}` or `/data`
   - Container: `/data`

3. **SSH Keys** (read-only):
   - Host: `~/.ssh`
   - Container: `/home/user/.ssh`

### GPU Support

The container is configured with:
- All GPUs enabled (`--gpus all`)
- Host IPC namespace for PyTorch multiprocessing (`--ipc host`)
- Unlimited memory lock (`--ulimit memlock=-1`)

## Working Inside the Container

### Directory Structure
- `/workspace`: Your project code (mounted from host)
- `/data`: Data directory (mounted from host)
- `/home/user`: User home directory

### Python Environment
- Python 3.10 is set as the default Python version
- All required packages are pre-installed
- pip is available for additional package installation

### Installing navsim

After entering the container, you need to install the navsim package:

```bash
cd /workspace
pip install -e .
```

**Note**: You only need to install the navsim package itself. All dependencies are already installed in the Docker image.

### Running DiffusionDrive

Once navsim is installed, you can run DiffusionDrive scripts:
```bash
# Example: Run PDM score evaluation
python navsim/planning/script/run_pdm_score.py
```

## Troubleshooting

### GPU Not Available
Ensure you have:
1. NVIDIA drivers installed on the host
2. NVIDIA Container Toolkit installed
3. Docker daemon restarted after installing nvidia-docker2

### Permission Issues
The container runs with a non-root user. If you encounter permission issues:
- Ensure your host directories have appropriate permissions
- Match the container user ID with your host user ID in `build.sh`
- The container user has sudo access without password if needed

### Build Failures
If the build fails:
1. Check your internet connection (for downloading packages)
2. Ensure you have sufficient disk space
3. Try building with `--no-cache` flag to force fresh downloads

### Container Already Exists
If you get an error about the container name already existing:
```bash
docker rm diffusiondrive  # or your custom container name
```

### PyTorch Version Compatibility
If you encounter issues with PyTorch 2.7.1:
- The WarmupCosLR scheduler has been fixed for compatibility
- Ensure your code is compatible with PyTorch 2.x API changes

## Security Considerations

- The container runs as a non-root user for security
- SSH keys are mounted read-only to prevent accidental modifications
- sudo access is configured for the container user when needed

## Performance Tips

1. **Data Location**: Keep your data on fast SSDs for better I/O performance
2. **GPU Memory**: Monitor GPU memory usage with `nvidia-smi` inside the container
3. **CPU Cores**: The container has access to all CPU cores by default

## Updating the Image

To update the Docker image with new dependencies:

1. Modify the `dockerfile` or `requirements.txt`
2. Rebuild the image:
   ```bash
   cd docker
   ./build.sh
   ```
3. Remove old containers and run with the new image

## Additional Notes

- The Dockerfile is based on the dockerdl project template
- All Python packages are installed in the user's local directory
- The container uses bash as the default shell
- Requirements are integrated directly into the Dockerfile for better caching
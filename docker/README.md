# Docker Files for DiffusionDrive

This directory contains Docker configuration files for running DiffusionDrive in a containerized environment.

## Files

- `dockerfile` - Main Docker image definition with CUDA, PyTorch, and all dependencies
- `build.sh` - Script to build the Docker image
- `run.sh` - Script to run the Docker container with proper GPU and volume configurations

## Quick Start

```bash
# Build the Docker image
./build.sh

# Run the container
./run.sh

# Inside container, install navsim
cd /workspace
pip install -e .
```

## Documentation

For detailed Docker setup instructions, including:
- Prerequisites and requirements
- Customization options
- Troubleshooting
- Performance optimization

See the full guide at [/docs/docker-setup.md](../docs/docker-setup.md)

## Version Information

For information about recent updates and version changes, see [/CHANGELOG.md](../CHANGELOG.md)
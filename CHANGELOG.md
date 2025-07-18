# DiffusionDrive Docker Changes

## Latest Updates (2025-07-08)

### PyTorch and Dependencies Update

#### Major Version Updates
- **PyTorch**: 2.4.1 → 2.7.1
  - Significant performance improvements
  - Better GPU memory management
  - Enhanced CUDA 12.x support
  - **Important**: Fixed compatibility issue with WarmupCosLR scheduler
  
- **TorchVision**: 0.19.1 → 0.22.1
  - Aligned with PyTorch 2.7.1
  - New transforms and models support
  
- **PyTorch Lightning**: 1.9.4 → 2.4.0
  - Breaking API changes - check your training scripts
  - Improved distributed training support
  - Better checkpoint handling

#### Python Version Constraints
- Changed from Python 3.9 to Python >=3.9,<=3.10
- Docker image uses Python 3.10 as default

#### Security Updates
- **Pillow**: Updated to >=10.4.0 (security fixes)
- **notebook**: Updated to >=7.3.2 (security fixes)
- **tornado**: Updated to >=6.4.2 (security fixes)
- **setuptools**: Updated to 78.1.1 (security fixes)

### Docker Configuration Changes

#### Genericization for Public Release
- Removed internal/proprietary configurations
- Made build arguments configurable
- Added environment variable support in run script

#### New Features
- Integrated requirements.txt directly into Dockerfile
  - Better Docker layer caching
  - Reduced build times for dependency updates
- Added diffusers and einops packages for diffusion model support
- Improved user/group ID handling

#### Build Script Updates
- Added explicit build arguments
- Better error messages
- Support for custom CUDA versions (minimum 11.8 for H100)

#### Run Script Updates
- Added environment variable support:
  - `WORKSPACE_DIR`: Custom workspace directory
  - `DATA_DIR`: Custom data directory
  - `CONTAINER_NAME`: Custom container name
- Improved documentation in script comments

### Compatibility Fixes

#### WarmupCosLR Scheduler
- Fixed compatibility with PyTorch 2.7.1
- The scheduler now properly handles the new PyTorch optimizer API

### Installation Process Changes
- Requirements are now installed during Docker build
- Only navsim package needs to be installed after container start
- Reduced setup time inside container

## Migration Guide

### From Previous Docker Version

1. **Rebuild the image** - Required due to PyTorch version change:
   ```bash
   cd docker
   ./build.sh
   ```

2. **Update your code** for PyTorch 2.7.1:
   - Check for deprecated APIs
   - Update any custom schedulers/optimizers
   - Review PyTorch Lightning code for API changes

3. **Modify build/run scripts** for your environment:
   - Set correct user/group IDs in build.sh
   - Set correct data directory in run.sh

### Known Issues and Workarounds

1. **PyTorch Lightning API Changes**:
   - Some callbacks may need updates
   - Trainer arguments might have changed

2. **GPU Memory**:
   - PyTorch 2.7.1 may have different memory patterns
   - Monitor and adjust batch sizes if needed

## Rollback Instructions

If you need to rollback to previous versions:

1. Checkout previous commit:
   ```bash
   git checkout <previous-commit-hash>
   ```

2. Rebuild Docker image with old configuration

3. Note: You may need to downgrade PyTorch in your local environment as well
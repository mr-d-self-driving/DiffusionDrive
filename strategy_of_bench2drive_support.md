# Strategy of Bench2drive Support

## Critical Context: Live CARLA Evaluation Requirement

**Important**: This project requires the trained model to control vehicles in real-time within the CARLA simulator. This requirement fundamentally influences the choice between the two approaches below.

## Approach 1: Convert Bench2Drive (CARLA) to NAVSIM Format 🔄

This approach involves a one-time, heavy data-processing task to transform your entire Bench2Drive dataset into the NAVSIM format that Diffusion Drive expects (specifically, the OpenScene data structure).

### What You Need to Do

1. **Develop a Conversion Script:** Write a comprehensive script that iterates through every scenario in the Bench2Drive dataset.
2. **Downsample Data:** CARLA data is high-frequency (e.g., 10-20Hz). You must downsample the frames and annotations to match NAVSIM's **2Hz** rate.
3. **Perform Core Transformations:** For every data point in every sampled frame, you must:
    * **Convert Coordinates:** Transform 3D points from CARLA's **Left-Handed** system to NAVSIM's **Right-Handed** system (e.g., `[x, y, z]` -> `[x, -y, z]`).
    * **Convert Rotations:** Transform yaw from **degrees (clockwise)** to **radians (counter-clockwise)**.
4. **Remap and Repackage:**
    * Map CARLA camera names to the expected NAVSIM camera slots.
    * Derive the simple `(left, straight, right)` command from the more complex CARLA route information.
    * Package all the transformed data into the specific file structure and format NAVSIM uses (e.g., aggregated log files in Pickle or Feather format).
5. **Mandatory Verification:** Create visualization tools to project the new 3D boxes onto 2D images and plot top-down trajectories to ensure your transformations are correct.

---

## Approach 2: Adapt Diffusion Drive to Read CARLA Data 🔧

This approach involves modifying the model's data-loading pipeline to make it compatible with the raw Bench2Drive dataset. The original data on disk remains unchanged.

### What You Need to Do

1. **Implement a New Data Loader:** This is the primary task. Create a new Python data loader class within the Diffusion Drive codebase that knows how to parse the Bench2Drive file structure (e.g., per-frame JSON files, separate sensor folders).
2. **Transform Data "In-Memory":** Instead of changing the files on disk, perform the necessary transformations inside the data loader just before the data is fed to the model.
    * The same coordinate and rotation transformation logic from Approach 1 is applied here, but "on-the-fly" in your code.
3. **Format Tensors:** Ensure the final output tensors from your new data loader (e.g., image batches, agent state vectors) have the **exact shape, data type, and normalization** that the Diffusion Drive model's `forward` method expects.
4. **Map Driving Command:** Convert the CARLA route information into the simplified `(left, right, straight)` integer or one-hot vector the model expects.

---

## Comparison of Approaches

| Aspect | Approach 1 (Data Conversion) | Approach 2 (Model Adaptation) |
| :--- | :--- | :--- |
| **Effort** | Heavy, one-time data processing task. Prone to errors that corrupt the entire dataset. | Focused software engineering task within the model's codebase. Easier to debug. |
| **Flexibility** | Low. You are locked into the converted dataset. | **High**. You can easily add flags to switch between training on NAVSIM and CARLA datasets. |
| **Data Integrity** | **Risky**. The original data is replaced. A bug in the script can permanently corrupt your dataset. | **Safe**. The original Bench2Drive data remains untouched. |
| **Alignment with Goal** | **Poor**. After training, you still need to write a *separate* data pipeline for your model to work with live CARLA data for evaluation. | ✅ **Excellent**. The model is trained and evaluated using the same data pipeline, making the transition from training to final evaluation seamless. |
| **Live CARLA Support** | ❌ **Requires duplicate implementation**. Need separate code to handle live sensor data. | ✅ **Native support**. Same loader works for both offline and online data. |
| **Caching** | Built-in (data is pre-converted) | Can be added (cache transformed data on first use) |

---

## Summary of Required Changes

| Data Element | Approach 1: Convert Dataset | Approach 2: Adapt Model |
| :--- | :--- | :--- |
| **Coordinates** | **On-Disk Conversion:** Convert all 3D points from LH to RH and save to new files. | **In-Memory Transformation:** Convert coordinates from LH to RH inside the data loader. |
| **Rotation** | **On-Disk Conversion:** Convert all yaw values (degrees/clockwise to rad/ccw) and save. | **In-Memory Transformation:** Convert yaw values inside the data loader. |
| **Driving Command** | **Pre-processing:** Analyze CARLA route to determine a simplified command and save it to the log. | **Live Mapping:** Map the CARLA command to the model's expected format inside the data loader. |
| **Data Structure** | **Complete Restructuring:** Repackage many small files (JSONs, JPGs) into large, aggregated NAVSIM logs. | **New Parser:** Write code to read the existing CARLA folder and file structure directly. |
| **Temporal Rate** | **Downsampling:** Discard frames to reduce the dataset from 10/20Hz to 2Hz on disk. | **Frame Selection:** The data loader's logic will only select every Nth frame to process. |

---

## Recommendation: Use Approach 2 (Model Adaptation)

Given the requirement for live CARLA evaluation, **Approach 2 is the clear choice**. It provides:

1. **Unified data pipeline** for training and live evaluation
2. **No data duplication** or conversion overhead
3. **Flexibility** to switch between datasets
4. **Native support** for real-time CARLA integration
5. **Caching capability** for training efficiency

The implementation effort is worthwhile as it directly supports your end goal of controlling vehicles in the CARLA simulator.

# Transwell Assay Code

This repository contains the Renca Transwell Migration Batch Cell Counter and Normalization script. It automates the process of watershed segmentation and intensity correction for cell counting.

## How to Use
1. Ensure your images are structured into specific control and experimental subfolders (e.g. `c1`, `s1`, etc.)
2. Run the script:
   ```bash
   python transwell_counter.py
   ```
3. Check the `cellcounter_results` directory and the generated CSV files for the counts and normalized data.

## Citation
If you use this repository in your research, **please consider starring the repository and citing it** to help support future development.

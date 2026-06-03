# Reproduction Notes

- CycleGAN is sensitive to data order, GPU kernels, image preprocessing, and dependency versions.
- This project reports metrics from generated directories rather than hard-coded thesis values.
- SSIM and LPIPS require paired or consistently ordered images; for unpaired translation, FID/KID are usually more meaningful.
- CMMD is reserved as an interface in `utils/metrics.py`; add a lab-approved implementation if exact CMMD reporting is required.


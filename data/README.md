# Data provenance and layout

## M-A island SEM dataset

- Source: https://figshare.com/articles/dataset/Image_data_and_labels/19232523
- DOI: `10.6084/m9.figshare.19232523.v2`
- License reported by Figshare: MIT
- Figshare file ID: `34171098`
- Published archive size: `876503324` bytes
- Published archive MD5: `0c17b87a65f6ff198eb6f4dd459e9330`

The source archive contains many redundant pre-generated crops. Prototype 1
uses only the combined appendix at
`dataset/appendix/i_ii_iii_iv_combined`, which contains 40 native-resolution
TIFF images and COCO polygon annotations.

Generated layout:

```text
data/
├── raw/ma_islands/
│   ├── dataset.zip
│   └── extracted/dataset/appendix/i_ii_iii_iv_combined/
├── processed/ma_islands/
│   ├── masks/
│   ├── manifests/
│   └── coarse/
│       ├── images/
│       ├── masks/
│       └── coarse_manifest.csv
└── splits/ma_islands/
    ├── fold_assignments.csv
    └── cv5_splits.csv
```

Raw and processed data are ignored by Git. Split manifests contain only paths,
identifiers, and assignments and are versioned for reproducibility.

The source COCO files set every annotation's `area` field to `3145728`, the
entire image area. The audit deliberately ignores that field, rasterizes the
polygon coordinates, and recomputes foreground pixel counts from the masks.

Step 2 creates one 512 x 384 coarse image per native image using bilinear
anti-aliased resampling. Binary masks are resized with nearest-neighbor
interpolation so that no intermediate label values are introduced.

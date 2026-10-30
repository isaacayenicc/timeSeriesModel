# Source-data placement and provenance

Copy your original sigtrain.mat and sigvalid.mat into data/raw. Preserve the source fdt_featcal.m and updated course PDF for reference. The release excludes original raw data because redistribution rights are not established.

Put all 20 separately uploaded sigfault1_part_001.mat through sigfault1_part_020.mat, the supplied manifest.json and optional sourceREADME together in data/raw/sigfault1_parts. They contain200 complete records, each32,000 samples at3,200 Hz. The8fields are t,fs,N,spd,spdnorm,x,spdmin,spdmax. No concatenation into an artificial continuous signal or additional xstandardization is performed. Source README and the included scripts/sigfault1_splitter.py describe splitting/reconstruction. Production verification independently validates checksums, logical field shapes/dtypes/data, ranges, count, order and scientific vector/time conditions.

The combined logical digest is705731dbfe79605f9ef10de1146c99cc2db95e85ab9f3b03f1cf2a14f51bf701. The original monolithic MAT byte fingerprint is source-manifest metadata; its exact original bytes were not independently available. A logically equivalent reconstructed MAT need not share its container hash.

No health/severity/onset labels or physical speed-unit metadata occur in the inspected structs. sigtest.mat and sigfault2.mat were not provided. The per-record audit and filemanifest record measured ranges and speed extrapolation; no clipping or invented labels is applied. Allraw parts remain unchanged.

The original supplied split manifest is also retained as data/fault_partition_manifest.json (metadata only). If needed, copy that exact file to data/raw/sigfault1_parts/manifest.json beside your twenty unchanged MAT parts.

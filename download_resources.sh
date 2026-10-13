#!/bin/bash
# Download the germline filtering resources for tumor-only mode (VarNet-T, GRCh38)
# from figshare (https://doi.org/10.6084/m9.figshare.34420128) and check their md5 checksums.
# Usage: download_resources.sh [output_dir]   (default: varnet_resources)
# Then pass the folder to filter.py and predict.py with --resource_dir.
set -euo pipefail

OUTPUT_DIR="${1:-varnet_resources}"
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "Downloading VarNet-T resources (3.9 GB) to $OUTPUT_DIR"
curl -L --fail --retry 5 -o 1000g_pon.hg38.vcf.gz https://ndownloader.figshare.com/files/70003872
curl -L --fail --retry 5 -o GrCh38-dbSNP-build-156-common-variants-0-indexed-CHROMPOS.npz https://ndownloader.figshare.com/files/70003932
curl -L --fail --retry 5 -o combined_gnomAD_v4.1_genomes_exomes_dict.hg38.chunks.npz https://ndownloader.figshare.com/files/70003977

md5sum -c <<'EOF'
e236330d8d156d2aad2d0930a2440177  1000g_pon.hg38.vcf.gz
1b01425dd8427c2f4e5fbc096fcab65c  GrCh38-dbSNP-build-156-common-variants-0-indexed-CHROMPOS.npz
fe306bf13d090b0f8be4ad12ab029f18  combined_gnomAD_v4.1_genomes_exomes_dict.hg38.chunks.npz
EOF

echo "Done. Run tumor-only mode with --resource_dir $OUTPUT_DIR"

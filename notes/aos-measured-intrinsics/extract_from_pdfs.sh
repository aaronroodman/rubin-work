#!/usr/bin/env bash
# Pull the MIW-build figures out of the pipeline's per-rotbin PDFs into figures/.
# Source: the rot_-3_3 (central) build of fam_danish_1_0 / pathA_50_34.
# Needs pdftoppm (poppler) + ImageMagick (convert).  Re-run after a rebuild.
set -euo pipefail
cd "$(dirname "$0")"
B=../../../aos/output/fam_danish_1_0_wep17_3_0_bin2x/pathA_50_34_i/build/rot_-3_3
V=$B/measured_intrinsic_nkeep_34_pathA_validation.pdf
C=$B/measured_intrinsic_nkeep_34_pathA_comparison.pdf
N=$B/measured_intrinsic_nkeep_34_pathA_final.pdf
F=figures; mkdir -p $F; R=200

# 1. validation p1 upper-left panel "V (normalized DOF x v-mode)" -> crop
pdftoppm -png -singlefile -r $R -f 1 -l 1 "$V" /tmp/_val_p1
read W H < <(identify -format "%w %h" /tmp/_val_p1.png)
convert /tmp/_val_p1.png -crop \
  "$(python3 -c "print(int($W*.50))")x$(python3 -c "print(int($H*.47))")+0+$(python3 -c "print(int($H*.03))")" \
  +repage "$F/miw_V_dof_vmode.png"

# 2. comparison p2 = Pupil Z5 iterations (original / iter1-3 / tabulated)
pdftoppm -png -singlefile -r $R -f 2 -l 2 "$C" "$F/miw_Z5_iterations"

# 3. the 3 MIW result pages (Z4..Z26 maps)
pdftoppm -png -singlefile -r $R -f 1 -l 1 "$N" "$F/miw_final_1"
pdftoppm -png -singlefile -r $R -f 2 -l 2 "$N" "$F/miw_final_2"
pdftoppm -png -singlefile -r $R -f 3 -l 3 "$N" "$F/miw_final_3"

# 4-6. validation pages: residual histograms / FWHM-equiv edge / cov+corr
pdftoppm -png -singlefile -r $R -f 9  -l 9  "$V" "$F/miw_residual_histograms"
pdftoppm -png -singlefile -r $R -f 16 -l 16 "$V" "$F/miw_fwhm_edge"
pdftoppm -png -singlefile -r $R -f 18 -l 18 "$V" "$F/miw_covariance_correlation"
echo "done -> $F/"

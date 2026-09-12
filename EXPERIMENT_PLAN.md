# Controlled experiment plan

All variants use the same dataset manifests, image size, optimizer, scheduler, checkpoint rule, threshold and seeds. The only changed factor is shown below.

| Variant | Changed factor |
|---|---|
| `bcrr_full` | grouped, compressed, dual-axis Mamba with router, gate and state control |
| `lb_unet_baseline` / `cnn_only` | remove Mamba, retain the same U-shaped local backbone and boundary heads |
| `plain_mamba` | remove boundary gate, router and state control |
| `single_axis` | remove vertical scans |
| `fixed_direction` | replace learned spatial routing by uniform direction weights |
| `no_boundary_gate` | remove the feature gate and reset supervision |
| `no_state_reset` | keep boundary-aware feature mixing but remove reset conditioning from scans |
| `no_grouping` | use one Mamba channel group |
| `no_compression` | remove channel compression before Mamba |
| `mamba_32` | Mamba only at 32x32 |
| `mamba_16_8` | Mamba only at 16x16 and 8x8 |
| `mamba_16` | Mamba only at 16x16 |
| `no_boundary_loss` | remove boundary supervision from the objective |
| `no_gate_loss` | remove gate supervision while retaining the gate architecture |
| `no_contour_loss` | remove contour continuity supervision |
| `segmentation_only` | retain only BCE + Dice for the main and auxiliary masks |

Primary metrics are Dice, IoU, mIoU, HD95 and Boundary F1. Secondary metrics are accuracy, sensitivity, specificity, precision and FPS. Every final evaluation writes a per-image CSV and, when `--save-predictions` is passed, probability, binary mask, boundary and overlay PNGs.

For a paper claim, aggregate at least three seeds with `summarize.py`. Treat a component as supported only when its mean change is accompanied by a consistent per-image direction and no material efficiency regression.


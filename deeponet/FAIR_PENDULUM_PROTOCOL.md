# Parameter-matched pendulum protocol

The historical width1000 comparison was not parameter matched and test results were inspected during configuration discussions. Test labels did not enter backpropagation, but these scores are not an untouched final test. Preserve all historical runs as exploratory records.

The user now permits changing network widths, depth, grade count and epoch allocation, and requires equal total model parameters. Training-MSE checkpoint selection remains mandatory; validation selects configurations only.

## Architecture

Four-grade MultiGrade uses the original hidden-feature transfer, frozen previous grades, branch/trunk inner products, per-grade scalar biases and unweighted cumulative residual predictions. Each hidden/readout width is w.

Traditional is a conventional unstacked DeepONet with branch `[100,w,w,w,w,4w+5]` and trunk `[1,w-1,w-1,w-1,w-1,4w+5]`. Hidden activations are ReLU; branch output is linear; trunk output is ReLU. Both heads have biases and the final inner product has a scalar bias. No masks, residual grades, Fourier/local features or padding parameters enter Traditional.

Both have exactly `14*w*w + 117*w + 4` total learned parameters. At w64:64836; at w128:244356. Both longest affine paths have five maps. Total affine maps differ (Traditional10, MultiGrade16), trunk hidden widths differ by one, readout widths differ, and active trainable parameters per update differ. Count equality is verified before every run; it is not a claim of identical computational cost or runtime.

## Selection and data separation

- Same unchanged original training set:10000 forcing/query pairs; same separate validation set:5000 new functions.
- CUDA, full batches, TF1-style Adam, seed42 and truncated-Glorot initialization family for both. Different graphs do not have identical initial tensors.
- Each candidate gets4000 updates per method. MultiGrade receives1000 per grade, now explicitly changed from2800/400/400/400 under the user's new epoch-allocation authorization. Traditional trains jointly throughout.
- Same six tuning configurations per method: widths64/128 and starting rates0.0003/0.001/0.003. All MultiGrade grades use the candidate LR; both decay ratios are0.01. Traditional decay is global; MultiGrade decay restarts per grade, with fresh per-grade Adam. Runtime is measured, not forced equal.
- Checkpoints remain selected by training MSE at initialization/every1000 updates/segment end. Validation never enters gradient computation.
- Per width, each method chooses its best LR by validation MSE. Choose a common width by the smallest geometric mean of the two methods' best validation MSEs, avoiding choosing the budget solely to favor one model.
- Tuning code loads NO test data and outputs NO test columns or curves.
- Freeze configuration, source hashes and training/validation hashes. Only then generate a fresh100000-function final test set, seed20261107. Retrain the frozen pair and record every-epoch train/validation/test metrics. Test values never select weights or settings.
- Report 20x success only if both final training-selected checkpoint MSE ratios meet20. Do not retune against this new final test after inspecting it.

## 2026-10-31 extension: batches, normalization and SVD

The completed development search is `results/20261031_214645_510144_fair_preprocessing_search`.
Its frozen configuration supersedes the earlier full-batch freeze. The six original
paired configurations were extended with six: widths64/128, batch1000, and
normalization none/standard/regularized-whitening. Each method received12
configurations total; the new groups reused each method's best LR at that width
from the full-batch screen. This is a bounded pilot, not an exhaustive joint search.

The selected common group is width64, batch1000, standard normalization.
Traditional LR is0.001; MultiGrade LR is0.003 for each grade. Both decay ratios
are0.01. Four grades receive1000 updates each; Traditional receives4000 joint
updates. Both models have64836 parameters and receive the identical shuffled
sample sequence (SHA256 checked), totaling4000000 examples or400 complete data
passes. The legacy CSV field `epoch` counts optimizer updates, not data passes.
The protocol's old prose phrase `full-batch update budget` is inaccurate for this
extension; the numeric `batch_size=1000` and examples-seen fields are authoritative.
The frozen executable source was retained unchanged for final verification.

Standard normalization subtracts each sensor's training mean and divides by its
training standard deviation, maps query t to2t/3-1, and divides targets by training
RMS. No target offset or extra Grade0 is added. Both methods receive identical
fixed transforms with no learnable parameters. Reported MSE is restored to
physical units. Whitening instead uses training covariance with ridge0.001 times
its largest eigenvalue; all100 sensor coordinates are retained.

Development results at the selected group: Traditional train1.658984566e-4,
validation2.249834204e-3; MultiGrade train8.650418787e-5,
validation1.824806297e-3. Ratios1.91781 and1.23292 do not meet20.
Grade3 improves validation slightly; Grade4 worsens it while reducing training
MSE. Training-selected checkpoints are preserved as requested.

SVD diagnosis is saved in `study_notes/pendulum_svd_diagnostic` and explained in
`study_notes/PENDULUM_NORMALIZATION_SVD.md`. It uses training input covariance
and a separate development Cartesian target matrix, never a final-test residual.

To reproduce the selected configuration in local VS Code:

1. File > Open Folder > `D:\Workspace_codex_DeepONet\Workspace_codex_MultiGrade_DeepONet`.
2. Ctrl+Shift+P > Python: Select Interpreter > Enter interpreter path >
   `D:\Workspace_codex_DeepONet\Workspace_codex_MultiGrade_DeepONet\.venv\python.exe`.
3. Terminal > New Terminal, then run:

```powershell
Set-Location "D:\Workspace_codex_DeepONet\Workspace_codex_MultiGrade_DeepONet"
& ".\.venv\python.exe" -B ".\tools\run_fair_pendulum.py" --width 64 --batch-size 1000 --normalization standard --grade-steps 1000 1000 1000 1000 --traditional-lr 0.001 --grade-lrs 0.003 0.003 0.003 0.003 --decay-ratio 0.01
```

That command is a development run with train/validation only, on CUDA. Final-test
reproduction requires the exact frozen configuration and its generated final
test file. Do not rank new settings using that final test after inspection.

## Earlier full-batch commands (historical development screen)

Open this project folder and use `.venv/python.exe`.

```powershell
& ".\.venv\python.exe" -B ".\tools\run_fair_pendulum.py" --check
& ".\.venv\python.exe" -B ".\tools\search_fair_pendulum.py"
```

The first is a CUDA-only eight-update check. The second is a real six-configuration-per-method search; do not launch a duplicate while it is running. It prints the frozen JSON path. Final-test generation and final training use that exact path, with `make_pendulum_validation.py --role final_test --count 100000 --seed 20261107 --frozen-config <path>` and `run_fair_pendulum.py --frozen-config <path> --final-test <generated-test.npz>`.

All training runs retain every update in CSV/Markdown and embedded SVG metadata. Development figures show train/validation only. Final figures include every-epoch test data after the configuration has been frozen. Dark navy theme and two model curves remain.


## 2026-10-31: frozen fresh-test outcome

Final CUDA run `results/20261031_215916_138326_fair_pendulum_final` completed. Both models have64836 parameters and4000 updates. Traditional train/test MSE1.658984566e-4/1.379008963e-3; MultiGrade8.650418787e-5/1.232621234e-3. Ratios1.917808/1.118761:20x target NOT achieved. All-update train/validation trajectories exactly match development, identical minibatch hashes pass, and7 SVGs retain4000 data rows per model with no smoothing/subsampling. SVGs validated and overview visually inspected. Final RESULT.md records timings and fairness limits. Fresh test is now reporting evidence, not a source for further tuning.

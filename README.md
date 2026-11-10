# UDL Assignment 2 (Group 91) — Image Generation using (β-)VAE, VQ-VAE and WGAN

**Course:** Unsupervised Deep Learning (S1-26_AIMLZG533)

## Status

All code (`udl_utils.py`, `build_notebooks.py`, and the three notebooks) was written and
**smoke-tested** (`QUICK_RUN = True`) on a local Mac with no CUDA GPU, to catch code-level bugs
cheaply before spending real GPU time. **The full training run (`QUICK_RUN = False`) has not
been executed yet.** Pick **one** of the platform sections below and follow it top to bottom —
each is fully self-contained (yes, the steps repeat across platforms on purpose, so you never
have to jump around). Once training finishes on whichever platform you used, skip to
**"Common steps after training"** near the end — that part is the same everywhere.

## How RESUME actually works

`RESUME = True` checkpoints (model + optimizer state) after **every epoch** for VAE/VQ-VAE/
PixelCNN, and every `QUICK_FID_EVERY` generator iterations for the GAN. So a disconnect/crash
loses at most one epoch (or one `QUICK_FID_EVERY` block of iterations), not the whole run —
just re-run the same cell/command and it continues from exactly where it left off. (An earlier
version of this code only saved the final checkpoint at the very end of each model's training,
which meant an interrupted run restarted that model from scratch — fixed now.)

## Folder layout

```
.
├── Assignment2_Problem_Statement.pdf
├── udl_utils.py                              # shared utilities (device, loaders, PSNR, FID, I/O)
├── build_notebooks.py                        # builds the three notebooks below via nbformat
├── requirements.txt                          # everything EXCEPT torch/torchvision
├── UDL_Group91_Assignment2_PartA.ipynb        # (beta-)VAE + VQ-VAE/PixelCNN
├── UDL_Group91_Assignment2_PartB.ipynb        # WGAN + WGAN-GP
├── UDL_Group91_Assignment2_PartC.ipynb        # helper notebook, builds the Part C PDF
├── outputs/
│   ├── checkpoints/   # model weights (*.pt)
│   ├── history/       # per-epoch/iteration JSON logs
│   ├── results/       # final metrics JSON + CSV tables (partA_results.json, partB_results.json)
│   ├── figures/       # PNG plots and sample grids
│   ├── latents/       # VQ-VAE discrete latent index maps (*.npy)
│   └── logs/          # stdout/stderr logs from background full-run processes
└── submission/         # final deliverables (populated after the full run + HTML/PDF export)
```

---

# Group 1 — BITS Lab (Prayogshala) GPU machine

### 1.1 Get the files there

```bash
git clone https://github.com/VijayYD/S1-26_AIMLZG533_Assignment2_Group91.git
cd S1-26_AIMLZG533_Assignment2_Group91
```

(If the repo is private, `git clone` prompts for credentials — use a
[personal access token](https://github.com/settings/tokens) as the password.)

### 1.2 Check the GPU and existing PyTorch install

**Do not upgrade, downgrade, or reinstall torch/torchvision** — the lab machine already has
`torch 2.8.0+cu128` installed with the GPU detected. `requirements.txt` deliberately excludes
torch/torchvision so installing it never touches them.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Confirm the version/CUDA-availability printed is the **same before and after** installing
`requirements.txt`. If it changed, stop and reinstall the lab's original torch build before
continuing. If `torchvision` is missing:

```bash
python -m pip install torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 1.3 Build the notebooks

Already built and committed. Only needed if you change `build_notebooks.py`:

```bash
python build_notebooks.py
```

### 1.4 Smoke test (QUICK_RUN = True)

Confirms this exact GPU/torch build works before committing real GPU hours — a few minutes.

First, make the `QUICK_RUN = True` copies. **This is Python code — don't paste it directly
into the bash prompt**, it'll fail with `bash: import: command not found`. Write it to a file
and run that file with `python3`:

```bash
cat > toggle_quickrun.py << 'PYEOF'
import nbformat as nbf
for name in ['UDL_Group91_Assignment2_PartA.ipynb', 'UDL_Group91_Assignment2_PartB.ipynb']:
    nb = nbf.read(name, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'QUICK_RUN = False' in cell.source:
            cell.source = cell.source.replace('QUICK_RUN = False', 'QUICK_RUN = True')
    nbf.write(nb, name.replace('.ipynb', '_quicksrc.ipynb'))
PYEOF
python3 toggle_quickrun.py
```

Then run the smoke tests:

```bash
python -m jupyter nbconvert --to notebook --execute --output UDL_Group91_Assignment2_PartA_quick.ipynb --ExecutePreprocessor.timeout=1800 UDL_Group91_Assignment2_PartA_quicksrc.ipynb
python -m jupyter nbconvert --to notebook --execute --output UDL_Group91_Assignment2_PartB_quick.ipynb --ExecutePreprocessor.timeout=1800 UDL_Group91_Assignment2_PartB_quicksrc.ipynb
```

Confirm both commands exit with no traceback and print the `RUN MODE: QUICK_RUN (smoke test)`
banner. Then **clean up before the real run** — important, `RESUME = True` would otherwise
mistake these 2-epoch smoke-test checkpoints for a finished full run:

```bash
rm -f UDL_Group91_Assignment2_PartA_quicksrc.ipynb UDL_Group91_Assignment2_PartA_quick.ipynb
rm -f UDL_Group91_Assignment2_PartB_quicksrc.ipynb UDL_Group91_Assignment2_PartB_quick.ipynb
rm -rf outputs data
```

### 1.5 Full run (QUICK_RUN = False)

The committed notebooks already have `QUICK_RUN = False`. Expect several hours total, dominated
by the 50,000-iteration WGAN/WGAN-GP runs and PixelCNN training — launch each as a background
process and monitor its log:

```bash
mkdir -p outputs/logs
nohup python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 UDL_Group91_Assignment2_PartA.ipynb > outputs/logs/partA.log 2>&1 &
tail -f outputs/logs/partA.log
```

Wait for Part A to finish (`Ctrl+C` just stops `tail`, not the background job), then run Part B
the same way (sequentially, unless you've confirmed GPU memory clearly allows both at once):

```bash
nohup python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 UDL_Group91_Assignment2_PartB.ipynb > outputs/logs/partB.log 2>&1 &
tail -f outputs/logs/partB.log
```

`RESUME = True` means either notebook can be safely re-run after a crash or a lab-session
timeout — it reloads any checkpoint + history already on disk and continues.

**→ Once both finish, jump to "Common steps after training" below.**

---

# Group 2 — Google Colab

### 2.1 Get the files there

Clone straight into Google Drive so the project (and its checkpoints) survive a runtime
disconnect — `/content` is wiped every time Colab disconnects, which would destroy
mid-training checkpoints; Drive persists.

1. New Colab notebook → **Runtime → Change runtime type → T4 GPU** (or whatever tier you have)
   → Save.
2. First cell:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   %cd /content/drive/MyDrive
   !git clone https://github.com/VijayYD/S1-26_AIMLZG533_Assignment2_Group91.git
   %cd /content/drive/MyDrive/S1-26_AIMLZG533_Assignment2_Group91
   ```

   If the repo is private, `git clone` prompts for credentials — use a
   [personal access token](https://github.com/settings/tokens) as the password, or clone with
   `https://<token>@github.com/VijayYD/S1-26_AIMLZG533_Assignment2_Group91.git`.

**Important — every shell command below needs a `!` in front of it in a Colab cell.** A Colab
cell runs Python by default, not shell, so pasting `python -c "..."` without `!` fails with
`SyntaxError: invalid syntax`. `%cd` is a magic (no `!`); everything else shown as a `bash`
block below needs `!` added when you paste it into Colab.

### 2.2 Check the GPU and existing PyTorch install

**Do not upgrade, downgrade, or reinstall torch/torchvision** — use whatever Colab's current
runtime already provides. `requirements.txt` deliberately excludes torch/torchvision.

```python
!python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
!python -m pip install -r requirements.txt
!python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Confirm the version/CUDA-availability is the **same before and after**. Torchvision ships
bundled with Colab's torch, so the lab's separate torchvision install step isn't needed here.

### 2.3 Build the notebooks

Already built and committed. Only needed if you change `build_notebooks.py`:

```python
!python build_notebooks.py
```

### 2.4 Smoke test (QUICK_RUN = True)

Confirms this GPU/torch build works before committing real GPU hours — a few minutes.

**Cell 1 — plain Python, no `!`** (makes the `QUICK_RUN = True` copies):

```python
import nbformat as nbf
for name in ['UDL_Group91_Assignment2_PartA.ipynb', 'UDL_Group91_Assignment2_PartB.ipynb']:
    nb = nbf.read(name, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'QUICK_RUN = False' in cell.source:
            cell.source = cell.source.replace('QUICK_RUN = False', 'QUICK_RUN = True')
    nbf.write(nb, name.replace('.ipynb', '_quicksrc.ipynb'))
```

**Cell 2 — shell:**

```python
!python -m jupyter nbconvert --to notebook --execute --output UDL_Group91_Assignment2_PartA_quick.ipynb --ExecutePreprocessor.timeout=1800 UDL_Group91_Assignment2_PartA_quicksrc.ipynb
```

**Cell 3 — shell:**

```python
!python -m jupyter nbconvert --to notebook --execute --output UDL_Group91_Assignment2_PartB_quick.ipynb --ExecutePreprocessor.timeout=1800 UDL_Group91_Assignment2_PartB_quicksrc.ipynb
```

Confirm both cells run with no traceback and print `RUN MODE: QUICK_RUN (smoke test)`. Then
**clean up before the real run** — important, `RESUME = True` would otherwise mistake these
2-epoch smoke-test checkpoints for a finished full run:

```python
!rm -f UDL_Group91_Assignment2_PartA_quicksrc.ipynb UDL_Group91_Assignment2_PartA_quick.ipynb
!rm -f UDL_Group91_Assignment2_PartB_quicksrc.ipynb UDL_Group91_Assignment2_PartB_quick.ipynb
!rm -rf outputs data
```

### 2.5 Full run (QUICK_RUN = False)

The committed notebooks already have `QUICK_RUN = False`. Expect several hours total — **free
Colab disconnects after ~90 minutes idle and at most ~12 hours continuous**, and the
50,000-iteration WGAN/WGAN-GP runs plus 50-epoch VAE/PixelCNN training will likely outlast one
sitting. That's fine: `RESUME = True` means you just re-run the same `nbconvert` command after
reconnecting and it continues from the last saved checkpoint instead of restarting. Keep the
tab active to reduce idle disconnects, or lower `WGAN_GEN_ITERS` / `WGANGP_GEN_ITERS` / the
epoch counts in the config cells if your session budget is tight (note any such change in
"Deviations" below).

```python
!mkdir -p outputs/logs
!python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 UDL_Group91_Assignment2_PartA.ipynb
```

This runs in the foreground and the cell stays busy until Part A finishes (that's normal and
fine on Colab — there's no background-job convenience here the way a lab terminal has). Once
it completes, do the same for Part B:

```python
!python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 UDL_Group91_Assignment2_PartB.ipynb
```

If the cell gets interrupted by a disconnect, just reconnect and re-run the same cell —
`RESUME = True` picks up where it left off.

**→ Once both finish, jump to "Common steps after training" below.**

---

# Group 3 — Kaggle Notebooks

### 3.1 Get the files there

1. New Kaggle Notebook → right panel → **Settings → Accelerator → GPU** (T4 x2 or P100,
   whichever is offered) → **Settings → Internet → On** (needed for `git clone` and `pip
   install`).
2. First cell:

   ```python
   %cd /kaggle/working
   !git clone https://github.com/VijayYD/S1-26_AIMLZG533_Assignment2_Group91.git
   %cd /kaggle/working/S1-26_AIMLZG533_Assignment2_Group91
   ```

   If the repo is private, `git clone` prompts for credentials — use a
   [personal access token](https://github.com/settings/tokens) as the password.

**Important — every shell command below needs a `!` in front of it in a Kaggle cell**, same
reason as Colab: a notebook cell runs Python by default, not shell.

**Kaggle session persistence is different from Colab/Drive** — `/kaggle/working` is **not**
automatically kept across a fresh interactive session the way Google Drive is. Within one
active session it persists fine across cells/time (so `RESUME = True` works for reconnects
within that session, and Kaggle GPU sessions run up to ~12 hours). But if you *end* the
session and start a new one, `/kaggle/working` resets. To carry checkpoints across sessions,
periodically zip and download `outputs/`, or "Save Version" (Kaggle's commit/snapshot feature)
so the output is preserved and can be reloaded as an input to a new session.

### 3.2 Check the GPU and existing PyTorch install

**Do not upgrade, downgrade, or reinstall torch/torchvision** — use whatever Kaggle's current
runtime already provides. `requirements.txt` deliberately excludes torch/torchvision.

```python
!python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
!python -m pip install -r requirements.txt
!python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Confirm the version/CUDA-availability is the **same before and after**. Torchvision ships
bundled with Kaggle's torch.

### 3.3 Build the notebooks

Already built and committed. Only needed if you change `build_notebooks.py`:

```python
!python build_notebooks.py
```

### 3.4 Smoke test (QUICK_RUN = True)

Confirms this GPU/torch build works before committing real GPU hours — a few minutes.

**Cell 1 — plain Python, no `!`:**

```python
import nbformat as nbf
for name in ['UDL_Group91_Assignment2_PartA.ipynb', 'UDL_Group91_Assignment2_PartB.ipynb']:
    nb = nbf.read(name, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'QUICK_RUN = False' in cell.source:
            cell.source = cell.source.replace('QUICK_RUN = False', 'QUICK_RUN = True')
    nbf.write(nb, name.replace('.ipynb', '_quicksrc.ipynb'))
```

**Cell 2 — shell:**

```python
!python -m jupyter nbconvert --to notebook --execute --output UDL_Group91_Assignment2_PartA_quick.ipynb --ExecutePreprocessor.timeout=1800 UDL_Group91_Assignment2_PartA_quicksrc.ipynb
```

**Cell 3 — shell:**

```python
!python -m jupyter nbconvert --to notebook --execute --output UDL_Group91_Assignment2_PartB_quick.ipynb --ExecutePreprocessor.timeout=1800 UDL_Group91_Assignment2_PartB_quicksrc.ipynb
```

Confirm both cells run with no traceback and print `RUN MODE: QUICK_RUN (smoke test)`. Then
**clean up before the real run**:

```python
!rm -f UDL_Group91_Assignment2_PartA_quicksrc.ipynb UDL_Group91_Assignment2_PartA_quick.ipynb
!rm -f UDL_Group91_Assignment2_PartB_quicksrc.ipynb UDL_Group91_Assignment2_PartB_quick.ipynb
!rm -rf outputs data
```

### 3.5 Full run (QUICK_RUN = False)

```python
!mkdir -p outputs/logs
!python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 UDL_Group91_Assignment2_PartA.ipynb
```

Runs in the foreground — the cell stays busy until Part A finishes. Then:

```python
!python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 UDL_Group91_Assignment2_PartB.ipynb
```

If you're worried about hitting the session time limit, periodically zip and download
`outputs/` (or "Save Version") so a checkpoint is never lost even if the session ends
unexpectedly — `RESUME = True` will continue from whatever's in `outputs/checkpoints/` +
`outputs/history/` once you restore it into a new session.

**→ Once both finish, jump to "Common steps after training" below.**

---

# Group 4 — Any other CUDA machine (own box, cloud VM, etc.)

Same as Group 1 (BITS Lab) — it's just a terminal with a CUDA GPU:

1. `git clone https://github.com/VijayYD/S1-26_AIMLZG533_Assignment2_Group91.git && cd S1-26_AIMLZG533_Assignment2_Group91`
2. Follow **1.2 → 1.5** above exactly (check torch/CUDA without touching the existing install,
   smoke test, then the full run).

**→ Once both finish, jump to "Common steps after training" below.**

---

# Common steps after training

These apply no matter which platform you used.

## Sanity-check results, retrain if needed

Check `outputs/results/partA_results.json` and `partB_results.json` against the expected
characteristics (VQ-VAE PSNR clearly above VAE PSNR, WGAN-GP FID lower than WGAN, no model
still improving sharply at its last epoch/iteration). If a model looks under-trained, increase
its epoch/iteration count in the relevant config cell, re-run just that notebook, and record
the change in "Deviations" below.

## Build Part C and export HTML

```bash
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 UDL_Group91_Assignment2_PartC.ipynb
python -m jupyter nbconvert --to html UDL_Group91_Assignment2_PartA.ipynb
python -m jupyter nbconvert --to html UDL_Group91_Assignment2_PartB.ipynb
python -m jupyter nbconvert --to html UDL_Group91_Assignment2_PartC.ipynb
```

(Add `!` in front of each line if you're on Colab/Kaggle.)

## Verify, then collect the submission

Open each HTML and the PDF; confirm the student table, all required plots/tables/grids are
present, no error tracebacks, and the `RUN MODE: FULL RUN` banner (not `QUICK_RUN`) is printed
near the top. Then copy into `submission/`:

- `UDL_Group91_Assignment2_PartA.ipynb` + `.html`
- `UDL_Group91_Assignment2_PartB.ipynb` + `.html`
- `UDL_Group91_Assignment2_PartC.pdf`
- `UDL_Group91_Assignment2_Lab_Screenshot.jpg` — **take this manually**: a screenshot of the
  session showing the environment cell / `nvidia-smi` output with the notebook running, as
  proof of GPU infrastructure in use. This cannot be generated automatically — if the
  assignment specifically requires the **BITS-provided** infrastructure for this screenshot,
  make sure at least the graded full run (or an equivalent one) was done on Group 1, not only
  on Colab/Kaggle.

## What to send back here

Once the full run finishes, share:
- `outputs/results/partA_results.json` and `outputs/results/partB_results.json` (or just paste
  the printed results tables), so the results table and runtimes below can be filled in from
  the real numbers.
- Any errors or tracebacks you hit that smoke-testing here didn't catch.
- Which platform(s) you ran on and roughly how long each part took.

## Configuration used

| Flag | Value |
|---|---|
| `SEED` | 42 |
| `QUICK_RUN` | `False` for submission |
| `RESUME` | `True` |
| Preprocessing | `transforms.ToTensor()` only — pixels in [0,1], no augmentation, no resizing |
| VAE | β ∈ {1,2,4,10}, 50 epochs, Adam lr=1e-3, batch 128 |
| VQ-VAE | K ∈ {512,256,128}, 40 epochs, Adam lr=2e-4, batch 128, EMA decay 0.99, commitment 0.25 |
| PixelCNN prior | 12 gated residual layers, 128 channels, 50 epochs, Adam lr=3e-4, batch 128 |
| WGAN | RMSprop lr=5e-5, clip [-0.01,0.01], n_critic=5, 50,000 generator iterations |
| WGAN-GP | Adam lr=1e-4 (β=0,0.9), gradient penalty λ=10, n_critic=5, 50,000 generator iterations |
| FID protocol | `torchmetrics` `FrechetInceptionDistance(feature=2048, normalize=True)`, 10,000 real vs 10,000 generated images, identical for every model |

## Total runtimes

_To be filled in after the full run (see `outputs/results/*.json` → `total_train_time_s` /
`train_time_s` fields and the per-epoch/iteration `epoch_time_s` / `iter_time_s` history)._

## Final results table

_To be filled in from `outputs/results/partA_results.json` and `partB_results.json` once the
full run completes — do not hand-type values here._

## Deviations from the prompt, and why

1. **Local build/smoke-test machine was not a GPU machine.** The working directory was created
   on a local Intel-based macOS laptop with no CUDA GPU (Intel UHD Graphics 630 integrated GPU
   only) and no PyTorch preinstalled. All code was written and smoke-tested there
   (`QUICK_RUN = True`, on CPU/Apple MPS via a throwaway local `.venv_smoketest` virtual
   environment — **not part of this submission**) to catch code-level bugs cheaply, but the
   full `QUICK_RUN = False` run was deliberately not attempted locally and is being run on GPU
   infrastructure instead, per the platform sections above. Two issues specific to that local
   Apple MPS backend were found and fixed in `udl_utils.py` / `build_notebooks.py` (an
   MPS/float64 limitation in the FID metric, and MPS memory growth over long training loops);
   neither affects CUDA, so they're corrections, not deviations, for the CUDA runs this README
   targets.
2. _(Any further deviations — changed epoch counts, platform used, retrained models — found
   while running the full pipeline will be listed here.)_

## Submission checklist

- [ ] `UDL_Group91_Assignment2_PartA.ipynb` + `.html` (full run, `QUICK_RUN = False`)
- [ ] `UDL_Group91_Assignment2_PartB.ipynb` + `.html` (full run, `QUICK_RUN = False`)
- [ ] `UDL_Group91_Assignment2_PartC.pdf`
- [ ] `UDL_Group91_Assignment2_Lab_Screenshot.jpg` — **take manually**, showing BITS-provided
      GPU infrastructure in use (see note above)

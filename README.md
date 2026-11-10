# UDL Assignment 2 (Group 91) — Image Generation using (β-)VAE, VQ-VAE and WGAN

**Course:** Unsupervised Deep Learning (S1-26_AIMLZG533)

## Status

All code (`udl_utils.py`, `build_notebooks.py`, and the three notebooks) was written and
**smoke-tested** (`QUICK_RUN = True`) on a local Mac with no CUDA GPU, to catch code-level bugs
cheaply before spending real GPU time. **The full training run (`QUICK_RUN = False`) has not
been executed yet.** You will run it yourself — on the BITS Lab (Prayogshala) GPU machine
and/or Google Colab — following the steps below, then report the results back.

These notebooks are plain PyTorch + `pathlib`, with no OS-specific shell magics and the
required `cuda → mps → cpu` device order, so the same commands work on **any machine with a
Python environment**: BITS Lab, Google Colab, Kaggle, your own GPU box, etc. Platform-specific
notes (getting the files there, session persistence) are called out separately below.

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

**Before you start a real run on any platform**, delete the `outputs/` folder if it already
exists (it may contain my local `QUICK_RUN = True` smoke-test artifacts, which must never be
picked up by `RESUME = True` as if they were a finished full run):

```bash
rm -rf outputs
```

## Step 0 — Get the files onto the platform

### Option A: BITS Lab (Prayogshala) GPU machine

Easiest: `git clone` the repo directly on the lab machine (it's public-enough to this group,
or use a personal access token as the password if it's private):

```bash
git clone https://github.com/VijayYD/S1-26_AIMLZG533_Assignment2_Group91.git
cd S1-26_AIMLZG533_Assignment2_Group91
```

Otherwise, copy this whole folder onto the lab machine (USB drive, `scp`, shared drive) —
whatever the lab normally uses for file transfer. Example with `scp`:

```bash
scp -r "UDL Assignment2" <lab-username>@<lab-host>:~/udl-assignment2
```

Then `ssh` in and `cd` into that folder for every command below.

### Option B: Google Colab

The project is on GitHub, so the simplest way to get it into Colab is `git clone` straight
into your Google Drive — no zip/upload step needed.

1. Open a new Colab notebook → **Runtime → Change runtime type → T4 GPU** (or whatever GPU
   tier you have access to) → Save.
2. In the first Colab cell, mount Drive and clone the repo there:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   %cd /content/drive/MyDrive
   !git clone https://github.com/VijayYD/S1-26_AIMLZG533_Assignment2_Group91.git
   %cd /content/drive/MyDrive/S1-26_AIMLZG533_Assignment2_Group91
   ```

   **Keep the project on Google Drive, not `/content` directly** — `/content` is wiped every
   time the Colab runtime disconnects, which would destroy checkpoints mid-training. Drive
   persists, so `RESUME = True` can pick back up after a disconnect.

   If the repo is private, you'll be prompted for GitHub credentials on `git clone` — use a
   [personal access token](https://github.com/settings/tokens) as the password, or clone with
   `https://<token>@github.com/VijayYD/S1-26_AIMLZG533_Assignment2_Group91.git` instead.

   If you already have the folder locally and prefer not to use git, zipping it and uploading
   to Drive still works — just make sure the `zip_path` in your unzip cell matches exactly
   where you uploaded it in the Drive file browser (the error you'll see otherwise is
   `FileNotFoundError: No such file or directory: '...zip'`).
3. **Every subsequent command in this README shown in a `bash` block needs a `!` in front of
   it when you paste it into a Colab cell** — a Colab code cell runs Python by default, not
   shell, so pasting `python -c "..."` without `!` fails with `SyntaxError: invalid syntax`
   (it's trying to run `python` as a Python statement). Prefix **every line**, e.g.:

   ```python
   !python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   !python -m pip install -r requirements.txt
   !python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

   The `%cd` from step 2 is a magic, not a shell command — it has no `!` and Colab remembers
   the directory for the rest of the session.

**Colab session limits:** free-tier Colab disconnects after ~90 minutes idle and at most ~12
hours continuous. The 50,000-iteration WGAN/WGAN-GP runs and the 50-epoch VAE/PixelCNN runs
may not finish in one sitting. That's fine — `RESUME = True` means you can just re-run the same
`nbconvert` command after reconnecting and it will continue from the last saved checkpoint
instead of restarting. Keep the browser tab open/active to avoid idle disconnects, or reduce
`WGAN_GEN_ITERS` / `WGANGP_GEN_ITERS` / the epoch counts in the config cells if your session
budget is tight (note any such change in the "Deviations" section below).

### Option C: any other machine with a CUDA GPU (your own box, Kaggle, etc.)

Same as Option A — copy the folder over, then follow the steps below from a terminal (or from
notebook cells with `!` if it's a hosted-notebook platform like Kaggle).

## Step 1 — Check the GPU and existing PyTorch install

**Do not upgrade, downgrade, or reinstall torch/torchvision** — use whatever the platform
already provides (the BITS lab has `torch 2.8.0+cu128` preinstalled; Colab ships its own
current CUDA build). `requirements.txt` deliberately excludes torch/torchvision so installing
it never touches them.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Confirm the torch version/CUDA-availability printed is the **same before and after** installing
`requirements.txt`. If it changed, something in `requirements.txt` pulled in a different torch —
stop and reinstall the platform's original build before continuing. If `torchvision` is
missing on the BITS lab machine specifically:

```bash
python -m pip install torchvision --index-url https://download.pytorch.org/whl/cu128
```

(On Colab, torchvision is already bundled with torch — this won't be needed there.)

## Step 2 — Build the notebooks

The three `.ipynb` files are already built and committed, but if you ever change
`build_notebooks.py`, rebuild with:

```bash
python build_notebooks.py
```

## Step 3 — Smoke test (QUICK_RUN = True)

Already done locally (on CPU/Apple MPS, no CUDA) to catch code-level bugs. **Repeat it here**
to confirm the environment itself (this GPU, this torch build) works before committing real
GPU hours — it only takes a few minutes:

```bash
python -c "
import nbformat as nbf
for name in ['UDL_Group91_Assignment2_PartA.ipynb', 'UDL_Group91_Assignment2_PartB.ipynb']:
    nb = nbf.read(name, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'QUICK_RUN = False' in cell.source:
            cell.source = cell.source.replace('QUICK_RUN = False', 'QUICK_RUN = True')
    nbf.write(nb, name.replace('.ipynb', '_quicksrc.ipynb'))
"
python -m jupyter nbconvert --to notebook --execute --output UDL_Group91_Assignment2_PartA_quick.ipynb --ExecutePreprocessor.timeout=1800 UDL_Group91_Assignment2_PartA_quicksrc.ipynb
python -m jupyter nbconvert --to notebook --execute --output UDL_Group91_Assignment2_PartB_quick.ipynb --ExecutePreprocessor.timeout=1800 UDL_Group91_Assignment2_PartB_quicksrc.ipynb
```

Open the two `_quick.ipynb` outputs (or just check the command exited with no traceback) and
confirm every cell ran cleanly and the `RUN MODE: QUICK_RUN (smoke test)` banner printed.

Then **clean up before the real run** — this is important, `RESUME = True` would otherwise pick
up these 2-epoch smoke-test checkpoints as if they were the finished full run:

```bash
rm -f UDL_Group91_Assignment2_PartA_quicksrc.ipynb UDL_Group91_Assignment2_PartA_quick.ipynb
rm -f UDL_Group91_Assignment2_PartB_quicksrc.ipynb UDL_Group91_Assignment2_PartB_quick.ipynb
rm -rf outputs data
```

## Step 4 — Full run (QUICK_RUN = False)

The committed notebooks already have `QUICK_RUN = False`. This is long-running (expect several
hours, dominated by the 50,000-iteration WGAN/WGAN-GP runs and the PixelCNN training) — launch
each as a background process and monitor its log rather than waiting in the foreground:

```bash
mkdir -p outputs/logs
nohup python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 UDL_Group91_Assignment2_PartA.ipynb > outputs/logs/partA.log 2>&1 &
```

Watch progress with `tail -f outputs/logs/partA.log`. Wait for Part A to finish, then run
Part B the same way (run them sequentially unless you've confirmed GPU memory clearly allows
both at once):

```bash
nohup python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 UDL_Group91_Assignment2_PartB.ipynb > outputs/logs/partB.log 2>&1 &
```

`RESUME = True` means either notebook can be safely re-run after a crash, a Colab disconnect,
or a lab-session timeout without losing finished models — it reloads any checkpoint + history
already on disk and continues.

## Step 5 — Sanity-check results, retrain if needed

After both finish, check `outputs/results/partA_results.json` and `partB_results.json` against
the expected characteristics (VQ-VAE PSNR clearly above VAE PSNR, WGAN-GP FID lower than WGAN,
no model still improving sharply at its last epoch/iteration). If a model looks under-trained,
increase its epoch/iteration count in the relevant config cell, re-run just that notebook, and
record the change in "Deviations" below.

## Step 6 — Build Part C and export HTML

```bash
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 UDL_Group91_Assignment2_PartC.ipynb
python -m jupyter nbconvert --to html UDL_Group91_Assignment2_PartA.ipynb
python -m jupyter nbconvert --to html UDL_Group91_Assignment2_PartB.ipynb
python -m jupyter nbconvert --to html UDL_Group91_Assignment2_PartC.ipynb
```

## Step 7 — Verify, then collect the submission

Open each HTML and the PDF; confirm the student table, all required plots/tables/grids are
present, no error tracebacks, and the `RUN MODE: FULL RUN` banner (not `QUICK_RUN`) is printed
near the top. Then copy into `submission/`:

- `UDL_Group91_Assignment2_PartA.ipynb` + `.html`
- `UDL_Group91_Assignment2_PartB.ipynb` + `.html`
- `UDL_Group91_Assignment2_PartC.pdf`
- `UDL_Group91_Assignment2_Lab_Screenshot.jpg` — **take this manually**: a screenshot of the
  lab/Colab session showing the environment cell / `nvidia-smi` output with the notebook
  running, as proof of using GPU infrastructure for the run. This cannot be generated
  automatically — if the assignment specifically requires the **BITS-provided** infrastructure
  for this screenshot, make sure at least the graded full run (or an equivalent one) was done
  there, not only on Colab.

## What to send back here

Once the full run finishes on whichever platform you used, share:
- `outputs/results/partA_results.json` and `outputs/results/partB_results.json` (or just paste
  the printed results tables), so the results table and runtimes in this README can be filled
  in from the real numbers.
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
   full `QUICK_RUN = False` run was deliberately not attempted locally and is being run on
   BITS Lab / Colab GPU infrastructure instead, per Step 0–4 above. Two issues specific to
   that local Apple MPS backend were found and fixed in `udl_utils.py` /
   `build_notebooks.py` (an MPS/float64 limitation in the FID metric, and MPS memory growth
   over long training loops); neither affects CUDA, so they're corrections, not deviations, for
   the CUDA runs this README targets.
2. _(Any further deviations — changed epoch counts, platform used, retrained models — found
   while running the full pipeline will be listed here.)_

## Submission checklist

- [ ] `UDL_Group91_Assignment2_PartA.ipynb` + `.html` (full run, `QUICK_RUN = False`)
- [ ] `UDL_Group91_Assignment2_PartB.ipynb` + `.html` (full run, `QUICK_RUN = False`)
- [ ] `UDL_Group91_Assignment2_PartC.pdf`
- [ ] `UDL_Group91_Assignment2_Lab_Screenshot.jpg` — **take manually**, showing BITS-provided
      GPU infrastructure in use (see note in Step 7)

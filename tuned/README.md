# Tuned builds of vLLM

Single-arch (one GPU variant, one CUDA toolkit) vLLM wheels, published as GitHub releases of this fork.
The design this follows is `docs/ARCHITECTURE.md` in zbrad/tuned-common (requirement IDs below, e.g. B3,
refer to it). Shared logic is vendored as `tuned/common.sh`; hardware and toolkit facts as
`tuned/common-data/`; keep both current with `tuned/sync-common.sh check|sync`. Repo-specific settings
are in `tuned/repo.env`. Variants: `gb10` (DGX Spark), `rtx40` and `rtx50`. Only `gb10` is built and
tested so far; `rtx40` and `rtx50` are untested here.

## Flow

```
CUDA_TAG=cu134 bash tuned/build.sh gb10        # build venv, dependencies, then the unstamped wheel (the long compile)
CUDA_TAG=cu134 bash tuned/full_test.sh gb10    # install that wheel --no-deps in the venv and run the gate
CUDA_TAG=cu134 bash tuned/package.sh gb10      # stamp staged copies, sha256, package-info.txt, DRAFT notes
CUDA_TAG=cu134 bash tuned/smoke_test.sh gb10   # install the packaged wheel, check it, run three upstream tests
# review tuned/releases/RELEASE_NOTES_<version>.md, remove the DRAFT banner, commit and push
CUDA_TAG=cu134 bash tuned/release.sh gb10      # every gate reported together; on pass writes RELEASE_READY_gb10 (publishes nothing)
CUDA_TAG=cu134 bash tuned/publish.sh gb10      # publish exactly what release.sh's gates saw (refuses if HEAD, notes or assets changed)
```

`tuned/wheel.sh <variant>` is the wheel step that `build.sh` calls after it has installed the venv; it can be
run on its own but refuses without that venv.

One of `CUDA_TAG` or `CUDA_VER` is required: there is no default toolkit. With neither, the scripts fail and
list the defined toolkits. `gb10` builds are refused for any toolkit but `cu134`: the pinned torch, flash-attention
and flashinfer wheels exist for cu134 only (`GPU_TUNED_REPO_TOOLKITS` in `tuned/repo.env`).

`tuned/run_gb10.sh` and `tuned/run_rtx50.sh` serve with the vLLM of the toolkit's venv (`CUDA_TAG` names the tree),
never a vLLM found on `PATH`.

## Where things are

Everything a build makes lives in `tuned/build/<cuda-tag>/` (ignored by git, one tree per toolkit, see C2):

- `venv/`: the build and runtime environment for that toolkit. The build never installs into the user site.
- `cmake/`: the CMake build tree and setuptools' build directory (`build --build-base`), so the repo's everyday
  `build/` is never written (C6).
- `ccache/`: the compiler cache for that toolkit (C4).
- `wheel/`: the unstamped wheel that `full_test.sh` tests. `package.sh` never modifies it (H3).
- `dist/`: the stamped wheel, its `.sha256`, and `package-info.txt` (written last).
- `releases/`: `TEST_RESULTS_<variant>.log`, `SMOKE_RESULTS_<variant>.log`, and `RELEASE_READY_<variant>` once
  `release.sh` passes.
- `release-label` and `build-commit`: written by `wheel.sh` at the end of a successful build.

Tracked release notes are `tuned/releases/RELEASE_NOTES_<version>.md`. `package.sh` writes a DRAFT if none exists.

## Identity

The wheel's version is vLLM's public version plus `+<variant>.cu<cuda>.<YYYYMMDD>[.<seq>]` (E1), e.g.
`0.11.2.dev278+gb10.cu134.20261019.0`. The date is the UTC committer date of the tuned commit and the sequence,
always written, is the lowest unused one among the repo's tags (E3). `wheel.sh` allocates the label once and
records it in `release-label`; `package.sh` and `release.sh` read it and never recompute it (E4). The release
tag is `v<version>`.

setuptools_scm's local part (`+g<sha>`, `.d<date>` when dirty) is dropped from a tuned build's version, so the
label is the identity. The commit is in the build info as `fork.commit`.

## Build info

The two extensions this repo's CMake builds with device code, `vllm/_C_stable_libtorch.abi3.so` and
`vllm/_moe_C_stable_libtorch.abi3.so`, carry the JSON build info (H7, section `.tuned_build_info`) in the
packaged wheel. Read it with `gpu_tuned_read_build_info <file> | jq .`. `package.sh` checks each carries device
code and a single target architecture before stamping.

## Test scope (recorded decision, G3)

`full_test.sh` runs: the gates (wheel label, build commit, the extensions load from the wheel, toolkit
libraries resolve, single-arch); the import check that `build.sh` used to run at its end; the stand-in
acceptance test `tuned/gb10_acceptance_test.sh` (small models for NemotronH and DeepSeek-V4, load and generate,
needs a Hugging Face download); and `python -m pytest tuned/tests`. The log ends with
`=== Full test suite summary: acceptance + tuned tests passed ===` only when all of these pass. It names what
it skips in its header: upstream's `tests/` suite is not run.

## Deviations and open items

- **Dependencies in the test venv.** The test and smoke steps use the toolkit's build venv, which holds the
  dependency set `build.sh` pinned from `requirements/gb10.txt` (or `cuda.txt`). A clean-venv resolution would
  re-download the GB10 torch and flashinfer wheels for every test. The wheel itself is installed `--no-deps`.
- **F4 (not met for torch).** vLLM ships no shared library with a stock torch name, and `package.sh` refuses one
  (its extension modules are named by module). But the extensions NEED torch's stock-named libraries from the
  zbrad/pytorch wheel, which still ships them (see the pytorch repo's `tuned/README.md`).
- **NCCL** comes from the torch wheel. There is no `nccl.pins` entry here, because vLLM links no NCCL itself.
- **Source files written by a build.** setuptools_scm writes `vllm/_version.py` and `vllm.egg-info/`, and
  FetchContent writes `.deps/`. All are gitignored; none is in the wheel's identity.
- **`VLLM_GB10_LOCAL_VERSION` and `gb10_version.py`** keep their GB10 names for every variant; the variant is in
  the value.
- **Stand-in acceptance models** are not the target models; the real checkpoints are not downloaded here.
- **Upstream's `AGENTS.md` and `.agents/`** are upstream's files. They are not followed by these scripts.

# Blob I/O Guide — Singularity jobs, dev box, and how data flows between them

**Last updated:** 2026-05-29
**Companion to:** `SWE_MASTER_SING_GCR_REPORT.md`. Read this when you need to **persist outputs from Singularity → blob → dev box**, or **stage inputs the other direction**.

## 1. The three blob mounts you'll see

| Where | Account | Container | Mount path | Used for |
|---|---|---|---|---|
| Inside Singularity job | `zhibinmain` | `murongma` | `/mnt/murongma` | **Canonical data path.** All trajectories, checkpoints, coordination files. |
| Dev box (`GCRAZGDL1683`) | `shuailu1` | `murong` | `/home/v-murongma/blob/murong/` | Where this report and tunnel artifacts live. Source code mirror. |
| Dev box (alt mount, **unreliable**) | (varies) | (varies) | `/home/v-murongma/blob/cache/` | Writes appear momentarily then vanish. **Do not use.** |

Critically: the Singularity mount (`zhibinmain/murongma`) is **NOT** auto-mounted on the dev box. To see files written by a Singularity job from the dev box, you must use a SAS URL + `azcopy` (see §3).

## 2. Persisting outputs from a Singularity job

**Pattern:** write everything to a path under `/mnt/murongma/<your-stage>/<run-id>/...`.

```bash
# in your Singularity job's command:
RUN_ID="${AMLT_EXPERIMENT_NAME}"          # set automatically by amlt
OUT="/mnt/murongma/sft_data/raw_trajectories/${RUN_ID}"
mkdir -p "$OUT"

python -m r2egym.agenthub.run.edit runagent_multiple \
   --traj_dir "$OUT" \
   ...
sync     # blobfuse2 needs an explicit sync to flush; not always required, but cheap insurance
```

**Verification inside the job (proved 2026-05-29 with `sing_blob_write_probe`):**
- 184 B text file: written, visible in blob via `azcopy list`. ✅
- 1.00 MiB binary: written, visible in blob, byte-for-byte. ✅
- Subdirectory creation: works. ✅
- Round-trip read inside the same job: works. ✅

**Standard layout** (per `DIND_INVESTIGATION_REPORT.md` §9.1) — please don't invent new top-levels without updating that:

```
/mnt/murongma/
├── sft_data/raw_trajectories/<run-id>/
├── sft_data/filtered/<run-id>/
├── sft_checkpoints/<run-id>/{step_<n>/,latest}
├── rl_data/trajectories/<run-id>/<batch-uuid>.parquet
├── rl_data/test_specs/<dataset>/
├── rl_checkpoints/<run-id>/{step_<n>/,latest}
└── coordination/<run-id>/{vllm_url.txt,policy_version.txt,heartbeat_*.txt}
```

## 3. Reading / copying outputs from the dev box

You'll need a SAS URL for the `zhibinmain/murongma` container, since the dev box doesn't mount it.

### 3.1 The current SAS (rotates — check expiry!)

Stored at: `~/blob/murong/code/SWE-Master/cred/zhibinmain_murongma_sas.url` (mode 600).

Helper to load it without comments:

```bash
SAS=$(bash ~/blob/murong/code/SWE-Master/cred/load_sas.sh)
```

Header in that file always shows the validity window. When near expiry, **regenerate from Azure Portal** → Storage account `zhibinmain` → Shared access signature (or per-container under Containers → murongma → Shared access tokens). Replace the URL line in the file; keep mode 600.

### 3.2 List what a Singularity job wrote

```bash
SAS=$(bash ~/blob/murong/code/SWE-Master/cred/load_sas.sh)
azcopy list "$SAS" --output-type=text | grep '<your-run-id>'
# or:
azcopy list "$SAS" --output-type=text | head -50
```

### 3.3 Download a file or a directory to the dev box

```bash
SAS=$(bash ~/blob/murong/code/SWE-Master/cred/load_sas.sh)

# extract base URL and SAS-token separately (azcopy copy wants them concatenated)
# Easiest: just append the path with the same ?sas after the container:
BASE="${SAS%%\?*}"   # https://zhibinmain.blob.core.windows.net/murongma
QS="${SAS#*\?}"      # sv=...&sig=...

# Single file
azcopy copy "${BASE}/sft_data/raw_trajectories/<run-id>/inst_0.json?${QS}" \
            ./inst_0.json

# Whole directory (recursive)
azcopy copy "${BASE}/sft_data/raw_trajectories/<run-id>/?${QS}" \
            ./local_copy/ --recursive
```

### 3.4 Upload from dev box (stage inputs into the Singularity-visible blob)

Same shape, source/destination flipped:

```bash
azcopy copy ./my_input_dataset/ \
            "${BASE}/rl_data/test_specs/my_dataset/?${QS}" --recursive
```

Then your Singularity job sees it at `/mnt/murongma/rl_data/test_specs/my_dataset/`.

## 4. Handing data between pipeline stages

There is **no** automatic IPC between Singularity, AML, and the dev box — only Blob. Stages connect via well-known paths under `/mnt/murongma/...`:

| Producer | Consumer | Channel |
|---|---|---|
| SFT data synth (Singularity rollout actors) | SFT trainer (Singularity training job) | `/mnt/murongma/sft_data/filtered/<run>/train.parquet` |
| SFT trainer | RL trainer | `/mnt/murongma/sft_checkpoints/<run>/latest` |
| RL trainer (writes new ckpt) | Policy server (in-job, hybrid engine) | Ray-internal weight sync (not Blob) |
| RL rollout (env.step → tunnel) | RL trainer | `/mnt/murongma/rl_data/trajectories/<run>/*.parquet` |
| Anywhere | Dev box for inspection | `azcopy copy` with SAS (see §3.3) |

For long async stages, write an "atomic marker" file (e.g., `latest` text file that contains the directory name) only **after** the actual directory is done. Consumers poll the marker file, not the directory listing.

## 5. Gotchas (learned the hard way)

- **`/blob/cache/...` is broken** on this dev box — writes appear, then disappear within seconds. Stick to `/blob/murong/...` (account `shuailu1/murong`) on the dev box.
- **`/mnt/murongma` and `/home/v-murongma/blob/murong` are different blob containers** in different accounts. Cross-visibility requires a SAS+azcopy hop.
- **Empty result earlier**: if a Singularity job appears to produce nothing in `/mnt/murongma`, the cause is almost always (a) the job crashed before reaching the write, (b) it wrote to a path *outside* `/mnt/...`, or (c) you're listing the wrong container. The blob mount itself works (proven 2026-05-29).
- **azcopy refuses URL with `&` unquoted** — always wrap the SAS URL in double quotes.
- **SAS expiry**: the saved SAS is **valid until 2026-06-05**. Set a calendar reminder and rotate; otherwise everything in §3 stops working.

## 6. Quick reference: prove the chain still works

```bash
# 1) submit a 30-second probe that writes a unique marker
amlt run /home/v-murongma/blob/murong/code/singularity_scripts/config/dind_test/sing_blob_write_probe.yaml \
         sing_blob_write_probe_$(date +%Y%m%d_%H%M)

# 2) tail the job log; look for marker_<stamp>_<rand>.txt
amlt logs <experiment> :blob_write_probe -f

# 3) verify on dev box
SAS=$(bash ~/blob/murong/code/SWE-Master/cred/load_sas.sh)
azcopy list "$SAS" --output-type=text | grep "marker_<stamp>"
```

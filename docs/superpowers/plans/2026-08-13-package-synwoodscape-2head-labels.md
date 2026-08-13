# Package SynWoodScape 2-Head Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the finalized ROI 8/4/+-6 occupancy and H=0.8 visibility labels into one training-ready local dataset folder, then remove temporary preview/prototype files.

**Architecture:** Add one focused packaging CLI under `tools/` that converts reviewed RGB occupancy labels into binary `*_occupancy.npy` files and copies training `*_visible.npy` masks. Keep the finalized reviewed label folder as source-of-truth and create a separate training root with metadata.

**Tech Stack:** Python 3.11, NumPy, Pillow, pytest, local ignored `dataset/` and `work_dirs/` folders.

## Global Constraints

- User-facing communication remains Korean.
- Do not modify `third_party/`.
- Final training ROI is `front=8m`, `rear=4m`, `half_width=6m`, `cell=0.05m`, shape `240x240`.
- Occupancy convention is drivable `1`, non-drivable `0`.
- Visibility convention is boolean training mask from H=0.8 gather-column visibility with ego excluded.
- `dataset/` contents are local and ignored by git.
- `work_dirs/` temporary scripts/previews can be deleted after packaging and verification.

---

### Task 1: Training Label Packaging CLI

**Files:**
- Create: `tools/package_synwoodscape_2head_labels.py`
- Create: `tests/tools/test_package_synwoodscape_2head_labels.py`

**Interfaces:**
- Consumes: `final_rgb/<sample_id>.png` and `visibility_h08/visible/<sample_id>.npy`
- Produces: `<output>/<sample_id>_occupancy.npy`, `<output>/<sample_id>_visible.npy`, `<output>/metadata.json`, `<output>/README.md`

- [ ] **Step 1: Write failing test**
- [ ] **Step 2: Run test to verify it fails because the script is missing**
- [ ] **Step 3: Implement minimal CLI and functions**
- [ ] **Step 4: Run test to verify it passes**

### Task 2: Generate Local Training Dataset

**Files:**
- Source: `dataset/annotated_roi_8-4-6_semantic_crop/`
- Output: `dataset/synwoodscape_2head_roi_8_4_6_h08/`

- [ ] **Step 1: Run the packaging CLI**
- [ ] **Step 2: Verify 500 occupancy and 500 visible npy files**
- [ ] **Step 3: Verify representative arrays have shape `240x240`, binary values, and matching ids**

### Task 3: Remove Temporary Work Files

**Files:**
- Delete ignored local folder contents under `work_dirs/`

- [ ] **Step 1: Delete `work_dirs` temporary files**
- [ ] **Step 2: Recreate empty `work_dirs/.gitkeep` only if needed**
- [ ] **Step 3: Verify `work_dirs` no longer contains preview/prototype files**

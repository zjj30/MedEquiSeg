#!/usr/bin/env python3
"""Build a clean, deterministic MedEquiSeg Public-5 source-and-results release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


RELEASE_DATE = "20260725"
ARCHIVE_PREFIX = "medequiseg_public5_reproducibility_release"
TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".svg",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}

# The release is a whitelist. Historical audits, backup manuscript sources,
# exploratory result families, checkpoints, logs, and private-cohort material
# are not selected and therefore cannot enter the archive accidentally.
EXACT_PATHS = (
    "README.md",
    "CITATION.cff",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "paper/protocol_v3_public5_release_readme.md",
    "paper/protocol_v3_environment_record.md",
    "paper/protocol_v3_split_provenance_audit_20260715.md",
    "paper/reference_metadata_audit_20260715.md",
    "paper/BMC_SUBMISSION_REQUIREMENTS_CHECKLIST.md",
    "paper/CLAIM_2024_checklist.md",
    "paper/submission_metadata_public5_template.yaml",
    "paper/analysis/audit_protocol_v3_public_grouping.py",
    "paper/analysis/benchmark_medclipseg_complexity.py",
    "paper/analysis/build_protocol_v3_public5_release_package.py",
    "paper/analysis/generate_complexity_table.py",
    "paper/analysis/generate_complete_base_sensitivity_statistics.py",
    "paper/analysis/generate_medequiseg_base_contrast_statistics.py",
    "paper/analysis/generate_medequiseg_factorial_manuscript_assets.py",
    "paper/analysis/generate_protocol_v3_qualitative_overlay.py",
    "paper/analysis/generate_protocol_v3_semantic_control_overlay.py",
    "paper/analysis/generate_strict_factorial_figure.py",
    "paper/analysis/summarize_medequiseg_no_rewrite_matched.py",
    "paper/analysis/summarize_covid_groupval_sensitivity.py",
    "paper/revision/STRATIFIED_DERANGEMENT_CONTROL_20260719.md",
    "paper/revision/analyze_stratified_derangement_control.py",
    "paper/revision/audit_atconv_forward_activation.py",
    "paper/revision/build_stratified_derangement_control.py",
    "paper/revision/generate_stratified_derangement_table.py",
    "paper/revision/test_stratified_derangement_control.py",
    "paper/results/protocol_v3_image_baseline_aggregate_public5.csv",
    "paper/results/protocol_v3_image_baseline_seed_metrics_public5.csv",
    "paper/results/protocol_v3_final_controls_20260718/atconv_forward_activation_audit_rerun_20260720.json",
    "paper/results/protocol_v3_public_grouping_audit_20260715/public_grouping_audit.csv",
    "paper/results/protocol_v3_public_grouping_audit_20260715/README.md",
    "paper/results/protocol_v3_boundary_20260715/boundary_aggregate_public5.csv",
    "paper/results/protocol_v3_boundary_20260715/boundary_seed_metrics_public5.csv",
    "paper/results/protocol_v3_qualitative_20260715/selection.csv",
    "paper/results/protocol_v3_semantic_control_20260718/selection.csv",
    "paper/results/medequiseg_factorial_public5_20260715/aggregate.csv",
    "paper/results/medequiseg_factorial_public5_20260715/seed_metrics.csv",
    "paper/results/medequiseg_factorial_public5_20260715/strict_semantic_control_statistics.csv",
    "paper/results/medequiseg_factorial_public5_20260715/strict_semantic_control_seed_metrics.csv",
    "paper/results/medequiseg_factorial_public5_20260715/manuscript_summary.md",
    "paper/results/medequiseg_base_contrast_20260718/base_contrast_statistics.csv",
    "paper/results/medequiseg_base_contrast_20260718/base_contrast_statistics.json",
    "paper/figures/medequiseg_architecture_draft_20260722.png",
    "paper/figures/medequiseg_effective_forward_fig2_v4.pdf",
    "paper/figures/medequiseg_effective_forward_fig2_v4.svg",
    "paper/figures/medequiseg_layer_details_supp_v1.pdf",
    "paper/figures/medequiseg_layer_details_supp_v1.svg",
    "paper/figures/medequiseg_operator_details_supp_v1.pdf",
    "paper/figures/medequiseg_operator_details_supp_v1.svg",
    "paper/figures/protocol_v3_qualitative_audit.pdf",
    "paper/figures/protocol_v3_semantic_control_audit.pdf",
    "paper/figures/strict_factorial_evidence.pdf",
    "paper/figures/generate_medequiseg_effective_forward_fig2.py",
    "paper/figures/generate_medequiseg_method_figures.py",
    "output/pdf/medequiseg_bmc_medical_imaging_manuscript_public5.pdf",
    "output/pdf/medequiseg_bmc_supplement_public5.pdf",
    "smoke_tests/protocol_v3/__init__.py",
    "smoke_tests/protocol_v3/core.py",
    "smoke_tests/protocol_v3/dataset_registry.yaml",
    "smoke_tests/protocol_v3/protocol_lock.yaml",
    "smoke_tests/protocol_v3/manifests/medclipseg_busi_full.csv",
    "smoke_tests/protocol_v3/manifests/medclipseg_clinicdb_full.csv",
    "smoke_tests/protocol_v3/manifests/medclipseg_busbra_full.csv",
    "smoke_tests/protocol_v3/manifests/medclipseg_brisc_full.csv",
    "smoke_tests/protocol_v3/manifests/medclipseg_covid19_full.csv",
    "smoke_tests/protocol_v3/manifests/medclipseg_brisc_duplicate_resolution.csv",
    "smoke_tests/run_protocol_v3.py",
    "smoke_tests/run_medequiseg_no_rewrite_matched.py",
    "smoke_tests/run_protocol_v3_covid_groupval_sensitivity.py",
    "smoke_tests/launch_medequiseg_no_rewrite_matched_busi.sh",
    "smoke_tests/build_protocol_v3_covid_groupval_sensitivity.py",
    "smoke_tests/lcaug_v2_direction.py",
    "smoke_tests/augmentation_plugins.py",
    "smoke_tests/causal_clip_recipe.py",
    "smoke_tests/causal_atconv_plugin.py",
    "smoke_tests/causal_text_encoder_plugins.py",
    "smoke_tests/biomedclip_offline.py",
    "smoke_tests/image_resize.py",
    "smoke_tests/text_encoders.py",
    "smoke_tests/build_causal_biomedclip_cache.py",
    "smoke_tests/build_text_encoder_ablation_cache.py",
    "smoke_tests/predict_protocol_v3.py",
    "smoke_tests/evaluate_predictions_v3.py",
    "smoke_tests/paper_metrics.py",
    "smoke_tests/train_baselines.py",
    "smoke_tests/summarize_protocol_v3_gate.py",
    "smoke_tests/protocol_v3/manifests/medclipseg_covid19_groupval_sensitivity.csv",
    "smoke_tests/protocol_v3/protocol_lock_covid_groupval_sensitivity.yaml",
)

ACTIVE_LATEX_PATHS = (
    "paper/latex/bmc_work_public5/bmc_attribution_table.tex",
    "paper/latex/bmc_work_public5/bmc_body.tex",
    "paper/latex/bmc_work_public5/bmc_boundary_tables.tex",
    "paper/latex/bmc_work_public5/bmc_complexity_table.tex",
    "paper/latex/bmc_work_public5/bmc_covid_cluster_table.tex",
    "paper/latex/bmc_work_public5/bmc_covid_groupval_table.tex",
    "paper/latex/bmc_work_public5/bmc_declarations.tex",
    "paper/latex/bmc_work_public5/bmc_factorial_ablation_detail_table.tex",
    "paper/latex/bmc_work_public5/bmc_factorial_ablation_table.tex",
    "paper/latex/bmc_work_public5/bmc_full_model_comparison_table.tex",
    "paper/latex/bmc_work_public5/bmc_mask_presence_table.tex",
    "paper/latex/bmc_work_public5/bmc_protocol_implementation_audit_table.tex",
    "paper/latex/bmc_work_public5/bmc_realized_trigger_table.tex",
    "paper/latex/bmc_work_public5/bmc_references.tex",
    "paper/latex/bmc_work_public5/bmc_stratified_derangement_table.tex",
    "paper/latex/bmc_work_public5/bmc_tables.tex",
    "paper/latex/bmc_work_public5/main_bmc.tex",
    "paper/latex/bmc_work_public5/main_bmc_supplement.tex",
    "paper/latex/bmc_work_public5/compile_bmc.sh",
    "paper/latex/bmc_work_public5/compile_bmc_supplement.sh",
    "paper/latex/bmc_work_public5/sn-jnl.cls",
    "paper/latex/bmc_work_public5/bst/sn-vancouver-num.bst",
)

GLOB_PATHS = (
    "paper/results/protocol_v3_complexity_20260715/*.csv",
    "paper/results/protocol_v3_complexity_20260715/*.json",
    "paper/results/protocol_v3_mask_presence_20260715/*.csv",
    "paper/results/protocol_v3_covid_subject_cluster_20260718/*",
    "paper/results/protocol_v3_covid_groupval_public5_20260725/*",
    "paper/results/protocol_v3_stratified_derangement_20260719/*.csv",
    "paper/results/protocol_v3_stratified_derangement_20260719/*.json",
    "paper/results/protocol_v3_stratified_derangement_20260719/control_maps/*.csv",
    "paper/results/protocol_v3_stratified_derangement_20260719/control_maps/*.json",
    "paper/results/medequiseg_no_rewrite_matched_20260725/*",
    "paper/results/medequiseg_complete_base_sensitivity_20260725/*",
)

FORBIDDEN_PATH_PARTS = (
    ".before_",
    "/backup/",
    "/backups/",
    "submission_backups",
    "strict_rewrite",
    "busi_hf",
    "private_manifest",
    "private_external",
    "__pycache__",
)
OBSOLETE_RUN_ID = "V3_ABL_" + "SHARED_NR"
RETAINED_RUN_IDS = {
    "V3_ABL_BASE",
    "V3_ABL_BIOMED",
    "V3_ABL_ATCONV",
    "V3_ABL_BIOMED_ATCONV",
    "V3_ABL_EQUIPROMPT",
}
FORBIDDEN_TEXT_PATTERNS = (
    OBSOLETE_RUN_ID,
    "Shared plan " + "(no rewrite)",
    "shared-plan " + "no-rewrite",
    "R11" + "NR",
    "six " + "configurations",
    "90 " + "training cells",
    "90-" + "cell ordered",
    "BUSI-HF",
    "busi_hf",
)
SECRET_PATTERN = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            f"output/submission/{ARCHIVE_PREFIX}_{RELEASE_DATE}.zip"
        ),
    )
    parser.add_argument(
        "--staging-copy",
        type=Path,
        default=None,
        help="Optional directory that receives the verified unpacked payload.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_paths(root: Path) -> list[Path]:
    selected = {root / relative for relative in (*EXACT_PATHS, *ACTIVE_LATEX_PATHS)}
    for pattern in GLOB_PATHS:
        selected.update(path for path in root.glob(pattern) if path.is_file())
    missing = sorted(path for path in selected if not path.is_file())
    if missing:
        raise FileNotFoundError(
            "Missing clean-release inputs:\n" + "\n".join(map(str, missing))
        )
    for path in selected:
        relative = "/" + path.relative_to(root).as_posix().lower()
        if any(part in relative for part in FORBIDDEN_PATH_PARTS):
            raise ValueError(f"Forbidden release path selected: {relative}")
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def sanitize_text(text: str, root: Path) -> str:
    replacements = {}
    candidates = (
        (str(root), "<PROJECT_ROOT>"),
        (str(Path.home()), "<USER_HOME>"),
    )
    for source, replacement in candidates:
        # Never replace a filesystem root: doing so would corrupt every path
        # separator in a Unix release or every drive-root occurrence on Windows.
        if source in {"", "/", "\\"} or re.fullmatch(r"[A-Za-z]:[\\/]", source):
            continue
        replacements[source] = replacement
    for source, replacement in sorted(
        replacements.items(), key=lambda item: len(item[0]), reverse=True
    ):
        text = text.replace(source, replacement)
    text = re.sub(r"[A-Za-z]:\\Users\\[^\\/\r\n\"']+", "<USER_HOME>", text)
    return SECRET_PATTERN.sub("<REDACTED_API_KEY>", text)


def retained_csv(text: str, relative: str) -> str:
    if relative not in {
        "paper/results/medequiseg_factorial_public5_20260715/aggregate.csv",
        "paper/results/medequiseg_factorial_public5_20260715/seed_metrics.csv",
    }:
        return text
    source = io.StringIO(text)
    rows = list(csv.DictReader(source))
    if not rows or "run_id" not in rows[0]:
        raise ValueError(f"Expected run_id column in {relative}")
    retained = [row for row in rows if row["run_id"] != OBSOLETE_RUN_ID]
    expected_per_run = 6 if relative.endswith("aggregate.csv") else 15
    expected = len(RETAINED_RUN_IDS) * expected_per_run
    if len(retained) != expected:
        raise ValueError(
            f"Unexpected retained row count for {relative}: {len(retained)} != {expected}"
        )
    counts = {run_id: 0 for run_id in RETAINED_RUN_IDS}
    for row in retained:
        if row["run_id"] not in counts:
            raise ValueError(f"Unexpected run_id in {relative}: {row['run_id']}")
        counts[row["run_id"]] += 1
    if any(count != expected_per_run for count in counts.values()):
        raise ValueError(f"Unbalanced retained run counts in {relative}: {counts}")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=list(rows[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(retained)
    return output.getvalue()


def retained_cluster_statistics(text: str, relative: str) -> str:
    prefix = "paper/results/protocol_v3_covid_subject_cluster_20260718/"
    if relative not in {
        prefix + "clustered_statistics.csv",
        prefix + "clustered_statistics.json",
    }:
        return text
    obsolete_comparison = "R11_minus_" + "R11" + "NR"
    if relative.endswith(".csv"):
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows or "comparison" not in rows[0]:
            raise ValueError(f"Expected comparison column in {relative}")
        retained = [row for row in rows if row["comparison"] != obsolete_comparison]
        if len(retained) != 4:
            raise ValueError(f"Unexpected retained comparison count in {relative}")
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(retained)
        return output.getvalue()

    payload = json.loads(text)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Expected rows list in {relative}")
    payload["rows"] = [
        row for row in rows if row.get("comparison") != obsolete_comparison
    ]
    if len(payload["rows"]) != 4:
        raise ValueError(f"Unexpected retained comparison count in {relative}")
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def retained_runner(text: str, relative: str) -> str:
    if relative != "smoke_tests/run_protocol_v3.py":
        return text
    public_block = '''PILOT_RUNS = (
    "V3_ABL_BASE",
    "V3_ABL_BIOMED",
    "V3_ABL_ATCONV",
    "V3_ABL_BIOMED_ATCONV",
    "V3_ABL_EQUIPROMPT",
)
CONFIRMATORY_RUNS = PILOT_RUNS
V3_RUNS = {
    "V3_ABL_BASE": {"recipe": "default", "augmentation": "lcaug_v2_hflip_dataset"},
    "V3_ABL_BIOMED": {"recipe": "biomed_lcaug", "augmentation": "lcaug_v2_hflip_dataset"},
    "V3_ABL_ATCONV": {"recipe": "default_atconv4", "augmentation": "lcaug_v2_hflip_dataset"},
    "V3_ABL_BIOMED_ATCONV": {"recipe": "biomed_lcaug_atconv4", "augmentation": "lcaug_v2_hflip_dataset"},
    "V3_ABL_EQUIPROMPT": {
        "recipe": "biomed_lcaug_v2_atconv4",
        "augmentation": "lcaug_v2_dynamic_shared_plan_dataset",
    },
}
PROMPT_CONTROLS'''
    updated, count = re.subn(
        r"PILOT_RUNS\s*=.*?\nPROMPT_CONTROLS",
        public_block,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError("Could not isolate the run registry in run_protocol_v3.py")
    return updated.replace(
        "Run Protocol V3 training, prompt controls, prediction, and evaluation.",
        "Run the five retained Public-5 configurations and prompt controls.",
    )


def copy_release_file(source: Path, destination: Path, root: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative = source.relative_to(root).as_posix()
    if source.suffix.lower() in TEXT_SUFFIXES:
        text = source.read_text(encoding="utf-8-sig", errors="strict")
        text = retained_csv(text, relative)
        text = retained_cluster_statistics(text, relative)
        text = retained_runner(text, relative)
        if source.suffix.lower() == ".svg":
            text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
        destination.write_text(sanitize_text(text, root), encoding="utf-8", newline="\n")
    else:
        shutil.copyfile(source, destination)


def verify_staging(staging: Path) -> None:
    issues: list[str] = []
    files = [path for path in staging.rglob("*") if path.is_file()]
    for path in files:
        relative = "/" + path.relative_to(staging).as_posix().lower()
        if any(part in relative for part in FORBIDDEN_PATH_PARTS):
            issues.append(f"forbidden path: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        if SECRET_PATTERN.search(text):
            issues.append(f"credential-like token: {relative}")
        if relative.endswith("build_protocol_v3_public5_release_package.py"):
            continue
        for pattern in FORBIDDEN_TEXT_PATTERNS:
            if pattern.lower() in text.lower():
                issues.append(f"obsolete text {pattern!r}: {relative}")
    expected_latex = {Path(path).name for path in ACTIVE_LATEX_PATHS}
    actual_latex = {
        path.name
        for path in (staging / "paper/latex/bmc_work_public5").glob("*")
        if path.is_file() and path.suffix in {".tex", ".sh", ".cls"}
    }
    expected_top = {
        Path(path).name
        for path in ACTIVE_LATEX_PATHS
        if "/bst/" not in path
    }
    if actual_latex != expected_top:
        issues.append(
            "active LaTeX source mismatch: "
            f"missing={sorted(expected_top - actual_latex)} "
            f"unexpected={sorted(actual_latex - expected_top)}"
        )
    if any("before" in name.lower() for name in expected_latex):
        issues.append("backup source entered the active LaTeX whitelist")
    if issues:
        raise ValueError(
            "Clean-release verification failed:\n" + "\n".join(sorted(set(issues)))
        )


def write_manifest(staging: Path, root: Path, sources: list[Path]) -> Path:
    manifest = staging / "release_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("release_path", "source_path", "bytes", "sha256"),
            lineterminator="\n",
        )
        writer.writeheader()
        for source in sources:
            relative = source.relative_to(root)
            released = staging / relative
            writer.writerow(
                {
                    "release_path": relative.as_posix(),
                    "source_path": relative.as_posix(),
                    "bytes": released.stat().st_size,
                    "sha256": sha256(released),
                }
            )
    return manifest


def verify_manifest(staging: Path) -> None:
    manifest = staging / "release_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    listed = set()
    for row in rows:
        path = staging / row["release_path"]
        listed.add(path.resolve())
        if not path.is_file():
            raise ValueError(f"Manifest path missing: {row['release_path']}")
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"Manifest byte mismatch: {row['release_path']}")
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Manifest hash mismatch: {row['release_path']}")
    payload = {
        path.resolve()
        for path in staging.rglob("*")
        if path.is_file() and path != manifest
    }
    if listed != payload:
        raise ValueError("Manifest coverage mismatch")


def write_deterministic_zip(staging: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(staging.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(ARCHIVE_PREFIX) / path.relative_to(staging)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 7, 25, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def verify_zip(output: Path, staging: Path) -> None:
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Corrupt archive member: {bad}")
        archived = {
            Path(name).relative_to(ARCHIVE_PREFIX).as_posix(): archive.read(name)
            for name in archive.namelist()
        }
    expected = {
        path.relative_to(staging).as_posix(): path.read_bytes()
        for path in staging.rglob("*")
        if path.is_file()
    }
    if archived != expected:
        raise ValueError("Archive payload differs from verified staging tree")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    sources = collect_paths(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="medequiseg_release_", dir=output.parent) as temp:
        staging = Path(temp)
        for source in sources:
            copy_release_file(source, staging / source.relative_to(root), root)
        verify_staging(staging)
        write_manifest(staging, root, sources)
        verify_manifest(staging)
        write_deterministic_zip(staging, output)
        verify_zip(output, staging)
        if args.staging_copy:
            staging_copy = (
                args.staging_copy
                if args.staging_copy.is_absolute()
                else root / args.staging_copy
            ).resolve()
            if staging_copy.exists():
                raise FileExistsError(f"Staging copy already exists: {staging_copy}")
            shutil.copytree(staging, staging_copy)
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{sha256(output)}  {output.name}\n", encoding="ascii")
    print(f"PASS files={len(sources)} bytes={output.stat().st_size} zip={output}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()

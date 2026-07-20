#!/usr/bin/env python3
"""Build and verify the public-dataset-only Protocol V3 release archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


TEXT_SUFFIXES = {
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

EXACT_PATHS = (
    "paper/protocol_v3_public5_release_readme.md",
    "paper/protocol_v3_public5_source_identity_20260720.md",
    "paper/submission_metadata_public5_template.yaml",
    "paper/protocol_v3_environment_record.md",
    "paper/reference_metadata_audit_20260715.md",
    "paper/protocol_v3_split_provenance_audit_20260715.md",
    "paper/analysis/audit_protocol_v3_public_grouping.py",
    "paper/analysis/build_public5_release_tables.py",
    "paper/analysis/build_protocol_v3_public5_release_package.py",
    "paper/analysis/finalize_public5_submission.py",
    "paper/analysis/benchmark_medclipseg_complexity.py",
    "paper/analysis/benchmark_protocol_v3_complexity.py",
    "paper/analysis/generate_complexity_table.py",
    "paper/analysis/generate_covid_subject_cluster_sensitivity.py",
    "paper/analysis/generate_protocol_v3_qualitative_overlay.py",
    "paper/analysis/generate_protocol_v3_semantic_control_overlay.py",
    "paper/analysis/generate_medequiseg_factorial_manuscript_assets.py",
    "paper/analysis/generate_medequiseg_base_contrast_statistics.py",
    "paper/analysis/generate_strict_factorial_figure.py",
    "paper/analysis/summarize_covid_groupval_sensitivity.py",
    "paper/analysis/test_finalize_public5_submission.py",
    "paper/revision/SHARED_PLAN_ATCONV_PROTOCOL_AUDIT_20260720.md",
    "paper/revision/STRATIFIED_DERANGEMENT_CONTROL_20260719.md",
    "paper/revision/analyze_stratified_derangement_control.py",
    "paper/revision/audit_atconv_forward_activation.py",
    "paper/revision/audit_protocol_v3_factorial_run_lineage.py",
    "paper/revision/audit_submission_package_public5_20260720.py",
    "paper/revision/audit_stratified_derangement_submission.py",
    "paper/revision/audit_submission_statistics_20260719.py",
    "paper/revision/build_stratified_derangement_control.py",
    "paper/revision/generate_stratified_derangement_table.py",
    "paper/revision/run_stratified_derangement_matrix.py",
    "paper/revision/test_stratified_derangement_control.py",
    "paper/tables/protocol_v3_public5_claim_evidence.csv",
    "paper/results/protocol_v3_image_baseline_aggregate_public5.csv",
    "paper/results/protocol_v3_image_baseline_seed_metrics_public5.csv",
    "paper/results/bmc_main_table_audit_20260718/audit.json",
    "paper/results/bmc_main_table_audit_20260718/README.md",
    "paper/results/bmc_statistical_table_audit_20260718/audit.json",
    "paper/results/bmc_statistical_table_audit_20260718/README.md",
    "paper/results/bmc_submission_audit_20260719/factorial_run_lineage_deep.json",
    "paper/results/bmc_submission_audit_20260719/submission_statistics_audit.json",
    "paper/results/bmc_submission_audit_20260720/stratified_derangement_submission_audit.json",
    "paper/results/protocol_v3_final_controls_20260718/atconv_forward_activation_audit_rerun_20260720.json",
    "paper/results/r11lr_seed123_raw/medclipseg_busi_seed123_true_summary.csv",
    "paper/results/r11lr_seed123_raw/medclipseg_clinicdb_seed123_true_summary.csv",
    "paper/results/protocol_v3_covid_subject_cluster_20260718/clustered_statistics.csv",
    "paper/results/protocol_v3_covid_subject_cluster_20260718/clustered_statistics.json",
    "paper/results/protocol_v3_covid_subject_cluster_20260718/README.md",
    "paper/results/protocol_v3_public_grouping_audit_20260715/public_grouping_audit.csv",
    "paper/results/protocol_v3_public_grouping_audit_20260715/README.md",
    "paper/results/protocol_v3_boundary_20260715/boundary_aggregate_public5.csv",
    "paper/results/protocol_v3_boundary_20260715/boundary_seed_metrics_public5.csv",
    "paper/results/protocol_v3_complexity_20260715/complexity_summary.csv",
    "paper/results/protocol_v3_qualitative_20260715/selection.csv",
    "paper/results/protocol_v3_semantic_control_20260718/selection.csv",
    "paper/results/medequiseg_factorial_public5_20260715/aggregate.csv",
    "paper/results/medequiseg_factorial_public5_20260715/seed_metrics.csv",
    "paper/results/medequiseg_factorial_public5_20260715/strict_rewrite_seed_metrics.csv",
    "paper/results/medequiseg_factorial_public5_20260715/strict_rewrite_statistics.csv",
    "paper/results/medequiseg_factorial_public5_20260715/strict_rewrite_statistics.json",
    "paper/results/medequiseg_factorial_public5_20260715/strict_semantic_control_statistics.csv",
    "paper/results/medequiseg_factorial_public5_20260715/strict_semantic_control_seed_metrics.csv",
    "paper/results/medequiseg_factorial_public5_20260715/manuscript_summary.md",
    "paper/results/medequiseg_base_contrast_20260718/base_contrast_statistics.csv",
    "paper/results/medequiseg_base_contrast_20260718/base_contrast_statistics.json",
    "paper/figures/medequiseg_architecture.pdf",
    "paper/figures/medequiseg_architecture.svg",
    "paper/figures/equiprompt_mechanism.pdf",
    "paper/figures/equiprompt_mechanism.svg",
    "paper/figures/protocol_v3_qualitative_audit.pdf",
    "paper/figures/protocol_v3_semantic_control_audit.pdf",
    "paper/figures/strict_factorial_evidence.pdf",
    "paper/figures/generate_medequiseg_method_figures.py",
    "output/pdf/medequiseg_bmc_medical_imaging_manuscript_public5.pdf",
    "output/pdf/medequiseg_bmc_supplement_public5.pdf",
    "smoke_tests/protocol_v3/__init__.py",
    "smoke_tests/protocol_v3/core.py",
    "smoke_tests/protocol_v3/dataset_registry.yaml",
    "smoke_tests/protocol_v3/protocol_lock.yaml",
    "smoke_tests/protocol_v3/protocol_lock_covid_groupval_sensitivity.yaml",
    "smoke_tests/protocol_v3/manifests/medclipseg_busi_full.csv",
    "smoke_tests/protocol_v3/manifests/medclipseg_clinicdb_full.csv",
    "smoke_tests/protocol_v3/manifests/medclipseg_busbra_full.csv",
    "smoke_tests/protocol_v3/manifests/medclipseg_brisc_full.csv",
    "smoke_tests/protocol_v3/manifests/medclipseg_covid19_full.csv",
    "smoke_tests/protocol_v3/manifests/medclipseg_covid19_groupval_sensitivity.csv",
    "smoke_tests/build_protocol_v3_covid_groupval_sensitivity.py",
    "smoke_tests/run_protocol_v3_covid_groupval_sensitivity.py",
    "smoke_tests/launch_protocol_v3_covid_groupval_sensitivity.sh",
    "smoke_tests/watch_protocol_v3_covid_groupval_sensitivity.sh",
    "smoke_tests/run_protocol_v3.py",
    "smoke_tests/run_protocol_v3_r11.py",
    "smoke_tests/run_protocol_v3_r11nr.py",
    "smoke_tests/run_protocol_v3_r11lr.py",
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
    "smoke_tests/run_medclipseg_protocol_v3.py",
    "smoke_tests/prepare_medclipseg_protocol_v3.py",
    "smoke_tests/evaluate_medclipseg_protocol_v3.py",
    "smoke_tests/run_protocol_v3_image_baseline_task.sh",
    "smoke_tests/run_ukan_image_aug_dataset.py",
    "smoke_tests/run_rolling_image_aug_dataset.py",
    "smoke_tests/predict_monai_protocol_v3.py",
    "smoke_tests/prepare_protocol_v3_nnunet.py",
    "smoke_tests/run_protocol_v3_rolling_canonical_task.sh",
    "smoke_tests/predict_rolling_protocol_v3.py",
)

GLOB_PATHS = (
    "paper/latex/bmc_work_public5/*.tex",
    "paper/latex/bmc_work_public5/compile_bmc*.sh",
    "paper/latex/bmc_work_public5/sn-jnl.cls",
    "paper/latex/bmc_work_public5/bst/sn-vancouver-num.bst",
    "paper/results/protocol_v3_complexity_20260715/*.csv",
    "paper/results/protocol_v3_complexity_20260715/*.json",
    "paper/results/protocol_v3_stratified_derangement_20260719/*.csv",
    "paper/results/protocol_v3_stratified_derangement_20260719/*.json",
    "paper/results/protocol_v3_stratified_derangement_20260719/control_maps/*.csv",
    "paper/results/protocol_v3_stratified_derangement_20260719/control_maps/*.json",
    "smoke_tests/protocol_v3/manifests/medclipseg_brisc_duplicate_resolution.csv",
    "smoke_tests/nnunet_trainers/*.py",
)

BLOCKED_RELEASE_PATHS = (
    "busi_hf_r8_811",
    "busi_hf_dev",
    "busi_hf_train50",
    "private_manifest",
    "paper/latex/bmc_work/",
)

BLOCKED_RELEASE_EXACT_PATHS = {
    "output/pdf/medequiseg_bmc_medical_imaging_manuscript.pdf",
    "output/pdf/medequiseg_bmc_supplement.pdf",
}

BLOCKED_TEXT_FRAGMENTS = (
    "/" + "tank2/",
    "/" + "home/",
    "C:\\Users\\",
    "datasets/" + "private_external",
    "smoke_tests/protocol_v3/manifests/" + "busi_hf",
    "API_KEY=",
)

PUBLIC_ONLY_TEXT_PREFIXES = (
    "paper/latex/bmc_work_public5/",
    "paper/figures/",
    "paper/results/",
    "paper/tables/",
)
PRIVATE_ARTIFACT_PATTERN = re.compile(
    r"BUSI[-_]HF|private\s+(?:cohort|development|analysis)|external\s+cohort",
    flags=re.IGNORECASE,
)

SECRET_PATTERN = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/submission/protocol_v3_public5_reproducibility_package_20260720.zip"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_paths(root: Path) -> list[Path]:
    selected = {root / relative for relative in EXACT_PATHS}
    for pattern in GLOB_PATHS:
        selected.update(root.glob(pattern))
    missing = [path for path in selected if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing release inputs:\n" + "\n".join(str(path) for path in missing))
    for path in selected:
        relative = path.relative_to(root).as_posix().lower()
        if any(fragment in relative for fragment in BLOCKED_RELEASE_PATHS):
            raise ValueError(f"Private manifest selected for release: {relative}")
        if relative in BLOCKED_RELEASE_EXACT_PATHS:
            raise ValueError(f"Full-cohort manuscript selected for Public-5 release: {relative}")
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def sanitize_text(text: str, root: Path) -> str:
    replacements = {
        str(root): "<PROJECT_ROOT>",
        str(Path(sys.prefix)): "<ENV_ROOT>",
        str(Path(sys.prefix).parent): "<ENV_ROOT>",
        str(Path.home()): "<USER_HOME>",
    }
    for source, replacement in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(source, replacement)
    text = re.sub(r"[A-Za-z]:\\Users\\[^\\/\r\n\"']+", "<USER_HOME>", text)
    private_manifest_prefix = "smoke_tests/protocol_v3/manifests/" + "busi_hf"
    text = re.sub(
        re.escape(private_manifest_prefix) + r"[^\s\"']*",
        "<PRIVATE_MANIFEST_NOT_RELEASED>",
        text,
    )
    private_lock_prefix = "smoke_tests/protocol_v3/protocol_lock_r8_" + "busi_hf"
    text = re.sub(
        re.escape(private_lock_prefix) + r"[^\s\"']*",
        "<PRIVATE_LOCK_NOT_RELEASED>",
        text,
    )
    private_data_prefix = "datasets/" + "private_external"
    text = text.replace(private_data_prefix, "<PRIVATE_DATA_NOT_RELEASED>")
    return SECRET_PATTERN.sub("<REDACTED_API_KEY>", text)


def copy_release_file(source: Path, destination: Path, root: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in TEXT_SUFFIXES:
        text = source.read_text(encoding="utf-8-sig", errors="strict")
        destination.write_text(sanitize_text(text, root), encoding="utf-8", newline="\n")
    else:
        shutil.copyfile(source, destination)


def verify_staging(staging: Path) -> None:
    issues: list[str] = []
    for path in staging.rglob("*"):
        if not path.is_file() or path.name == "release_manifest.csv":
            continue
        relative = path.relative_to(staging).as_posix().lower()
        if any(fragment in relative for fragment in BLOCKED_RELEASE_PATHS):
            issues.append(f"blocked path: {relative}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        # The verifier necessarily contains the blocked literals it searches for.
        if relative == "paper/analysis/build_protocol_v3_public5_release_package.py":
            continue
        text = path.read_text(encoding="utf-8", errors="strict")
        for fragment in BLOCKED_TEXT_FRAGMENTS:
            if fragment in text:
                issues.append(f"sensitive fragment {fragment!r}: {relative}")
        if SECRET_PATTERN.search(text):
            issues.append(f"API credential pattern: {relative}")
        if relative.startswith(PUBLIC_ONLY_TEXT_PREFIXES) and PRIVATE_ARTIFACT_PATTERN.search(text):
            issues.append(f"non-public result or manuscript reference: {relative}")
    if issues:
        raise ValueError("Public-5 release verification failed:\n" + "\n".join(sorted(set(issues))))


def write_manifest(staging: Path, root: Path, sources: list[Path]) -> Path:
    manifest = staging / "release_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("release_path", "source_path", "bytes", "sha256"),
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


def write_deterministic_zip(staging: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    prefix = "protocol_v3_public5_reproducibility_package"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(staging.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(prefix) / path.relative_to(staging)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 7, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent not in output.parents:
        raise ValueError("Unexpected output layout")

    sources = collect_paths(root)
    with tempfile.TemporaryDirectory(
        prefix="protocol_v3_public5_release_",
        dir=output.parent,
    ) as temporary:
        staging = Path(temporary)
        for source in sources:
            copy_release_file(source, staging / source.relative_to(root), root)
        write_manifest(staging, root, sources)
        verify_staging(staging)
        write_deterministic_zip(staging, output)

    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{sha256(output)}  {output.name}\n", encoding="ascii")
    print(f"PASS files={len(sources)} bytes={output.stat().st_size} zip={output}")
    print(f"sha256={sha256(output)}")


if __name__ == "__main__":
    main()

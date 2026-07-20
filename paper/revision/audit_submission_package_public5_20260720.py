#!/usr/bin/env python3
"""Audit the Public-5 BMC submission bundle without inventing author metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path


TEXT_SUFFIXES = {".csv", ".json", ".md", ".py", ".sh", ".svg", ".tex", ".txt", ".yaml", ".yml"}
ZIP_PREFIX = "protocol_v3_public5_reproducibility_package/"
REQUIRED_ARCHIVE_MEMBERS = {
    "paper/protocol_v3_public5_release_readme.md",
    "paper/protocol_v3_public5_source_identity_20260720.md",
    "paper/results/bmc_submission_audit_20260719/factorial_run_lineage_deep.json",
    "paper/results/bmc_submission_audit_20260719/submission_statistics_audit.json",
    "paper/results/bmc_submission_audit_20260720/stratified_derangement_submission_audit.json",
    "paper/results/protocol_v3_final_controls_20260718/atconv_forward_activation_audit_rerun_20260720.json",
    "paper/results/protocol_v3_stratified_derangement_20260719/casefirst_true_vs_stratified_derangement.csv",
    "paper/latex/bmc_work_public5/main_bmc.tex",
    "paper/latex/bmc_work_public5/main_bmc_supplement.tex",
    "paper/revision/audit_submission_package_public5_20260720.py",
    "output/pdf/medequiseg_bmc_medical_imaging_manuscript_public5.pdf",
    "output/pdf/medequiseg_bmc_supplement_public5.pdf",
}
BLOCKED_MEMBER_FRAGMENTS = {
    "busi_hf_r8_811",
    "busi_hf_dev",
    "busi_hf_train50",
    "private_manifest",
    "paper/latex/bmc_work/",
}
BLOCKED_EXACT_MEMBERS = {
    ZIP_PREFIX + "output/pdf/medequiseg_bmc_medical_imaging_manuscript.pdf",
    ZIP_PREFIX + "output/pdf/medequiseg_bmc_supplement.pdf",
}
BLOCKED_TEXT_FRAGMENTS = {
    "/" + "tank2/",
    "/" + "home/",
    "C:\\Users\\",
    "datasets/" + "private_external",
    "smoke_tests/protocol_v3/manifests/" + "busi_hf",
}
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
        "--archive",
        type=Path,
        default=Path("output/submission/protocol_v3_public5_reproducibility_package_20260720.zip"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/results/bmc_submission_audit_20260720/submission_package_public5_readiness.json"),
    )
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_check(checks: list[dict], check_id: str, passed: bool, detail: object) -> None:
    checks.append(
        {
            "check_id": check_id,
            "status": "PASS" if passed else "BLOCKED",
            "detail": detail,
        }
    )


def audit_archive(archive: Path, checks: list[dict]) -> dict:
    if not archive.is_file():
        add_check(checks, "archive_exists", False, str(archive))
        return {}

    archive_sha = sha256_file(archive)
    checksum_path = archive.with_suffix(archive.suffix + ".sha256")
    declared_sha = ""
    if checksum_path.is_file():
        declared_sha = checksum_path.read_text(encoding="ascii").split()[0].lower()
    add_check(
        checks,
        "archive_sha256",
        bool(declared_sha) and declared_sha == archive_sha,
        {"computed": archive_sha, "declared": declared_sha},
    )

    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        name_set = set(names)
        blocked_names = sorted(
            name
            for name in names
            if name in BLOCKED_EXACT_MEMBERS
            or any(fragment in name.lower() for fragment in BLOCKED_MEMBER_FRAGMENTS)
        )
        add_check(checks, "archive_private_member_paths", not blocked_names, blocked_names)

        missing = sorted(
            member
            for member in REQUIRED_ARCHIVE_MEMBERS
            if ZIP_PREFIX + member not in name_set
        )
        add_check(checks, "archive_required_members", not missing, missing)

        manifest_name = ZIP_PREFIX + "release_manifest.csv"
        manifest_rows: list[dict] = []
        if manifest_name in name_set:
            manifest_text = bundle.read(manifest_name).decode("utf-8-sig")
            manifest_rows = list(csv.DictReader(io.StringIO(manifest_text)))
        add_check(checks, "archive_manifest_present", bool(manifest_rows), len(manifest_rows))

        manifest_errors = []
        release_paths = set()
        for row in manifest_rows:
            release_path = row.get("release_path", "")
            member_name = ZIP_PREFIX + release_path
            if release_path in release_paths:
                manifest_errors.append(f"duplicate:{release_path}")
                continue
            release_paths.add(release_path)
            if member_name not in name_set:
                manifest_errors.append(f"missing:{release_path}")
                continue
            data = bundle.read(member_name)
            if str(len(data)) != row.get("bytes"):
                manifest_errors.append(f"bytes:{release_path}")
            if sha256_bytes(data) != row.get("sha256"):
                manifest_errors.append(f"sha256:{release_path}")
        add_check(checks, "archive_manifest_integrity", not manifest_errors, manifest_errors[:20])

        sensitive_hits = []
        for name in names:
            if not name.startswith(ZIP_PREFIX) or name.endswith("/"):
                continue
            relative = name[len(ZIP_PREFIX) :]
            if relative == "paper/analysis/build_protocol_v3_public5_release_package.py":
                continue
            if Path(relative).suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = bundle.read(name).decode("utf-8", errors="strict")
            for fragment in BLOCKED_TEXT_FRAGMENTS:
                if fragment in text:
                    sensitive_hits.append({"file": relative, "fragment": fragment})
            if SECRET_PATTERN.search(text):
                sensitive_hits.append({"file": relative, "fragment": "API credential pattern"})
            if relative.startswith(PUBLIC_ONLY_TEXT_PREFIXES) and PRIVATE_ARTIFACT_PATTERN.search(text):
                sensitive_hits.append({"file": relative, "fragment": "non-public result/manuscript reference"})
        add_check(checks, "archive_sensitive_text", not sensitive_hits, sensitive_hits[:20])

    return {
        "path": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": archive_sha,
        "manifest_entries": len(manifest_rows),
    }


def audit_manuscript(root: Path, checks: list[dict]) -> dict:
    main_pdf = root / "output/pdf/medequiseg_bmc_medical_imaging_manuscript_public5.pdf"
    supplement_pdf = root / "output/pdf/medequiseg_bmc_supplement_public5.pdf"
    add_check(
        checks,
        "main_pdf",
        main_pdf.is_file() and main_pdf.stat().st_size > 100_000,
        {"path": str(main_pdf), "bytes": main_pdf.stat().st_size if main_pdf.is_file() else 0},
    )
    add_check(
        checks,
        "supplement_pdf",
        supplement_pdf.is_file() and supplement_pdf.stat().st_size > 50_000,
        {"path": str(supplement_pdf), "bytes": supplement_pdf.stat().st_size if supplement_pdf.is_file() else 0},
    )

    source_dir = root / "paper/latex/bmc_work_public5"
    main_tex = source_dir / "main_bmc.tex"
    supplement_tex = source_dir / "main_bmc_supplement.tex"
    declarations_tex = source_dir / "bmc_declarations.tex"
    body_tex = source_dir / "bmc_body.tex"
    main_text = main_tex.read_text(encoding="utf-8")
    supplement_text = supplement_tex.read_text(encoding="utf-8")
    declarations = declarations_tex.read_text(encoding="utf-8")
    body = body_tex.read_text(encoding="utf-8")

    title_placeholders = sorted(
        marker
        for marker in (
            "\\fnm{First}",
            "\\fnm{Second}",
            "\\sur{Author}",
            "corresponding.author@institution.edu",
            "\\orgname{Institution}",
            "\\city{City}",
            "\\country{Country}",
            "% TODO BEFORE SUBMISSION",
        )
        if marker in main_text
    )
    add_check(checks, "title_page_metadata", not title_placeholders, title_placeholders)

    supplement_placeholders = sorted(
        marker
        for marker in (
            "\\fnm{First}",
            "\\fnm{Second}",
            "\\sur{Author}",
            "corresponding.author@institution.edu",
            "\\orgname{Institution}",
            "\\city{City}",
            "\\country{Country}",
            "% TODO BEFORE SUBMISSION",
        )
        if marker in supplement_text
    )
    add_check(
        checks,
        "supplement_title_page_metadata",
        not supplement_placeholders,
        supplement_placeholders,
    )

    declaration_markers = sorted(
        marker
        for marker in (
            "TO BE CONFIRMED",
            "TO BE COMPLETED",
            "must be confirmed before submission",
            "will be inserted here after deposit",
            "must obtain institutional confirmation",
            "must confirm the institutionally appropriate",
        )
        if marker in declarations
    )
    add_check(checks, "declarations_complete", not declaration_markers, declaration_markers)

    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(source_dir.glob("*.tex"))
    )
    private_reference_patterns = (
        r"BUSI[-_]HF",
        r"private\s+(?:cohort|development|analysis)",
        r"external\s+cohort",
    )
    private_references = sorted(
        pattern
        for pattern in private_reference_patterns
        if re.search(pattern, source_text, flags=re.IGNORECASE)
    )
    add_check(
        checks,
        "public5_source_scope",
        not private_references,
        private_references,
    )
    add_check(
        checks,
        "busi_hf_ethics_gate",
        not private_references,
        {"included": bool(private_references), "public_only_route": not private_references},
    )

    repository_pending = (
        "will be deposited" in declarations
        or "will be inserted here after deposit" in declarations
        or not re.search(r"https://[^\s{}]+", declarations)
    )
    add_check(checks, "repository_doi_or_url", not repository_pending, "pending" if repository_pending else "present")

    cover_letter = root / "paper/COVER_LETTER_PUBLIC5_DRAFT_20260720.md"
    cover_text = cover_letter.read_text(encoding="utf-8") if cover_letter.is_file() else ""
    cover_placeholders = sorted(
        set(re.findall(r"\[[^\]]*(?:INSERT|TO BE|TBD|TODO|AUTHOR|INSTITUTION|EMAIL|PHONE)[^\]]*\]", cover_text, re.I | re.S))
    )
    add_check(
        checks,
        "cover_letter_final",
        cover_letter.is_file() and not cover_placeholders,
        cover_placeholders if cover_letter.is_file() else "missing",
    )

    return {
        "main_pdf": str(main_pdf),
        "supplement_pdf": str(supplement_pdf),
        "busi_hf_included": bool(private_references),
    }


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    archive = args.archive if args.archive.is_absolute() else root / args.archive
    output = args.output if args.output.is_absolute() else root / args.output

    checks: list[dict] = []
    manuscript = audit_manuscript(root, checks)
    archive_meta = audit_archive(archive.resolve(), checks)
    blockers = [check for check in checks if check["status"] == "BLOCKED"]
    payload = {
        "status": "READY_FOR_UPLOAD" if not blockers else "NO_GO_FOR_UPLOAD",
        "scientific_experiment_status": "FROZEN",
        "project_root": str(root),
        "manuscript": manuscript,
        "archive": archive_meta,
        "checks_passed": sum(check["status"] == "PASS" for check in checks),
        "checks_blocked": len(blockers),
        "blocking_check_ids": [check["check_id"] for check in blockers],
        "checks": checks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "checks_passed", "checks_blocked", "blocking_check_ids")}, indent=2))
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

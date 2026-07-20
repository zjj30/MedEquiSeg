#!/usr/bin/env python3
"""Validate author facts and deterministically finalize the Public-5 submission."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml


TITLE = (
    "MedEquiSeg: Shared Augmentation and Privileged-Prompt Reliability "
    "in Multimodal Medical Image Segmentation"
)
AUTHOR_BEGIN = "% SUBMISSION_METADATA_BEGIN"
AUTHOR_END = "% SUBMISSION_METADATA_END"
PLACEHOLDER_PATTERN = re.compile(
    r"(?:TO BE CONFIRMED|TO BE COMPLETED|INSERT\b|\bTBD\b|\bTODO\b|"
    r"First Author|Second Author|corresponding\.author@institution\.edu|"
    r"\[\s*(?:DATE|AUTHOR|INSTITUTION|DEPARTMENT|CITY|COUNTRY|EMAIL|PHONE|"
    r"ETHICS|WAIVER|FUNDING|CONFLICT|REPOSITORY|DOI|URL|DEGREE)[^\]]*\])",
    flags=re.IGNORECASE | re.DOTALL,
)
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
ORCID_PATTERN = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
URL_PATTERN = re.compile(r"^https://[^\s]+$", flags=re.IGNORECASE)
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--apply", action="store_true", help="Write final files and run the full build/audit pipeline.")
    parser.add_argument("--skip-pipeline", action="store_true", help="With --apply, write files without compiling or auditing.")
    parser.add_argument("--preview-dir", type=Path, default=Path("tmp/public5_submission_preview"))
    parser.add_argument("--compile-preview", action="store_true", help="Compile rendered files in an isolated temporary project tree.")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def required_text(value: Any, label: str, *, minimum: int = 1) -> str:
    text = str(value or "").strip()
    if len(text) < minimum:
        raise ValueError(f"{label} is required (minimum {minimum} characters)")
    if PLACEHOLDER_PATTERN.search(text) or "<" in text or ">" in text:
        raise ValueError(f"{label} still contains a placeholder")
    return text


def require_true(value: Any, label: str) -> None:
    if value is not True:
        raise ValueError(f"{label} must be explicitly set to true by the corresponding author")


def normalize_id(value: Any, label: str) -> str:
    identifier = str(value or "").strip()
    if not identifier or not re.fullmatch(r"[A-Za-z0-9]+", identifier):
        raise ValueError(f"{label} must be a non-empty alphanumeric identifier")
    return identifier


def valid_orcid(value: str) -> bool:
    if not ORCID_PATTERN.fullmatch(value):
        return False
    compact = value.replace("-", "")
    total = 0
    for character in compact[:15]:
        total = (total + int(character)) * 2
    check_value = (12 - (total % 11)) % 11
    expected = "X" if check_value == 10 else str(check_value)
    return compact[-1] == expected


def load_and_validate(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    submission = payload.get("submission") or {}
    authors = payload.get("authors") or []
    affiliations = payload.get("affiliations") or []
    declarations = payload.get("declarations") or {}

    if not isinstance(authors, list) or not authors:
        raise ValueError("authors must contain at least one author")
    if not isinstance(affiliations, list) or not affiliations:
        raise ValueError("affiliations must contain at least one affiliation")

    submission_date = required_text(submission.get("date"), "submission.date")
    if not ISO_DATE_PATTERN.fullmatch(submission_date):
        raise ValueError("submission.date must use YYYY-MM-DD")
    date.fromisoformat(submission_date)
    journal = required_text(submission.get("journal"), "submission.journal")
    if journal != "BMC Medical Imaging":
        raise ValueError("submission.journal must remain BMC Medical Imaging for this package")
    repository_url = required_text(submission.get("repository_url"), "submission.repository_url")
    if not URL_PATTERN.fullmatch(repository_url):
        raise ValueError("submission.repository_url must be a complete HTTPS URL")
    require_true(submission.get("repository_is_permanent"), "submission.repository_is_permanent")
    require_true(submission.get("originality_confirmed"), "submission.originality_confirmed")
    require_true(
        submission.get("not_under_consideration_elsewhere"),
        "submission.not_under_consideration_elsewhere",
    )
    require_true(submission.get("all_authors_approved"), "submission.all_authors_approved")
    equal_contribution_statement = str(submission.get("equal_contribution_statement") or "").strip()

    normalized_affiliations: list[dict[str, str]] = []
    affiliation_ids: set[str] = set()
    for index, item in enumerate(affiliations, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"affiliations[{index}] must be a mapping")
        identifier = normalize_id(item.get("id"), f"affiliations[{index}].id")
        if identifier in affiliation_ids:
            raise ValueError(f"duplicate affiliation id: {identifier}")
        affiliation_ids.add(identifier)
        normalized_affiliations.append(
            {
                "id": identifier,
                "department": str(item.get("department") or "").strip(),
                "institution": required_text(item.get("institution"), f"affiliations[{index}].institution"),
                "city": required_text(item.get("city"), f"affiliations[{index}].city"),
                "postal_code": str(item.get("postal_code") or "").strip(),
                "country": required_text(item.get("country"), f"affiliations[{index}].country"),
            }
        )

    normalized_authors: list[dict[str, Any]] = []
    initials_seen: set[str] = set()
    corresponding_count = 0
    equal_contribution_count = 0
    for index, item in enumerate(authors, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"authors[{index}] must be a mapping")
        given = required_text(item.get("given_names"), f"authors[{index}].given_names")
        family = required_text(item.get("family_name"), f"authors[{index}].family_name")
        initials = required_text(item.get("initials"), f"authors[{index}].initials").upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9-]{0,7}", initials):
            raise ValueError(f"authors[{index}].initials must be 1-8 uppercase letters/digits")
        if initials in initials_seen:
            raise ValueError(f"duplicate author initials: {initials}")
        initials_seen.add(initials)
        corresponding = item.get("corresponding") is True
        corresponding_count += int(corresponding)
        equal_contribution = item.get("equal_contribution") is True
        equal_contribution_count += int(equal_contribution)
        email = str(item.get("email") or "").strip()
        if corresponding and not EMAIL_PATTERN.fullmatch(email):
            raise ValueError(f"authors[{index}].email must be valid for the corresponding author")
        if email and not EMAIL_PATTERN.fullmatch(email):
            raise ValueError(f"authors[{index}].email is not valid")
        refs_raw = item.get("affiliations") or []
        if not isinstance(refs_raw, list) or not refs_raw:
            raise ValueError(f"authors[{index}].affiliations must be a non-empty list")
        refs = [normalize_id(value, f"authors[{index}].affiliations") for value in refs_raw]
        missing = sorted(set(refs) - affiliation_ids)
        if missing:
            raise ValueError(f"authors[{index}] references unknown affiliations: {missing}")
        orcid = str(item.get("orcid") or "").strip()
        if orcid and not valid_orcid(orcid):
            raise ValueError(f"authors[{index}].orcid must be a checksum-valid ORCID")
        normalized_authors.append(
            {
                "given_names": given,
                "family_name": family,
                "initials": initials,
                "degree": str(item.get("degree") or "").strip(),
                "email": email,
                "phone": str(item.get("phone") or "").strip(),
                "corresponding": corresponding,
                "equal_contribution": equal_contribution,
                "affiliations": refs,
                "orcid": orcid,
            }
        )
    if corresponding_count < 1:
        raise ValueError("at least one corresponding author is required")
    if equal_contribution_count == 1:
        raise ValueError("equal contribution must identify at least two authors")
    if equal_contribution_count >= 2:
        equal_contribution_statement = required_text(
            equal_contribution_statement,
            "submission.equal_contribution_statement",
            minimum=20,
        )
    elif equal_contribution_statement:
        raise ValueError(
            "submission.equal_contribution_statement is set but no authors are marked equal_contribution"
        )

    normalized_declarations = {
        "ethics_approval": required_text(
            declarations.get("ethics_approval"), "declarations.ethics_approval", minimum=40
        ),
        "consent_for_publication": required_text(
            declarations.get("consent_for_publication"),
            "declarations.consent_for_publication",
            minimum=15,
        ),
        "competing_interests": required_text(
            declarations.get("competing_interests"), "declarations.competing_interests", minimum=15
        ),
        "funding": required_text(declarations.get("funding"), "declarations.funding", minimum=10),
        "contributions": required_text(
            declarations.get("contributions"), "declarations.contributions", minimum=40
        ),
        "acknowledgements": required_text(
            declarations.get("acknowledgements"), "declarations.acknowledgements", minimum=5
        ),
        "ai_use": required_text(declarations.get("ai_use"), "declarations.ai_use", minimum=40),
    }
    contributions = normalized_declarations["contributions"]
    missing_initials = sorted(
        initials for initials in initials_seen if not re.search(rf"\b{re.escape(initials)}\b", contributions)
    )
    if missing_initials:
        raise ValueError(f"declarations.contributions omits author initials: {missing_initials}")
    if not re.search(r"ethic|waiver|exempt|approval", normalized_declarations["ethics_approval"], re.I):
        raise ValueError("declarations.ethics_approval must state the approved ethics/waiver disposition")

    return {
        "submission": {
            "date": submission_date,
            "journal": journal,
            "repository_url": repository_url,
            "equal_contribution_statement": equal_contribution_statement,
        },
        "authors": normalized_authors,
        "affiliations": normalized_affiliations,
        "declarations": normalized_declarations,
    }


LATEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(value: str) -> str:
    return "".join(LATEX_REPLACEMENTS.get(character, character) for character in value)


def wrapped_latex(value: str) -> str:
    return textwrap.fill(latex_escape(value), width=92, break_long_words=False, break_on_hyphens=False)


def render_author_block(metadata: dict[str, Any]) -> str:
    authors = metadata["authors"]
    affiliations = metadata["affiliations"]
    corresponding_authors = [author for author in authors if author["corresponding"]]
    starred_affiliations = {author["affiliations"][0] for author in corresponding_authors}
    equal_statement = metadata["submission"]["equal_contribution_statement"]
    lines = [AUTHOR_BEGIN]
    for author in authors:
        command = r"\author*" if author["corresponding"] else r"\author"
        refs = ",".join(author["affiliations"])
        line = (
            f"{command}[{refs}]{{\\fnm{{{latex_escape(author['given_names'])}}} "
            f"\\sur{{{latex_escape(author['family_name'])}}}}}"
        )
        if author["email"]:
            line += f"\\email{{{latex_escape(author['email'])}}}"
        lines.append(line)
        if author["equal_contribution"]:
            lines.append(f"\\equalcont{{{latex_escape(equal_statement)}}}")
    lines.append("")
    for affiliation in affiliations:
        command = r"\affil*" if affiliation["id"] in starred_affiliations else r"\affil"
        department = (
            f"\\orgdiv{{{latex_escape(affiliation['department'])}}}, "
            if affiliation["department"]
            else ""
        )
        postal_code = (
            f", \\postcode{{{latex_escape(affiliation['postal_code'])}}}"
            if affiliation["postal_code"]
            else ""
        )
        lines.append(
            f"{command}[{affiliation['id']}]{{{department}"
            f"\\orgname{{{latex_escape(affiliation['institution'])}}}, "
            f"\\orgaddress{{\\city{{{latex_escape(affiliation['city'])}}}{postal_code}, "
            f"\\country{{{latex_escape(affiliation['country'])}}}}}}}"
        )
    lines.append(AUTHOR_END)
    return "\n".join(lines)


def render_main_tex(current: str, metadata: dict[str, Any]) -> str:
    block = render_author_block(metadata)
    if AUTHOR_BEGIN in current and AUTHOR_END in current:
        rendered, count = re.subn(
            rf"{re.escape(AUTHOR_BEGIN)}.*?{re.escape(AUTHOR_END)}",
            lambda _match: block,
            current,
            count=1,
            flags=re.DOTALL,
        )
        if count != 1:
            raise ValueError("Could not replace the existing submission metadata block")
        return rendered
    start = current.find("% TODO BEFORE SUBMISSION:")
    end = current.find("\n\\abstract{", start)
    if start < 0 or end < 0:
        raise ValueError("Could not locate the placeholder author block in main_bmc.tex")
    return current[:start] + block + "\n" + current[end:]


def render_declarations(metadata: dict[str, Any]) -> str:
    declarations = metadata["declarations"]
    repository_url = metadata["submission"]["repository_url"]
    availability = (
        "The five public datasets are available from their cited sources. "
        "The audited Public-5 Protocol V3 source-and-results package is permanently available at "
        f"\\url{{{repository_url}}}. The archive contains the protocol lock and registry, sanitized "
        "public split manifests with their case-level text prompts, model and evaluation code, "
        "three-seed result tables, statistical outputs, figures, and manuscript source. Medical "
        "images, masks, third-party checkpoints, non-public cohort artifacts, and raw training logs "
        "are not redistributed."
    )
    sections = [
        ("Ethics approval and consent to participate", declarations["ethics_approval"], False),
        ("Consent for publication", declarations["consent_for_publication"], False),
        (
            "Availability of data and materials",
            availability,
            True,
        ),
        ("Competing interests", declarations["competing_interests"], False),
        ("Funding", declarations["funding"], False),
        ("Authors' contributions", declarations["contributions"], False),
        ("Acknowledgements", declarations["acknowledgements"], False),
        ("Use of generative artificial intelligence", declarations["ai_use"], False),
    ]
    lines = [r"\section*{Declarations}", ""]
    for heading, body, is_raw_latex in sections:
        rendered_body = textwrap.fill(body, width=92, break_long_words=False) if is_raw_latex else wrapped_latex(body)
        lines.extend((f"\\subsection*{{{heading}}}", rendered_body, ""))
    return "\n".join(lines).rstrip() + "\n"


def render_cover_letter(metadata: dict[str, Any]) -> str:
    submission = metadata["submission"]
    declarations = metadata["declarations"]
    corresponding_authors = [author for author in metadata["authors"] if author["corresponding"]]
    affiliations = {item["id"]: item for item in metadata["affiliations"]}
    rendered_date = date.fromisoformat(submission["date"]).strftime("%B %d, %Y").replace(" 0", " ")
    opening = textwrap.fill(
        f"Please consider our manuscript, **“{TITLE},”** for publication in "
        f"*{submission['journal']}*.",
        width=92,
        break_long_words=False,
    )
    author_confirmation = textwrap.fill(
        "All datasets analyzed in this submission are publicly released and described in the "
        f"manuscript. {declarations['ethics_approval']} The corresponding authors confirm that "
        "the manuscript is original, is not under consideration elsewhere, and has been approved "
        f"by all authors. {declarations['competing_interests']} {declarations['funding']}",
        width=92,
        break_long_words=False,
    )
    signatures = []
    for corresponding in corresponding_authors:
        primary = affiliations[corresponding["affiliations"][0]]
        degree = f", {corresponding['degree']}" if corresponding["degree"] else ""
        department = f"{primary['department']}, " if primary["department"] else ""
        location = f"{primary['city']}"
        if primary["postal_code"]:
            location += f" {primary['postal_code']}"
        location += f", {primary['country']}"
        phone_line = f"  \n{corresponding['phone']}" if corresponding["phone"] else ""
        orcid_line = f"  \nORCID: {corresponding['orcid']}" if corresponding["orcid"] else ""
        signatures.append(
            f"{corresponding['given_names']} {corresponding['family_name']}{degree}  \n"
            f"{department}{primary['institution']}  \n"
            f"{location}  \n"
            f"{corresponding['email']}{phone_line}{orcid_line}"
        )
    signature_block = "\n\n".join(signatures)
    availability = textwrap.fill(
        "The manuscript source, compiled supplement, analysis code, sanitized manifests, and "
        f"audit artifacts are permanently available at {submission['repository_url']}.",
        width=92,
        break_long_words=False,
    )
    return f"""# Cover Letter - Public-5 Submission Route

{rendered_date}

Editor-in-Chief  
{submission['journal']}

Dear Editor,

{opening}

The study evaluates a multimodal segmentation configuration across five public
medical-image datasets using fixed train, validation, and test roles, three
training seeds, native-resolution evaluation, and a 90-cell ordered ablation.
The central result is attribution-focused: most aggregate improvement is
associated with a shared augmentation policy, while the data do not establish
an additional macro-accuracy benefit from prompt rewriting or the BioMedCLIP-ATConv
stack beyond that policy. Frozen-checkpoint semantic controls show why performance
with target-derived prompts should not be interpreted as clinical-report understanding.

The manuscript combines a reproduced multi-dataset comparison with explicit
protocol, boundary-metric, semantic-control, and implementation audits. It
discloses that the shared sampler is a follow-up amendment on the fixed
data/evaluation protocol, that only two registered ATConv replacements enter the
effective forward graph, and that the release is an audited implementation rather
than a bitwise reconstruction of the original training source snapshot.

{author_confirmation}

{availability}

Thank you for your consideration.

Sincerely,

{signature_block}
"""


def assert_no_placeholders(label: str, content: str) -> None:
    match = PLACEHOLDER_PATTERN.search(content)
    if match:
        raise ValueError(f"Rendered {label} still contains placeholder text: {match.group(0)!r}")


def render_outputs(root: Path, metadata: dict[str, Any]) -> dict[Path, str]:
    main_path = root / "paper/latex/bmc_work_public5/main_bmc.tex"
    supplement_path = root / "paper/latex/bmc_work_public5/main_bmc_supplement.tex"
    declarations_path = root / "paper/latex/bmc_work_public5/bmc_declarations.tex"
    cover_path = root / "paper/COVER_LETTER_PUBLIC5_DRAFT_20260720.md"
    current_main = main_path.read_text(encoding="utf-8")
    current_supplement = supplement_path.read_text(encoding="utf-8")
    outputs = {
        main_path: render_main_tex(current_main, metadata),
        supplement_path: render_main_tex(current_supplement, metadata),
        declarations_path: render_declarations(metadata),
        cover_path: render_cover_letter(metadata),
    }
    for path, content in outputs.items():
        assert_no_placeholders(path.name, content)
    return outputs


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_previews(root: Path, preview_dir: Path, outputs: dict[Path, str]) -> Path:
    destination = preview_dir if preview_dir.is_absolute() else root / preview_dir
    destination.mkdir(parents=True, exist_ok=True)
    for source, content in outputs.items():
        atomic_write(destination / source.name, content)
    return destination


def apply_outputs(root: Path, outputs: dict[Path, str]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = root / "paper/submission_backups_public5" / stamp
    for path in outputs:
        target = backup / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path, content in outputs.items():
        atomic_write(path, content)
    return backup


def compile_preview(root: Path, outputs: dict[Path, str]) -> None:
    temp_parent = root / "tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="public5_finalizer_compile_", dir=temp_parent) as temporary:
        preview_root = Path(temporary)
        shutil.copytree(
            root / "paper/latex/bmc_work_public5",
            preview_root / "paper/latex/bmc_work_public5",
        )
        shutil.copytree(root / "paper/figures", preview_root / "paper/figures")
        for source, content in outputs.items():
            target = preview_root / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(target, content)
        run_command(["bash", "paper/latex/bmc_work_public5/compile_bmc.sh"], preview_root)
        run_command(["bash", "paper/latex/bmc_work_public5/compile_bmc_supplement.sh"], preview_root)
        logs = (
            preview_root / "paper/latex/build_bmc_public5/main_bmc.log",
            preview_root / "paper/latex/build_bmc_supplement_public5/main_bmc_supplement.log",
        )
        bad_pattern = re.compile(r"Undefined|Overfull|^!", flags=re.MULTILINE)
        for log_path in logs:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            match = bad_pattern.search(text)
            if match:
                raise RuntimeError(f"Preview compile log failed at {log_path.name}: {match.group(0)}")
        print("PREVIEW_COMPILE_PASS")


def run_command(command: list[str], root: Path) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def run_final_pipeline(root: Path) -> dict[str, Any]:
    python = sys.executable
    run_command(["bash", "paper/latex/bmc_work_public5/compile_bmc.sh"], root)
    run_command(["bash", "paper/latex/bmc_work_public5/compile_bmc_supplement.sh"], root)
    run_command([python, "paper/analysis/build_public5_release_tables.py"], root)
    run_command(
        [
            python,
            "paper/analysis/build_protocol_v3_public5_release_package.py",
            "--project-root",
            str(root),
            "--output",
            str(root / "output/submission/protocol_v3_public5_reproducibility_package_20260720.zip"),
        ],
        root,
    )
    audit_path = root / "paper/results/bmc_submission_audit_20260720/submission_package_public5_readiness.json"
    run_command(
        [
            python,
            "paper/revision/audit_submission_package_public5_20260720.py",
            "--project-root",
            str(root),
            "--archive",
            str(root / "output/submission/protocol_v3_public5_reproducibility_package_20260720.zip"),
            "--output",
            str(audit_path),
        ],
        root,
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if payload.get("status") != "READY_FOR_UPLOAD":
        raise RuntimeError(
            "Final readiness audit did not pass: " + ", ".join(payload.get("blocking_check_ids") or [])
        )
    return payload


def synthetic_metadata() -> dict[str, Any]:
    return {
        "submission": {
            "date": "2026-07-20",
            "journal": "BMC Medical Imaging",
            "repository_url": "https://doi.org/10.0000/example",
        },
        "authors": [
            {
                "given_names": "Ada",
                "family_name": "Example",
                "initials": "AE",
                "degree": "PhD",
                "email": "ada@example.edu",
                "phone": "",
                "corresponding": True,
                "affiliations": ["1"],
                "orcid": "0000-0002-1825-0097",
            },
            {
                "given_names": "Bo",
                "family_name": "Example",
                "initials": "BE",
                "degree": "",
                "email": "",
                "phone": "",
                "corresponding": False,
                "affiliations": ["1"],
                "orcid": "",
            },
        ],
        "affiliations": [
            {"id": "1", "department": "Department", "institution": "University", "city": "City", "country": "Country"}
        ],
        "declarations": {
            "ethics_approval": "The institutional ethics committee confirmed that secondary analysis of these public de-identified datasets was exempt from additional approval.",
            "consent_for_publication": "Not applicable. No identifiable participant information is included.",
            "competing_interests": "The authors declare that they have no competing interests.",
            "funding": "This research received no specific grant from any funding agency.",
            "contributions": "AE conceived the study and drafted the manuscript; BE validated the analysis; AE and BE approved the final manuscript.",
            "acknowledgements": "Not applicable.",
            "ai_use": "OpenAI Codex was used for code-assisted checks and language editing. All authors reviewed and take responsibility for the final work.",
        },
    }


def run_self_test() -> None:
    metadata = synthetic_metadata()
    placeholder_main = "% TODO BEFORE SUBMISSION: replace placeholders.\n\\author*[1]{x}\n\n\\abstract{Text}\n"
    main = render_main_tex(placeholder_main, metadata)
    declarations = render_declarations(metadata)
    cover = render_cover_letter(metadata)
    for label, content in (("main", main), ("declarations", declarations), ("cover", cover)):
        assert_no_placeholders(label, content)
    if main.count(AUTHOR_BEGIN) != 1 or main.count(AUTHOR_END) != 1:
        raise AssertionError("Author metadata markers are not stable")
    if "https://doi.org/10.0000/example" not in declarations or "ada@example.edu" not in main:
        raise AssertionError("Rendered output omitted required metadata")
    print("SELF_TEST_PASS")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.metadata is None:
        raise SystemExit("--metadata is required unless --self-test is used")
    root = args.project_root.resolve()
    metadata_path = args.metadata if args.metadata.is_absolute() else root / args.metadata
    metadata = load_and_validate(metadata_path)
    outputs = render_outputs(root, metadata)
    if not args.apply:
        preview = write_previews(root, args.preview_dir, outputs)
        if args.compile_preview:
            compile_preview(root, outputs)
        print(f"VALIDATION_PASS preview={preview}")
        return
    backup = apply_outputs(root, outputs)
    print(f"APPLY_PASS backup={backup}")
    if args.skip_pipeline:
        print("PIPELINE_SKIPPED")
        return
    result = run_final_pipeline(root)
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks_passed": result["checks_passed"],
                "checks_blocked": result["checks_blocked"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

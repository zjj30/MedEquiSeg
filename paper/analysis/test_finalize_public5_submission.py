#!/usr/bin/env python3

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from finalize_public5_submission import (
    AUTHOR_BEGIN,
    AUTHOR_END,
    assert_no_placeholders,
    load_and_validate,
    render_cover_letter,
    render_declarations,
    render_main_tex,
    render_outputs,
)


def valid_payload() -> dict:
    return {
        "submission": {
            "date": "2026-07-20",
            "journal": "BMC Medical Imaging",
            "repository_url": "https://repository.example.edu/record/123",
            "repository_is_permanent": True,
            "originality_confirmed": True,
            "not_under_consideration_elsewhere": True,
            "all_authors_approved": True,
            "equal_contribution_statement": "Ada Example and Bo Example contributed equally to this work.",
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
                "equal_contribution": True,
                "affiliations": [1],
                "orcid": "0000-0002-1825-0097",
            },
            {
                "given_names": "Bo",
                "family_name": "Example",
                "initials": "BE",
                "degree": "",
                "email": "bo@example.edu",
                "phone": "",
                "corresponding": True,
                "equal_contribution": True,
                "affiliations": [1],
                "orcid": "",
            },
        ],
        "affiliations": [
            {
                "id": 1,
                "department": "Department of Imaging",
                "institution": "Example University",
                "city": "Example City",
                "postal_code": "12345",
                "country": "Example Country",
            }
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


def validate(payload: dict) -> dict:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "metadata.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return load_and_validate(path)


class FinalizePublic5SubmissionTests(unittest.TestCase):
    def test_valid_metadata_and_rendering(self) -> None:
        metadata = validate(valid_payload())
        current = "% TODO BEFORE SUBMISSION: replace.\n\\author*[1]{placeholder}\n\n\\abstract{Body}\n"
        main = render_main_tex(current, metadata)
        declarations = render_declarations(metadata)
        cover = render_cover_letter(metadata)
        for label, content in (("main", main), ("declarations", declarations), ("cover", cover)):
            assert_no_placeholders(label, content)
        self.assertIn("ada@example.edu", main)
        self.assertIn("bo@example.edu", main)
        self.assertEqual(main.count(r"\author*"), 2)
        self.assertEqual(main.count(r"\equalcont"), 2)
        self.assertIn("Ada Example", cover)
        self.assertIn("Bo Example", cover)
        self.assertIn(r"\url{https://repository.example.edu/record/123}", declarations)
        self.assertNotIn(r"\textbackslash{}url", declarations)
        self.assertIn("https://repository.example.edu/record/123", cover)

    def test_author_block_rerender_is_idempotent(self) -> None:
        metadata = validate(valid_payload())
        current = "% TODO BEFORE SUBMISSION: replace.\n\\author*[1]{placeholder}\n\n\\abstract{Body}\n"
        first = render_main_tex(current, metadata)
        second = render_main_tex(first, metadata)
        self.assertEqual(first, second)
        self.assertEqual(first.count(AUTHOR_BEGIN), 1)
        self.assertEqual(first.count(AUTHOR_END), 1)

    def test_render_outputs_updates_main_and_supplement(self) -> None:
        metadata = validate(valid_payload())
        template = (
            f"{AUTHOR_BEGIN}\n"
            r"\author*[1]{\fnm{First} \sur{Author}}" "\n"
            f"{AUTHOR_END}\n"
            r"\abstract{Body}" "\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            latex_dir = root / "paper/latex/bmc_work_public5"
            latex_dir.mkdir(parents=True)
            main_path = latex_dir / "main_bmc.tex"
            supplement_path = latex_dir / "main_bmc_supplement.tex"
            main_path.write_text(template, encoding="utf-8")
            supplement_path.write_text(template, encoding="utf-8")
            outputs = render_outputs(root, metadata)
        self.assertIn(main_path, outputs)
        self.assertIn(supplement_path, outputs)
        self.assertNotIn(r"\fnm{First}", outputs[supplement_path])
        self.assertIn("ada@example.edu", outputs[supplement_path])

    def test_requires_at_least_one_corresponding_author(self) -> None:
        payload = valid_payload()
        for author in payload["authors"]:
            author["corresponding"] = False
        with self.assertRaisesRegex(ValueError, "at least one corresponding"):
            validate(payload)

    def test_equal_contribution_requires_two_authors(self) -> None:
        payload = valid_payload()
        payload["authors"][1]["equal_contribution"] = False
        with self.assertRaisesRegex(ValueError, "at least two authors"):
            validate(payload)

    def test_contributions_must_name_every_author(self) -> None:
        payload = valid_payload()
        payload["declarations"]["contributions"] = (
            "AE conceived the study, performed all analyses, wrote the manuscript, and approved the final version."
        )
        with self.assertRaisesRegex(ValueError, "omits author initials"):
            validate(payload)

    def test_author_confirmations_are_hard_gates(self) -> None:
        for field in (
            "repository_is_permanent",
            "originality_confirmed",
            "not_under_consideration_elsewhere",
            "all_authors_approved",
        ):
            with self.subTest(field=field):
                payload = copy.deepcopy(valid_payload())
                payload["submission"][field] = False
                with self.assertRaisesRegex(ValueError, "explicitly set to true"):
                    validate(payload)

    def test_orcid_checksum_is_validated(self) -> None:
        payload = valid_payload()
        payload["authors"][0]["orcid"] = "0000-0002-1825-0098"
        with self.assertRaisesRegex(ValueError, "checksum-valid ORCID"):
            validate(payload)

    def test_rejects_unapproved_ethics_placeholder(self) -> None:
        payload = valid_payload()
        payload["declarations"]["ethics_approval"] = "INSERT THE APPROVED ETHICS STATEMENT BEFORE SUBMISSION"
        with self.assertRaisesRegex(ValueError, "placeholder"):
            validate(payload)

    def test_latex_options_are_not_placeholders(self) -> None:
        assert_no_placeholders("main", r"\documentclass[pdflatex,sn-vancouver-num]{sn-jnl}")
        with self.assertRaisesRegex(ValueError, "placeholder"):
            assert_no_placeholders("cover", "[DATE]")


if __name__ == "__main__":
    unittest.main()

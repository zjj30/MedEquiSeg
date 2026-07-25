#!/usr/bin/env python3
"""Run a fresh matched rewrite/no-rewrite MedEquiSeg control.

Both arms use the current Protocol V3 code snapshot, BioMedCLIP text
representation, ATConv projectors, the same shared augmentation policy, fixed
splits, and the same per-case/per-epoch augmentation plan.  The only intended
difference is whether direction-bearing prompt phrases are rewritten after
discrete flips or right-angle rotations.
"""

from __future__ import annotations

import run_protocol_v3 as runner


REWRITE_RUN_ID = "V3_CTRL_REWRITE_MATCHED_20260725"
NO_REWRITE_RUN_ID = "V3_CTRL_NO_REWRITE_MATCHED_20260725"

MATCHED_RUNS = {
    REWRITE_RUN_ID: {
        "recipe": "biomed_lcaug_v2_atconv4",
        "augmentation": "lcaug_v2_dynamic_shared_plan_dataset",
    },
    NO_REWRITE_RUN_ID: {
        "recipe": "biomed_lcaug_v2_atconv4",
        "augmentation": "lcaug_v2_dynamic_shared_plan_no_text_rewrite_dataset",
    },
}


def main() -> None:
    runner.V3_RUNS.update(MATCHED_RUNS)
    runner.PILOT_RUNS = tuple(MATCHED_RUNS)
    runner.CONFIRMATORY_RUNS = tuple(MATCHED_RUNS)
    runner.CODE_FILES = (
        *runner.CODE_FILES,
        "smoke_tests/run_medequiseg_no_rewrite_matched.py",
    )
    runner.main()


if __name__ == "__main__":
    main()

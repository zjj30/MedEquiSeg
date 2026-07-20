#!/usr/bin/env python3
"""Run R11 Dynamic Shared-Plan LCAug under Protocol V3."""

from __future__ import annotations

import run_protocol_v3 as runner


RUN_ID = "V3_R11"
RUN_CONFIG = {
    "recipe": "biomed_lcaug_v2_atconv4",
    "augmentation": "lcaug_v2_dynamic_shared_plan_dataset",
}


def main() -> None:
    runner.V3_RUNS[RUN_ID] = RUN_CONFIG
    runner.PILOT_RUNS = (RUN_ID,)
    runner.CONFIRMATORY_RUNS = (RUN_ID,)
    runner.CODE_FILES = (*runner.CODE_FILES, "smoke_tests/run_protocol_v3_r11.py")
    runner.main()


if __name__ == "__main__":
    main()

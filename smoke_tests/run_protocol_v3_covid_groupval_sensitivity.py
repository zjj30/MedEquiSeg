#!/usr/bin/env python3
"""Run R11 and R11NR on the COVID-19 grouped-validation sensitivity split."""

from __future__ import annotations

import run_protocol_v3 as runner


RUNS = {
    "V3_R11_COVID_GROUPVAL": {
        "recipe": "biomed_lcaug_v2_atconv4",
        "augmentation": "lcaug_v2_dynamic_shared_plan_dataset",
    },
    "V3_R11NR_COVID_GROUPVAL": {
        "recipe": "biomed_lcaug_v2_atconv4",
        "augmentation": "lcaug_v2_dynamic_shared_plan_no_text_rewrite_dataset",
    },
}


def main() -> None:
    runner.V3_RUNS.update(RUNS)
    run_ids = tuple(RUNS)
    runner.PILOT_RUNS = run_ids
    runner.CONFIRMATORY_RUNS = run_ids
    runner.CODE_FILES = (
        *runner.CODE_FILES,
        "smoke_tests/build_protocol_v3_covid_groupval_sensitivity.py",
        "smoke_tests/run_protocol_v3_covid_groupval_sensitivity.py",
    )
    runner.main()


if __name__ == "__main__":
    main()

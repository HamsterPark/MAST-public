"""DataInterpreter: interprets experiment results using Claude."""

from __future__ import annotations

import json
import logging

from mast.core.types import SkillResult
from mast.llm.client import ClaudeClient
from mast.llm.prompts.interpreter_system import INTERPRETER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class DataInterpreter:
    """Interprets experiment results using Claude."""

    def __init__(self, client: ClaudeClient):
        self._client = client

    def interpret_result(self, result: SkillResult, context: str = "") -> str:
        """Analyze a skill result and provide natural language interpretation.

        For example, for STS data: identify peaks, gaps, features.
        For scan data: assess image quality, identify structures.
        """
        result_summary = {
            "skill_name": result.skill_name,
            "success": result.success,
            "data": result.data,
            "elapsed_s": round(result.elapsed_s, 3),
        }
        if result.error:
            result_summary["error"] = result.error

        # Include state info if available
        if result.state_after is not None:
            state = result.state_after
            result_summary["state_after"] = {
                "bias_v": state.bias_v,
                "current_a": state.current_a,
                "z_pos_m": state.z_pos_m,
            }

        prompt_parts = [
            f"Analyze the following STM experiment result:\n\n"
            f"```json\n{json.dumps(result_summary, indent=2, default=str)}\n```"
        ]
        if context:
            prompt_parts.append(f"\nAdditional context: {context}")

        prompt_parts.append(
            "\nProvide a concise interpretation including:\n"
            "1. What was measured and whether it succeeded\n"
            "2. Key features or observations in the data\n"
            "3. Any anomalies or concerns\n"
            "4. Data quality assessment"
        )

        try:
            return self._client.single_turn(
                prompt="\n".join(prompt_parts),
                system=INTERPRETER_SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.error("Failed to interpret result: %s", e)
            return f"Interpretation failed: {e}"

    def suggest_next_steps(self, results: list[SkillResult], goal: str = "") -> str:
        """Based on accumulated results, suggest what to do next."""
        summaries = []
        for i, r in enumerate(results, 1):
            entry = {
                "step": i,
                "skill": r.skill_name,
                "success": r.success,
                "data_keys": list(r.data.keys()) if r.data else [],
            }
            if r.error:
                entry["error"] = r.error
            summaries.append(entry)

        prompt_parts = [
            f"Based on the following sequence of STM experiment results, "
            f"suggest the next steps:\n\n"
            f"```json\n{json.dumps(summaries, indent=2, default=str)}\n```"
        ]
        if goal:
            prompt_parts.append(f"\nExperiment goal: {goal}")

        prompt_parts.append(
            "\nProvide:\n"
            "1. Assessment of progress so far\n"
            "2. Recommended next 1-3 actions\n"
            "3. Any parameter adjustments to consider\n"
            "4. Potential issues to watch for"
        )

        try:
            return self._client.single_turn(
                prompt="\n".join(prompt_parts),
                system=INTERPRETER_SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.error("Failed to suggest next steps: %s", e)
            return f"Suggestion failed: {e}"

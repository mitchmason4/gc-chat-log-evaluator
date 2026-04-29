"""Report generator for the GC Chat Log Evaluator.

Aggregates conversation evaluations into a complete EvaluationReport and provides
CSV and JSON export functionality.
"""

import csv
import io
import json
from datetime import datetime, timezone

from .models import (
    ConversationEvaluation,
    EvaluationReport,
    EvaluationSuite,
    GoalSummary,
)


def build_report(
    suite: EvaluationSuite,
    conversation_evaluations: list[ConversationEvaluation],
    duration: float,
) -> EvaluationReport:
    """Aggregate conversation evaluations into a complete EvaluationReport.

    Computes per-goal statistics (conversations classified, successes, failures,
    success rate), unclassified count, and overall statistics.

    Args:
        suite: The EvaluationSuite that was used for evaluation.
        conversation_evaluations: List of ConversationEvaluation results.
        duration: Total execution duration in seconds.

    Returns:
        An EvaluationReport with aggregated statistics.
    """
    total_conversations = len(conversation_evaluations)
    total_successes = sum(1 for e in conversation_evaluations if e.success)
    total_failures = total_conversations - total_successes
    overall_success_rate = (
        total_successes / total_conversations if total_conversations > 0 else 0.0
    )

    # Count unclassified
    unclassified_count = sum(
        1 for e in conversation_evaluations if e.classified_goal == "unclassified"
    )

    # Build per-goal summaries
    goal_stats: dict[str, dict] = {}
    for evaluation in conversation_evaluations:
        if evaluation.classified_goal == "unclassified":
            continue
        goal_name = evaluation.classified_goal
        if goal_name not in goal_stats:
            goal_stats[goal_name] = {"classified": 0, "successes": 0}
        goal_stats[goal_name]["classified"] += 1
        if evaluation.success:
            goal_stats[goal_name]["successes"] += 1

    goal_summaries = []
    for goal_name, stats in goal_stats.items():
        classified = stats["classified"]
        successes = stats["successes"]
        failures = classified - successes
        success_rate = successes / classified if classified > 0 else 0.0
        goal_summaries.append(
            GoalSummary(
                goal_name=goal_name,
                conversations_classified=classified,
                successes=successes,
                failures=failures,
                success_rate=success_rate,
            )
        )

    return EvaluationReport(
        suite_name=suite.name,
        timestamp=datetime.now(timezone.utc),
        duration_seconds=duration,
        goal_summaries=goal_summaries,
        unclassified_count=unclassified_count,
        conversation_evaluations=conversation_evaluations,
        total_conversations=total_conversations,
        total_successes=total_successes,
        total_failures=total_failures,
        overall_success_rate=overall_success_rate,
    )


def export_csv(report: EvaluationReport) -> str:
    """Export EvaluationReport as CSV string.

    Columns: conversation_id, identified_goal, success, explanation.
    Includes a summary row at the end with overall statistics.

    Args:
        report: The EvaluationReport to export.

    Returns:
        A CSV-formatted string.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow(["conversation_id", "identified_goal", "success", "explanation"])

    # Data rows
    for evaluation in report.conversation_evaluations:
        writer.writerow([
            evaluation.conversation_id,
            evaluation.classified_goal,
            evaluation.success,
            evaluation.achievement_explanation,
        ])

    # Summary row
    writer.writerow([
        "OVERALL",
        f"{report.total_conversations} conversations",
        f"{report.overall_success_rate:.1%} success rate",
        f"{report.total_successes} successes, {report.total_failures} failures",
    ])

    return output.getvalue()


def export_json(report: EvaluationReport) -> str:
    """Export EvaluationReport as valid JSON string using Pydantic's model_dump.

    Includes full conversation message history for each ConversationEvaluation.

    Args:
        report: The EvaluationReport to export.

    Returns:
        A JSON-formatted string.
    """
    return json.dumps(report.model_dump(mode="json"), indent=2)

"""CLI entry point for the GC Chat Log Evaluator.

Parses command-line arguments, loads chat log and evaluation suite files,
runs the two-step evaluation pipeline, prints progress and a formatted
summary report to the console, and optionally exports results to a file.
"""

import argparse
import os
import sys
import threading

from .app_config import load_app_config, merge_config, validate_required_config
from .chat_log_parser import parse_chat_log
from .judge_llm import JudgeLLMClient, JudgeLLMError
from .models import AppConfig, EvaluationReport, ProgressEvent, ProgressEventType
from .orchestrator import EvaluationOrchestrator
from .progress import ProgressEmitter
from .report import export_csv, export_json
from .suite_loader import load_evaluation_suite


def _parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="GC Chat Log Evaluator — LLM-as-judge evaluation for Genesys Cloud chat logs"
    )
    parser.add_argument(
        "chat_log",
        help="Path to chat log file (JSON or CSV)",
    )
    parser.add_argument(
        "evaluation_suite",
        help="Path to evaluation suite file (YAML or JSON)",
    )
    parser.add_argument(
        "--ollama-url",
        help="Ollama base URL override",
    )
    parser.add_argument(
        "--ollama-model",
        help="Ollama model name override",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="LLM request timeout in seconds override",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "json"],
        help="Export format (csv or json)",
    )
    parser.add_argument(
        "--output-file",
        help="Output file path (defaults to stdout if --output-format is set)",
    )
    return parser.parse_args(argv)


def _merge_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Merge CLI argument overrides into the base config.

    CLI values take highest precedence over env vars and config file.

    Args:
        config: Base AppConfig loaded from env/file.
        args: Parsed CLI arguments.

    Returns:
        New AppConfig with CLI overrides applied.
    """
    overrides: dict = {}

    if args.ollama_url is not None:
        overrides["ollama_base_url"] = args.ollama_url
    if args.ollama_model is not None:
        overrides["ollama_model"] = args.ollama_model
    if args.timeout is not None:
        overrides["llm_timeout"] = args.timeout

    return merge_config(config, overrides)


def _detect_chat_log_format(file_path: str) -> str:
    """Detect chat log format from file extension.

    Args:
        file_path: Path to the chat log file.

    Returns:
        Format string: "json" or "csv".

    Raises:
        ValueError: If the extension is not .json or .csv.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".json":
        return "json"
    elif ext == ".csv":
        return "csv"
    else:
        raise ValueError(
            f"Unsupported chat log file extension '{ext}'. Use .json or .csv"
        )


def _progress_printer(progress_queue, stop_event: threading.Event) -> None:
    """Print progress events from the queue to the console.

    Runs in a separate thread, consuming events until stop_event is set
    and the queue is drained.

    Args:
        progress_queue: Queue subscribed to the ProgressEmitter.
        stop_event: Event signaling that execution is complete.
    """
    while not stop_event.is_set() or not progress_queue.empty():
        try:
            event: ProgressEvent = progress_queue.get(timeout=0.5)
            _print_progress_event(event)
        except Exception:
            continue


def _print_progress_event(event: ProgressEvent) -> None:
    """Format and print a single progress event to the console.

    Args:
        event: The progress event to print.
    """
    prefix = {
        ProgressEventType.EVALUATION_STARTED: "🚀",
        ProgressEventType.EVALUATION_IN_PROGRESS: "📋",
        ProgressEventType.EVALUATION_COMPLETED: "  ✓" if event.success else "  ✗",
        ProgressEventType.EVALUATION_RUN_COMPLETED: "🏁",
    }.get(event.event_type, "•")

    print(f"{prefix} {event.message}")


def _print_report(report: EvaluationReport) -> None:
    """Print a formatted evaluation report summary to the console.

    Shows per-goal success rates, unclassified count, and overall statistics.

    Args:
        report: The EvaluationReport to display.
    """
    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT: {report.suite_name}")
    print("=" * 60)
    print(f"Duration: {report.duration_seconds:.1f}s")
    print(f"Conversations: {report.total_conversations}")
    print(f"Overall: {report.total_successes}/{report.total_conversations} "
          f"({report.overall_success_rate:.0%} success rate)")
    print("-" * 60)

    # Per-goal summaries
    if report.goal_summaries:
        print("Per-Goal Results:")
        for summary in report.goal_summaries:
            print(f"  {summary.goal_name}: "
                  f"{summary.successes}/{summary.conversations_classified} "
                  f"({summary.success_rate:.0%} success rate)")
        print()

    # Unclassified count
    if report.unclassified_count > 0:
        print(f"Unclassified: {report.unclassified_count} conversations")
        print()

    print("-" * 60)
    if report.overall_success_rate >= 1.0:
        print("✅ ALL CONVERSATIONS ACHIEVED THEIR GOALS")
    else:
        print(f"📊 {report.overall_success_rate:.0%} overall success rate")
    print("=" * 60)


def main(argv=None) -> None:
    """CLI entry point. Parse args, load files, run evaluation, print report, exit with code.

    Args:
        argv: Optional argument list for testing (defaults to sys.argv[1:]).
    """
    args = _parse_args(argv)

    # Load base config from env vars / config file
    config = load_app_config()

    # Merge CLI overrides (highest precedence)
    config = _merge_cli_overrides(config, args)

    # Validate required config
    missing = validate_required_config(config)
    if missing:
        print(
            f"Error: Missing required configuration: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Detect chat log format from extension
    try:
        chat_log_format = _detect_chat_log_format(args.chat_log)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Load chat log file
    try:
        with open(args.chat_log, "r", encoding="utf-8") as f:
            chat_log_content = f.read()
        conversations = parse_chat_log(chat_log_content, chat_log_format)
    except FileNotFoundError:
        print(f"Error: Chat log file not found: {args.chat_log}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, Exception) as e:
        print(f"Error loading chat log: {e}", file=sys.stderr)
        sys.exit(1)

    # Load evaluation suite file
    try:
        suite = load_evaluation_suite(args.evaluation_suite)
    except FileNotFoundError:
        print(
            f"Error: Evaluation suite file not found: {args.evaluation_suite}",
            file=sys.stderr,
        )
        sys.exit(1)
    except (ValueError, Exception) as e:
        print(f"Error loading evaluation suite: {e}", file=sys.stderr)
        sys.exit(1)

    # Verify Ollama connection
    judge = JudgeLLMClient(
        base_url=config.ollama_base_url,
        model=config.ollama_model or "",
        timeout=config.llm_timeout,
    )
    try:
        judge.verify_connection()
    except JudgeLLMError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Set up progress emitter and console printer thread
    emitter = ProgressEmitter()
    progress_queue = emitter.subscribe()
    stop_event = threading.Event()
    printer_thread = threading.Thread(
        target=_progress_printer,
        args=(progress_queue, stop_event),
        daemon=True,
    )
    printer_thread.start()

    # Run the evaluation
    orchestrator = EvaluationOrchestrator(config=config, progress_emitter=emitter)
    report = orchestrator.run_evaluation(suite, conversations)

    # Signal printer thread to stop and wait for it
    stop_event.set()
    printer_thread.join(timeout=5)

    # Print the formatted report
    _print_report(report)

    # Export if requested
    if args.output_format:
        if args.output_format == "csv":
            export_content = export_csv(report)
        else:
            export_content = export_json(report)

        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(export_content)
            print(f"\nReport exported to: {args.output_file}")
        else:
            print("\n" + export_content)

    sys.exit(0)


if __name__ == "__main__":
    main()

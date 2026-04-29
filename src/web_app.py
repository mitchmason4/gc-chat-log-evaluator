"""Flask web application for the GC Chat Log Evaluator.

Provides a web UI for uploading chat logs and evaluation suites, triggering
evaluation, viewing results grouped by goal, and streaming progress via SSE.
"""

import json
import os
import queue
import threading
from typing import Optional

from flask import (
    Flask,
    Response,
    redirect,
    render_template,
    request,
    url_for,
)
from pydantic import ValidationError

from .app_config import load_app_config, merge_config, validate_required_config
from .chat_log_parser import parse_chat_log
from .judge_llm import JudgeLLMError
from .models import AppConfig, EvaluationReport
from .orchestrator import EvaluationOrchestrator
from .progress import ProgressEmitter
from .report import export_csv, export_json
from .suite_loader import load_evaluation_suite_from_string


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "templates"
        ),
    )
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

    # App state
    app.config["latest_report"]: Optional[EvaluationReport] = None
    app.config["progress_emitter"] = ProgressEmitter()
    app.config["run_active"] = False

    @app.route("/")
    def home():
        """Home page with dual file upload and Ollama config inputs."""
        base_config = load_app_config()
        return render_template(
            "home.html",
            config=base_config,
            errors=None,
        )

    @app.route("/run", methods=["POST"])
    def run():
        """Trigger evaluation from form submission."""
        base_config = load_app_config()

        # Read form fields
        ollama_model = request.form.get("ollama_model", "").strip()
        ollama_url = request.form.get("ollama_url", "").strip()
        timeout = request.form.get("timeout", "").strip()

        # Read uploaded chat log file
        chat_log_file = request.files.get("chat_log_file")
        if not chat_log_file or chat_log_file.filename == "":
            return render_template(
                "home.html",
                config=base_config,
                errors=["Please upload a chat log file (JSON or CSV)."],
            )

        # Read uploaded evaluation suite file
        suite_file = request.files.get("evaluation_suite_file")
        if not suite_file or suite_file.filename == "":
            return render_template(
                "home.html",
                config=base_config,
                errors=["Please upload an evaluation suite file (JSON or YAML)."],
            )

        # Determine chat log format from extension
        chat_log_filename = chat_log_file.filename.lower()
        if chat_log_filename.endswith(".json"):
            chat_log_fmt = "json"
        elif chat_log_filename.endswith(".csv"):
            chat_log_fmt = "csv"
        else:
            return render_template(
                "home.html",
                config=base_config,
                errors=["Unsupported chat log format. Use .json or .csv"],
            )

        # Determine suite format from extension
        suite_filename = suite_file.filename.lower()
        if suite_filename.endswith(".json"):
            suite_fmt = "json"
        elif suite_filename.endswith((".yaml", ".yml")):
            suite_fmt = "yaml"
        else:
            return render_template(
                "home.html",
                config=base_config,
                errors=["Unsupported evaluation suite format. Use .json, .yaml, or .yml"],
            )

        # Read and parse chat log
        try:
            chat_log_content = chat_log_file.read().decode("utf-8")
        except UnicodeDecodeError:
            return render_template(
                "home.html",
                config=base_config,
                errors=["Chat log file must be valid UTF-8 text."],
            )

        try:
            conversations = parse_chat_log(chat_log_content, chat_log_fmt)
        except (ValueError, ValidationError) as e:
            return render_template(
                "home.html",
                config=base_config,
                errors=[f"Invalid chat log: {e}"],
            )

        # Read and parse evaluation suite
        try:
            suite_content = suite_file.read().decode("utf-8")
        except UnicodeDecodeError:
            return render_template(
                "home.html",
                config=base_config,
                errors=["Evaluation suite file must be valid UTF-8 text."],
            )

        try:
            suite = load_evaluation_suite_from_string(suite_content, suite_fmt)
        except (ValueError, ValidationError) as e:
            return render_template(
                "home.html",
                config=base_config,
                errors=[f"Invalid evaluation suite: {e}"],
            )

        # Merge web overrides with base config
        web_overrides = {}
        if ollama_model:
            web_overrides["ollama_model"] = ollama_model
        if ollama_url:
            web_overrides["ollama_base_url"] = ollama_url
        if timeout:
            web_overrides["llm_timeout"] = timeout

        merged_config = merge_config(base_config, web_overrides)

        # Validate required config
        missing = validate_required_config(merged_config)
        if missing:
            errors = [
                f"Missing required configuration: {', '.join(missing)}"
            ]
            return render_template(
                "home.html",
                config=base_config,
                errors=errors,
            )

        # Create a fresh progress emitter for this run
        progress_emitter = ProgressEmitter()
        app.config["progress_emitter"] = progress_emitter
        app.config["latest_report"] = None
        app.config["run_active"] = True

        # Start evaluation in a background thread
        def run_evaluation():
            try:
                orchestrator = EvaluationOrchestrator(
                    config=merged_config,
                    progress_emitter=progress_emitter,
                )
                report = orchestrator.run_evaluation(suite, conversations)
                app.config["latest_report"] = report
            except JudgeLLMError:
                pass
            finally:
                app.config["run_active"] = False

        thread = threading.Thread(target=run_evaluation, daemon=True)
        thread.start()

        return redirect(url_for("results"))

    @app.route("/results")
    def results():
        """Results page displaying evaluations grouped by classified goal."""
        report = app.config.get("latest_report")
        run_active = app.config.get("run_active", False)
        return render_template("results.html", report=report, run_active=run_active)

    @app.route("/results/export")
    def export():
        """Download report as CSV or JSON file."""
        report = app.config.get("latest_report")
        if report is None:
            return redirect(url_for("results"))

        fmt = request.args.get("format", "json").lower()

        if fmt == "csv":
            content = export_csv(report)
            return Response(
                content,
                mimetype="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=report.csv"
                },
            )
        else:
            content = export_json(report)
            return Response(
                content,
                mimetype="application/json",
                headers={
                    "Content-Disposition": "attachment; filename=report.json"
                },
            )

    @app.route("/progress")
    def progress():
        """SSE endpoint streaming ProgressEvent data to the browser."""
        emitter: ProgressEmitter = app.config["progress_emitter"]

        def event_stream():
            q = emitter.subscribe()
            try:
                while True:
                    try:
                        event = q.get(timeout=30)
                        data = event.model_dump(mode="json")
                        yield f"data: {json.dumps(data)}\n\n"
                        # Stop streaming after evaluation_run_completed
                        if event.event_type.value == "evaluation_run_completed":
                            break
                    except queue.Empty:
                        # Send keepalive comment
                        yield ": keepalive\n\n"
            finally:
                emitter.unsubscribe(q)

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=True)

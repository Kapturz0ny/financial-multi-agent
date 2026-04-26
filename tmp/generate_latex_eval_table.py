"""Run repeated report evaluations and render a LaTeX table.

Example:
    uv run python tmp/generate_latex_eval_table.py --mode parallel --trials 5
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass

from src.crews import StockAnalysisCrewFactory
from src.utils.report_evaluator import ReportEvaluator


MODE_ALIASES = {
    "sequential": "sequential",
    "parallel": "parallel",
    "concurrent": "concurrent",
    "group-chat": "group_chat_v0",
    "group_chat_v0": "group_chat_v0",
    "group-chat-cs-rag": "group_chat_v1",
    "group_chat_v1": "group_chat_v1",
}

MODE_LABELS = {
    "sequential": "Sequential",
    "parallel": "Parallel",
    "concurrent": "Concurrent",
    "group_chat_v0": "Group Chat",
    "group_chat_v1": "Group Chat (CS + RAG)",
}

DIMENSION_WEIGHTS = {
    "structure": 0.25,
    "data_richness": 0.25,
    "sophistication": 0.20,
    "actionability": 0.20,
    "sentiment_balance": 0.10,
}


@dataclass
class TrialResult:
    structure: float
    data_richness: float
    sophistication: float
    actionability: float
    sentiment_balance: float
    total: float
    execution_time: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LaTeX table from repeated crew runs.")
    parser.add_argument("--mode", default="parallel", help="Mode: parallel/sequential/concurrent/group-chat/group-chat-cs-rag")
    parser.add_argument("--provider", default=None, help="Optional provider override (e.g. gemini, openai, local)")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "NVDA", "GOOGL"], help="Tickers to evaluate")
    parser.add_argument("--trials", type=int, default=5, help="Number of runs per ticker")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Pause between attempts")
    parser.add_argument("--decimals", type=int, default=1, help="Output decimal precision")
    parser.add_argument("--raw-dimensions", action="store_true", help="Use raw 0-100 dimensions instead of weighted max(25/25/20/20/10)")
    parser.add_argument("--caption", default=None, help="LaTeX table caption override")
    parser.add_argument("--label", default="tab:res_parallel", help="LaTeX table label")
    parser.add_argument("--output", default=None, help="Optional output .tex file path")
    return parser.parse_args()


def resolve_mode(user_mode: str) -> str:
    key = user_mode.strip().lower()
    if key not in MODE_ALIASES:
        available = ", ".join(sorted(MODE_ALIASES))
        raise ValueError(f"Unknown mode '{user_mode}'. Available: {available}")
    return MODE_ALIASES[key]


def run_trial(mode: str, provider: str | None, ticker: str, raw_dimensions: bool) -> TrialResult:
    crew = StockAnalysisCrewFactory.create(mode, provider)
    result = crew.run(ticker)

    if not isinstance(result, dict):
        raise RuntimeError("Crew did not return a result dictionary.")
    if result.get("error"):
        raise RuntimeError(str(result["error"]))

    report = str(result.get("report", ""))
    if not report.strip():
        raise RuntimeError("Crew returned an empty report.")

    evaluator = ReportEvaluator()
    evaluation = evaluator.evaluate(report, ticker)
    dims = evaluation["dimension_scores"]

    if raw_dimensions:
        structure = float(dims["structure"])
        data_richness = float(dims["data_richness"])
        sophistication = float(dims["sophistication"])
        actionability = float(dims["actionability"])
        sentiment_balance = float(dims["sentiment_balance"])
    else:
        structure = float(dims["structure"]) * DIMENSION_WEIGHTS["structure"]
        data_richness = float(dims["data_richness"]) * DIMENSION_WEIGHTS["data_richness"]
        sophistication = float(dims["sophistication"]) * DIMENSION_WEIGHTS["sophistication"]
        actionability = float(dims["actionability"]) * DIMENSION_WEIGHTS["actionability"]
        sentiment_balance = float(dims["sentiment_balance"]) * DIMENSION_WEIGHTS["sentiment_balance"]

    total = float(evaluation["overall_score"])
    execution_time = float(result.get("execution_time", math.nan))

    return TrialResult(
        structure=structure,
        data_richness=data_richness,
        sophistication=sophistication,
        actionability=actionability,
        sentiment_balance=sentiment_balance,
        total=total,
        execution_time=execution_time,
    )


def mean_pm_std(values: list[float], decimals: int) -> str:
    if not values:
        return "N/A"
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def build_latex_table(
    mode: str,
    trials: int,
    tickers: list[str],
    aggregated: dict[str, dict[str, list[float]]],
    decimals: int,
    label: str,
    caption: str | None,
    raw_dimensions: bool,
) -> str:
    mode_label = MODE_LABELS.get(mode, mode)
    if caption is None:
        caption = (
            f"Wyniki ewaluacji dla trybu {mode_label} "
            f"(srednia z {trials} prob $\\pm$ odchylenie standardowe)"
        )

    if raw_dimensions:
        dim_row = "& (max 100) & (max 100) & (max 100) & (max 100) & (max 100) & \\textbf{(max 100)} & \\\\"  # noqa: E501
    else:
        dim_row = "& (max 25) & (max 25) & (max 20) & (max 20) & (max 10) & \\textbf{(max 100)} & \\\\"  # noqa: E501

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "\\textbf{Spolka} & \\textbf{Struktura} & \\textbf{Bogactwo} & \\textbf{Prof. jezyka} & \\textbf{Uzytecznosc} & \\textbf{Sentyment} & \\textbf{Wynik calkowity} & \\textbf{Czas [s]} \\\\ ",
        dim_row,
        "\\midrule",
    ]

    for ticker in tickers:
        row_data = aggregated[ticker]
        structure = mean_pm_std(row_data["structure"], decimals)
        richness = mean_pm_std(row_data["data_richness"], decimals)
        sophistication = mean_pm_std(row_data["sophistication"], decimals)
        actionability = mean_pm_std(row_data["actionability"], decimals)
        sentiment = mean_pm_std(row_data["sentiment_balance"], decimals)
        total = mean_pm_std(row_data["total"], decimals)
        timing = mean_pm_std(row_data["execution_time"], decimals)

        lines.append(
            f"\\textbf{{{ticker}}} & {structure} & {richness} & {sophistication} & {actionability} & {sentiment} & \\textbf{{{total}}} & {timing} \\\\"  # noqa: E501
        )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    if args.trials <= 0:
        raise ValueError("--trials must be greater than 0")

    mode = resolve_mode(args.mode)
    tickers = [ticker.upper() for ticker in args.tickers]

    aggregated: dict[str, dict[str, list[float]]] = {
        ticker: {
            "structure": [],
            "data_richness": [],
            "sophistication": [],
            "actionability": [],
            "sentiment_balance": [],
            "total": [],
            "execution_time": [],
        }
        for ticker in tickers
    }

    for ticker in tickers:
        print(f"\\n=== {ticker} | mode={mode} | trials={args.trials} ===")
        for attempt in range(1, args.trials + 1):
            try:
                result = run_trial(mode, args.provider, ticker, args.raw_dimensions)
            except Exception as exc:
                print(f"[{ticker}] trial {attempt}/{args.trials} failed: {exc}", file=sys.stderr)
            else:
                aggregated[ticker]["structure"].append(result.structure)
                aggregated[ticker]["data_richness"].append(result.data_richness)
                aggregated[ticker]["sophistication"].append(result.sophistication)
                aggregated[ticker]["actionability"].append(result.actionability)
                aggregated[ticker]["sentiment_balance"].append(result.sentiment_balance)
                aggregated[ticker]["total"].append(result.total)
                aggregated[ticker]["execution_time"].append(result.execution_time)
                print(
                    f"[{ticker}] trial {attempt}/{args.trials} ok | "
                    f"score={result.total:.2f} | time={result.execution_time:.2f}s"
                )

            if args.sleep_seconds > 0 and attempt < args.trials:
                time.sleep(args.sleep_seconds)

    latex = build_latex_table(
        mode=mode,
        trials=args.trials,
        tickers=tickers,
        aggregated=aggregated,
        decimals=args.decimals,
        label=args.label,
        caption=args.caption,
        raw_dimensions=args.raw_dimensions,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"\\nSaved LaTeX table to: {args.output}")
    else:
        print("\\n" + latex)

    for ticker in tickers:
        success_n = len(aggregated[ticker]["total"])
        if success_n < args.trials:
            print(
                f"Warning: {ticker} has {success_n}/{args.trials} successful runs.",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from datetime import datetime
import re
import time as sleep_time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from markdown_it import MarkdownIt

from src.auth import DAILY_QUERY_LIMIT, logout_button, record_usage, remaining, require_login, reset_today_usage
from src.config import LLMProvider, get_default_provider
from src.crews import CrewMode, StockAnalysisCrewFactory
from src.llm_router import select_provider
from src.tools.qdrant_tools import qdrant_service
from src.utils.chart_builder import ChartBuilder
from src.utils.pdf_exporter import PDFReportExporter
from src.utils.report_evaluator import ReportEvaluator

INTERVAL_MAPPING = [
    {"period": "1d", "interval": "1m"},
    {"period": "5d", "interval": "30m"},
    {"period": "1mo", "interval": "1d"},
    {"period": "6mo", "interval": "1d"},
    {"period": "ytd", "interval": "1d"},
    {"period": "1y", "interval": "1d"},
    {"period": "5y", "interval": "1wk"},
    {"period": "max", "interval": "1wk"},
]

RATE_LIMIT_WAIT_MS_PATTERN = re.compile(r"try again in\s+(\d+)ms", re.IGNORECASE)
RATE_LIMIT_WAIT_S_PATTERN = re.compile(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)


# Calculate basic metrics from the stock data
def calculate_metrics(data):
    last_close = data["Close"].iloc[-1]
    prev_close = data["Close"].iloc[0]
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high = data["High"].max()
    low = data["Low"].min()
    volume = data["Volume"].sum()
    return last_close, change, pct_change, high, low, volume


# Add simple moving average (SMA) and exponential moving average (EMA) indicators
def add_technical_indicators(data: pd.DataFrame, indicators: dict) -> pd.DataFrame:
    """Add selected technical indicators to the dataframe."""
    return ChartBuilder.add_indicators(data, indicators)


def format_markdown(text):
    md = MarkdownIt()
    tokens = md.parse(text)
    formatted = ""
    for token in tokens:
        if token.type == "paragraph_open":
            formatted += "\n\n"
        formatted += token.content
    return formatted.strip()


def escape_markdown_specials(text: str) -> str:
    text = text.replace("$", r"\$")
    return text


def load_stock_data(symbol: str, period: dict) -> pd.DataFrame:
    """Load and process stock data - wrapper for ChartBuilder."""
    if not symbol or not symbol.strip():
        raise ValueError("Please enter a valid stock symbol before updating.")
    return ChartBuilder.load_and_process_data(symbol, period["period"])


def is_rate_limit_error(message: object) -> bool:
    if message is None:
        return False
    lowered = str(message).lower()
    markers = (
        "rate limit",
        "rate_limit_exceeded",
        "error code: 429",
        "status code: 429",
        "http 429",
        "tokens per min",
        "too many requests",
    )
    return any(marker in lowered for marker in markers)


def retry_delay_seconds(message: object, default_seconds: float = 2.0) -> float:
    if message is None:
        return default_seconds

    text = str(message)

    match = RATE_LIMIT_WAIT_MS_PATTERN.search(text)
    if match:
        delay = int(match.group(1)) / 1000.0
        # Add a small safety buffer to avoid immediate re-throttle.
        return max(0.5, min(20.0, delay + 0.35))

    match = RATE_LIMIT_WAIT_S_PATTERN.search(text)
    if match:
        delay = float(match.group(1))
        # Add a small safety buffer to avoid immediate re-throttle.
        return max(0.5, min(20.0, delay + 0.35))

    return default_seconds


if "stock_fig" not in st.session_state:
    st.session_state.stock_fig = None
if "stock_metrics" not in st.session_state:
    st.session_state.stock_metrics = None
if "report" not in st.session_state:
    st.session_state.report = None
if "report_mode" not in st.session_state:
    st.session_state.report_mode = None
if "report_provider" not in st.session_state:
    st.session_state.report_provider = None
if "execution_time" not in st.session_state:
    st.session_state.execution_time = None
if "selected_indicators" not in st.session_state:
    st.session_state.selected_indicators = {}
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "report_context" not in st.session_state:
    st.session_state.report_context = None
if "report_evidence" not in st.session_state:
    st.session_state.report_evidence = None
if "fallback_active" not in st.session_state:
    st.session_state.fallback_active = False
if "fallback_reason" not in st.session_state:
    st.session_state.fallback_reason = None
if "last_effective_provider" not in st.session_state:
    st.session_state.last_effective_provider = None

st.set_page_config("Stock Investment Report", layout="wide")
st.title("📈 Stock Investment Analysis Platform")

current_user = require_login()
username = current_user["username"]

st.sidebar.header("Configuration")
st.sidebar.markdown(f"👤 Zalogowano jako **{current_user.get('name') or username}**")

remaining_today = remaining(username)
if username == "demo":
    if st.sidebar.button("Resetuj limit Demo (dzisiaj)", width='stretch'):
        removed_rows = reset_today_usage(username)
        remaining_today = remaining(username)
        st.sidebar.success(f"Zresetowano dzienny limit Demo. Usunięte wpisy: {removed_rows}.")

st.sidebar.info(f"Pozostało zapytań dziś: {remaining_today}/{DAILY_QUERY_LIMIT}")
logout_button(location="sidebar")
st.sidebar.divider()
ticker = st.sidebar.text_input("Stock symbol (eg. AAPL)")
time_period = st.sidebar.selectbox("Time period", [period["period"] for period in INTERVAL_MAPPING])
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line"])

_provider_labels = {
    LLMProvider.GEMINI.value: "Gemini",
    LLMProvider.OPENAI.value: "OpenAI",
    LLMProvider.LOCAL.value: "Local Llama (Ollama)",
}
_provider_options = [p.value for p in LLMProvider]
_default_provider_value = get_default_provider().lower()
_default_index = _provider_options.index(_default_provider_value) if _default_provider_value in _provider_options else 0

llm_provider = st.sidebar.selectbox(
    "LLM Provider",
    options=_provider_options,
    index=_default_index,
    format_func=lambda x: _provider_labels.get(x, x),
)

crew_mode = st.sidebar.radio(
    "Analysis Mode",
    options=[
        CrewMode.SEQUENTIAL.value,
        CrewMode.PARALLEL.value,
        CrewMode.CONCURRENT.value,
        CrewMode.GROUP_CHAT_V0.value,
        CrewMode.GROUP_CHAT_V1.value
    ],
    format_func=lambda x: {
        CrewMode.SEQUENTIAL.value: "Sequential",
        CrewMode.PARALLEL.value: "Parallel",
        CrewMode.CONCURRENT.value: "Concurrent (Multi-round)",
        CrewMode.GROUP_CHAT_V0.value: "Group Chat (Original)",
        CrewMode.GROUP_CHAT_V1.value: "Group Chat (CS + RAG)"
    }.get(x, str(x)),
    horizontal=True
)

st.sidebar.subheader("📊 Technical Indicators")
with st.sidebar.expander("Select Indicators", expanded=True):
    st.write("**Moving Averages**")
    indicators = {
        "SMA 20": st.checkbox("SMA 20", value=True),
        "SMA 50": st.checkbox("SMA 50", value=False),
        "SMA 200": st.checkbox("SMA 200", value=False),
        "EMA 20": st.checkbox("EMA 20", value=False),
        "EMA 50": st.checkbox("EMA 50", value=False),
    }

    st.write("**Volatility**")
    indicators.update({
        "Bollinger Bands": st.checkbox("Bollinger Bands", value=False),
    })

sidebar_col1, sidebar_col2 = st.sidebar.columns(spec=[0.4, 0.6], gap="small")

if sidebar_col1.button("Update", type="primary", width='stretch'):
    with st.spinner("Loading stock data…"):
        try:
            data = load_stock_data(ticker, next(filter(lambda x: x["period"] == time_period, INTERVAL_MAPPING)))

            last_close, change, pct_change, high, low, volume = calculate_metrics(data)
            st.session_state.stock_metrics = {
                "last_close": last_close,
                "change": change,
                "pct_change": pct_change,
                "high": high,
                "low": low,
                "volume": volume,
            }

            # Add selected technical indicators
            data = add_technical_indicators(data, indicators)

            # Build chart using ChartBuilder
            fig = ChartBuilder.build_chart(data, ticker, indicators, time_period, chart_type)

            st.session_state.stock_fig = fig
            st.session_state.selected_indicators = indicators
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error("Unable to load stock data. Please check the ticker symbol and your internet connection.")
            st.exception(e)

if sidebar_col2.button("Generate report", type="primary", width='stretch'):
    with st.spinner("Running multi-agent analysis…"):
        try:
            effective_provider = select_provider(
                username,
                requested_provider=llm_provider,
                fallback_state=st.session_state,
            )
            st.session_state.last_effective_provider = effective_provider

            max_attempts = 3 if effective_provider != LLMProvider.LOCAL.value else 1
            result = None
            current_error = None

            for attempt in range(1, max_attempts + 1):
                try:
                    crew = StockAnalysisCrewFactory.create(crew_mode, effective_provider)
                    result = crew.run(ticker)
                    current_error = result.get("error") if isinstance(result, dict) else None
                except Exception as run_error:
                    current_error = str(run_error)
                    result = {"error": current_error}

                if current_error and is_rate_limit_error(current_error) and attempt < max_attempts:
                    delay_seconds = retry_delay_seconds(current_error)
                    st.warning(
                        f"Rate limit po stronie {effective_provider.upper()}. "
                        f"Ponawiam automatycznie za {delay_seconds:.1f}s (próba {attempt + 1}/{max_attempts})."
                    )
                    sleep_time.sleep(delay_seconds)
                    continue

                break

            if result is None:
                raise RuntimeError("Analysis returned no result.")

            if isinstance(result, dict) and result.get("error"):
                if is_rate_limit_error(result["error"]):
                    suggested_wait = retry_delay_seconds(result["error"])
                    st.error(
                        "Analysis Error: osiągnięto limit szybkości API (429). "
                        f"Spróbuj ponownie za około {suggested_wait:.1f}s albo wybierz provider Local."
                    )
                    with st.expander("Szczegóły błędu 429", expanded=False):
                        st.code(result["error"])
                else:
                    st.error(f"Analysis Error: {result['error']}")
                st.session_state.report = None
                st.session_state.report_context = None
                st.session_state.report_evidence = None
            else:
                report_md = format_markdown(str(result["report"]))
                report_cleaned = escape_markdown_specials(report_md)
                st.session_state.report = report_cleaned
                st.session_state.report_mode = result["mode"]
                st.session_state.report_provider = result["provider"]
                st.session_state.execution_time = result["execution_time"]
                st.session_state.report_context = result.get("context_data")

                if crew_mode in (CrewMode.GROUP_CHAT_V1.value, CrewMode.CONCURRENT.value):
                    st.session_state.report_evidence = qdrant_service.get_all_evidence()
                else:
                    st.session_state.report_evidence = None

                # Charge quota only after a successful analysis run.
                record_usage(username, effective_provider, ticker=ticker, mode=crew_mode)
        except ValueError as e:
            error_message = str(e)
            key_error_markers = (
                "api key",
                "api_key",
                "token",
                "credential",
                "openai",
                "anthropic",
                "groq",
            )
            if any(marker in error_message.lower() for marker in key_error_markers):
                st.error(f"Configuration Error: {error_message}\n\nPlease ensure API keys are set in your .env file.")
            else:
                st.error(f"Configuration Error: {error_message}")
        except Exception as e:
            error_message = str(e)
            if is_rate_limit_error(error_message):
                suggested_wait = retry_delay_seconds(error_message)
                st.error(
                    "Analysis Error: osiągnięto limit szybkości API (429). "
                    f"Spróbuj ponownie za około {suggested_wait:.1f}s albo wybierz provider Local."
                )
                with st.expander("Szczegóły błędu 429", expanded=False):
                    st.code(error_message)
            else:
                st.error("Analysis failed and was not counted against your daily limit.")
                st.exception(e)

if st.session_state.get("fallback_active") and st.session_state.get("fallback_reason"):
    st.warning(st.session_state["fallback_reason"])

if st.session_state.stock_metrics is not None:
    last_close = st.session_state.stock_metrics["last_close"]
    change = st.session_state.stock_metrics["change"]
    pct_change = st.session_state.stock_metrics["pct_change"]
    high = st.session_state.stock_metrics["high"]
    low = st.session_state.stock_metrics["low"]
    volume = st.session_state.stock_metrics["volume"]

    st.metric(
        label=f"{ticker} Last Price",
        value=f"{last_close:.2f} USD",
        delta=f"{change:.2f} ({pct_change:.2f}%)",
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("High", f"{high:.2f} USD")
    col2.metric("Low", f"{low:.2f} USD")
    col3.metric("Volume", f"{volume:,}")

if st.session_state.stock_fig is not None:
    st.plotly_chart(st.session_state.stock_fig, width='stretch', key="chart_display")

if st.session_state.report is not None:
    st.header("Investment Report")

    # Display metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        mode_labels = {
            CrewMode.SEQUENTIAL.value: "Sequential",
            CrewMode.PARALLEL.value: "Parallel",
            CrewMode.CONCURRENT.value: "Concurrent (Multi-round)",
            CrewMode.GROUP_CHAT_V0.value: "Group Chat V0",
            CrewMode.GROUP_CHAT_V1.value: "Group Chat (CS + RAG)"
        }
        mode_label = mode_labels.get(st.session_state.report_mode, st.session_state.report_mode)
        st.metric("Analysis Mode", mode_label)
    with col2:
        provider_label = _provider_labels.get(st.session_state.report_provider, st.session_state.report_provider)
        st.metric("LLM Provider", provider_label)
    with col3:
        st.metric("Execution Time", f"{st.session_state.execution_time:.1f}s")

    st.divider()

    # Export to PDF button
    exporter = PDFReportExporter()
    pdf_buffer = exporter.export(
        ticker=ticker,
        report_text=st.session_state.report,
        fig=st.session_state.stock_fig,
        metrics=st.session_state.stock_metrics,
        indicators=st.session_state.selected_indicators,
        mode=st.session_state.report_mode,
        provider=st.session_state.report_provider,
        execution_time=st.session_state.execution_time
    )

    col_pdf, col_eval = st.columns(2)

    with col_pdf:
        st.download_button(
            label="📥 Download Report as PDF",
            data=pdf_buffer,
            file_name=f"{ticker.upper()}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            key="download_pdf_btn"
        )

    with col_eval:
        if st.button("📊 Evaluate Report Quality", type="secondary", width='stretch'):
            with st.spinner("Evaluating report quality..."):
                evaluator = ReportEvaluator()
                st.session_state.evaluation_results = evaluator.evaluate(
                    st.session_state.report,
                    ticker.upper()
                )

    st.divider()

    # Display context storage if available
    if st.session_state.report_context:
        with st.expander("📦 View Context Storage (Blackboard Data)", expanded=False):
            st.info("This shared memory contains the structured facts and claims used by agents during the debate.")

            ctx_data = st.session_state.report_context
            tab_facts, tab_claims, tab_evidence, tab_json = st.tabs(["📌 Facts", "💡 Claims", "📚 Raw Evidence (Qdrant)", "📄 Raw JSON"])

            with tab_facts:
                if ctx_data.get("facts"):
                    df_facts = pd.DataFrame(ctx_data["facts"])
                    cols = ['id', 'agent', 'content']
                    st.dataframe(df_facts[cols], width='stretch', hide_index=True)
                else:
                    st.write("No facts recorded in this session.")

            with tab_claims:
                if ctx_data.get("claims"):
                    df_claims = pd.DataFrame(ctx_data["claims"])
                    cols = ['id', 'agent', 'content']
                    if 'refutes_id' in df_claims.columns:
                        cols.append('refutes_id')
                    st.dataframe(df_claims[cols], width='stretch', hide_index=True)
                else:
                    st.write("No claims recorded in this session.")

            with tab_evidence:
                if st.session_state.get("report_evidence"):
                    df_evidence = pd.DataFrame(st.session_state.report_evidence)
                    st.dataframe(df_evidence, width='stretch', hide_index=True)
                else:
                    st.write("No raw evidence recorded in the vector database for this session.")

            with tab_json:
                st.json(ctx_data)

    # Display evaluation results if available
    if st.session_state.evaluation_results is not None:
        eval_data = st.session_state.evaluation_results

        st.subheader("📊 Report Quality Evaluation")

        # Overall score display
        score_col1, score_col2, score_col3 = st.columns([1, 1, 1])

        with score_col1:
            score = eval_data['overall_score']
            grade = eval_data['grade']

            # Color based on grade
            if grade == 'A':
                score_color = "🟢"
            elif grade == 'B':
                score_color = "🟡"
            elif grade == 'C':
                score_color = "🟠"
            else:
                score_color = "🔴"

            st.metric("Overall Quality Score", f"{score:.1f}/100", delta=f"Grade: {grade}")
            st.markdown(f"{score_color} **Quality Level:** {grade}")

        with score_col2:
            best_dim = max(eval_data['dimension_scores'].items(), key=lambda x: x[1])
            st.metric("Strongest Dimension", best_dim[0].replace('_', ' ').title(), f"{best_dim[1]:.1f}/100")

        with score_col3:
            worst_dim = min(eval_data['dimension_scores'].items(), key=lambda x: x[1])
            st.metric("Needs Improvement", worst_dim[0].replace('_', ' ').title(), f"{worst_dim[1]:.1f}/100")

        st.divider()

        # Dimension scores with visualization
        st.subheader("Quality Dimensions")

        dim_scores = eval_data['dimension_scores']

        # Create radar chart data
        import plotly.graph_objects as go

        categories = [k.replace('_', ' ').title() for k in dim_scores.keys()]
        values = list(dim_scores.values())

        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Score',
            line_color='rgb(31, 119, 180)',
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))

        # Add target line at 70
        fig_radar.add_trace(go.Scatterpolar(
            r=[70] * len(categories),
            theta=categories,
            mode='lines',
            name='Target (70)',
            line=dict(color='green', dash='dash', width=2)
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Quality Dimensions Radar Chart",
            height=400
        )

        st.plotly_chart(fig_radar, width='stretch')

        # Detailed metrics in expandable sections
        with st.expander("📋 Detailed Metrics", expanded=False):
            metrics_tabs = st.tabs([
                "Structure",
                "Data Richness",
                "Professional Sophistication",
                "Actionability",
                "Sentiment"
            ])

            with metrics_tabs[0]:
                st.json(eval_data['metrics']['structure'])

            with metrics_tabs[1]:
                st.json(eval_data['metrics']['data_richness'])

            with metrics_tabs[2]:
                st.json(eval_data['metrics']['sophistication'])

            with metrics_tabs[3]:
                st.json(eval_data['metrics']['actionability'])

            with metrics_tabs[4]:
                st.json(eval_data['metrics']['sentiment'])

        # Recommendations
        if eval_data['recommendations']:
            st.subheader("💡 Improvement Recommendations")

            for i, rec in enumerate(eval_data['recommendations'], 1):
                priority_emoji = {
                    'Critical': '🔴',
                    'High': '🟠',
                    'Medium': '🟡',
                    'Low': '🟢',
                    'Info': 'ℹ️'
                }.get(rec['priority'], 'ℹ️')

                with st.container():
                    st.markdown(f"**{priority_emoji} {rec['category']} ({rec['priority']} Priority)**")
                    st.markdown(f"*Issue:* {rec['issue']}")
                    st.markdown(f"*Suggestion:* {rec['suggestion']}")
                    if i < len(eval_data['recommendations']):
                        st.divider()

    st.divider()
    st.markdown(st.session_state.report)


if not st.session_state.get("stock_fig") and not st.session_state.get("report"):
    st.markdown(
        """
## AI-Powered Stock Analysis Platform

Welcome to stock analysis platform! It uses Artificial Intelligence and Large Language Models (LLMs) to provide professional investment insights.

**Key Features:**

* Analyzes sentiment from Reddit (wallstreetbets, stocks, investing subreddits).
* Performs detailed fundamental and technical analysis.
* Integrates research from the web and news sources.

To get a detailed, AI-generated report, select a stock symbol and provide Google Gemini API key. This platform is designed to help investors make data-driven decisions in the stock market.

**Disclaimer:** This analysis is for informational purposes only and is not financial or investment advice."""
    )

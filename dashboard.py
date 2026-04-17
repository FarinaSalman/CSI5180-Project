import time
import threading
from queue import Empty
import json
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update, ctx, ALL

from ui_bridge import ui_queue
from pipeline import (
    handle_text_bypass_input,
    handle_live_voice_pipeline,
    reset_pipeline_state,
)


BOOKS_PATH = Path(__file__).with_name("books.json")
with open(BOOKS_PATH, "r", encoding="utf-8") as f:
    BOOKS_DB = json.load(f)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

global server_ui_state, ui_state_lock


def send_text_to_pipeline(text: str):
    threading.Thread(
        target=handle_text_bypass_input,
        args=(text,),
        daemon=True,
    ).start()


def set_system_locked_data(is_locked: bool):
    return {"locked": is_locked}

def set_wake_state_data(is_awake: bool):
    return {"awake": is_awake}

def set_listening_data(is_listening: bool):
    return {"listening": is_listening}


def add_history_message(history: list, role: str, text: str):
    if not text or not text.strip():
        return history or []
    updated = list(history or [])
    updated.append({"role": role, "text": text.strip()})
    return updated


def start_timer_data(timer_data: dict, duration: int | None = None):
    timer_data = dict(timer_data or {})

    if duration is None or duration <= 0:
        timer_data["status"] = "stopped"
        timer_data["start_time"] = None
        return timer_data

    timer_data["duration"] = duration
    timer_data["elapsed"] = 0
    timer_data["status"] = "running"
    timer_data["start_time"] = time.time()
    return timer_data


def pause_timer_data(timer_data: dict):
    timer_data = dict(timer_data or {})
    if timer_data.get("status") == "running":
        now = time.time()
        start_time = timer_data.get("start_time", now)
        timer_data["elapsed"] = max(0, int(now - start_time))
        timer_data["status"] = "paused"
        timer_data["start_time"] = None
    return timer_data


def stop_timer_data(timer_data: dict):
    timer_data = dict(timer_data or {})
    timer_data["status"] = "stopped"
    timer_data["elapsed"] = 0
    timer_data["start_time"] = None
    return timer_data


def reset_timer_data(timer_data: dict):
    timer_data = dict(timer_data or {})
    timer_data["elapsed"] = 0
    timer_data["start_time"] = None
    timer_data["status"] = "stopped"
    return timer_data


def get_timer_display(timer_data: dict):
    timer_data = timer_data or {"status": "stopped", "elapsed": 0, "duration": 0, "start_time": None}

    duration = timer_data.get("duration", 0)
    elapsed = timer_data.get("elapsed", 0)

    if timer_data.get("status") == "running" and timer_data.get("start_time") is not None:
        elapsed = max(0, int(time.time() - timer_data["start_time"]))

    remaining = max(0, duration - elapsed)

    if remaining == 0 and timer_data.get("status") == "running":
        timer_data["status"] = "stopped"

    minutes = remaining // 60
    seconds = remaining % 60

    return f"{minutes:02d}:{seconds:02d}"


def open_book_data(title: str, pages: list[str], page_index: int = 0, current_state: dict | None = None):
    current_state = current_state or {}
    pages = pages or [""]

    page_index = max(0, min(page_index, len(pages) - 1))

    return {
        "is_open": True,
        "title": title.strip() if title else "Untitled",
        "pages": pages,
        "page_index": page_index,
        "font_size": current_state.get("font_size", 18),
        "brightness": current_state.get("brightness", 100),
        "dark_mode": current_state.get("dark_mode", False),
    }

def close_book_data(current_state: dict | None = None):
    current_state = current_state or {}
    return {
        "is_open": False,
        "title": "",
        "pages": [],
        "page_index": 0,
        "font_size": current_state.get("font_size", 18),
        "brightness": current_state.get("brightness", 100),
        "dark_mode": current_state.get("dark_mode", False),
    }


initial_system_state = {"locked": True}
initial_listening_state = {"listening": False}
initial_wake_state = {"awake": False}
initial_history = []
initial_timer_state = {
    "status": "stopped",
    "elapsed": 0,
    "duration": 0,
    "start_time": None,
}
initial_book_candidate_state = {
    "active": False,
    "query": "",
    "options": [],
    "page": 0,
    "total": 0,
}
initial_reader_state = close_book_data()


ui_state_lock = threading.Lock()

server_ui_state = {
    "system_state": initial_system_state,
    "listening_state": initial_listening_state,
    "wake_state": initial_wake_state,
    "history": initial_history,
    "timer_state": initial_timer_state,
    "reader_state": initial_reader_state,
    "book_candidate_state": dict(initial_book_candidate_state),
    "input_text": "",
}

app.layout = html.Div(
    className="bg-light",
    children=[
        dcc.Location(id="page-url", refresh=False),
        html.Div(id="page-load-trigger", style={"display": "none"}),

        dcc.Store(id="system-state", data=initial_system_state),
        dcc.Store(id="listening-state", data=initial_listening_state),
        dcc.Store(id="wake-state", data=initial_wake_state),
        dcc.Store(id="history-state", data=initial_history),
        dcc.Store(id="timer-state", data=initial_timer_state),
        dcc.Store(id="reader-state", data=initial_reader_state),
        

        dcc.Interval(id="timer-interval", interval=1000, n_intervals=0),
        dcc.Interval(id="bridge-poll", interval=100, n_intervals=0),

        dcc.Store(
            id="book-candidate-state",
            data={
                "active": False,
                "query": "",
                "options": [],
                "page": 0,
                "total": 0,
            },
        ),

        html.Div(
            className="container-fluid py-4",
            children=[
                html.Div(
                    className="text-center mb-4",
                    children=[
                         html.H1("Atlas", className="fw-bold"),
                        html.P("E-reader Interactive Dashboard", className="text-muted mb-0"),
                    ],
                ),
                html.Div(
                    className="row g-4 mb-4",
                    children=[
                        html.Div(
                            className="col-md-6",
                            children=[
                                html.Div(
                                    className="card shadow-sm border-0 h-100",
                                    children=[
                                        html.Div(
                                            className="card-body",
                                            children=[
                                                html.H4("Input", className="card-title mb-3"),
                                                html.Div(
                                                    className="input-group mb-3",
                                                    children=[
                                                        dcc.Textarea(
                                                            id="input-text",
                                                            placeholder="Enter text here",
                                                            className="form-control",
                                                            style={
                                                                "height": "130px",
                                                                "resize": "none",
                                                            },
                                                        )
                                                    ],
                                                ),
                                                html.Div(
                                                    className="d-flex gap-2 mb-4",
                                                    children=[
                                                        html.Button(
                                                            "Submit",
                                                            id="submit-button",
                                                            className="btn btn-primary px-4",
                                                            n_clicks=0,
                                                        ),
                                                        html.Button(
                                                            "Speak",
                                                            id="speak-button",
                                                            className="btn btn-outline-secondary px-4",
                                                            n_clicks=0,
                                                        ),
                                                    ],
                                                ),
                                                html.H5("Output History", className="mb-3"),
                                                html.Div(
                                                    id="output-history",
                                                    className="border rounded p-3 bg-light",
                                                    style={
                                                        "height": "220px",
                                                        "overflowY": "auto",
                                                        "textAlign": "left",
                                                        "fontSize": "0.95rem",
                                                    },
                                                ),
                                            ],
                                        )
                                    ],
                                )
                            ],
                        ),
                        html.Div(
                            className="col-md-6",
                            children=[
                                html.Div(
                                    className="card shadow-sm border-0 h-100",
                                    children=[
                                        html.Div(
                                            className="card-body text-center d-flex flex-column justify-content-center",
                                            children=[
                                                html.H4("System Status", className="card-title mb-3"),
                                                html.Div(
                                                    className="d-flex justify-content-center gap-3 mb-3 flex-wrap",
                                                    children=[
                                                        html.Span(
                                                            "SYSTEM LOCKED",
                                                            id="system-status-badge",
                                                            className="badge text-bg-danger fs-6 px-3 py-2",
                                                        ),
                                                        html.Span(
                                                            "NOT LISTENING",
                                                            id="listening-status-badge",
                                                            className="badge text-bg-secondary fs-6 px-3 py-2",
                                                        ),
                                                        html.Span(
                                                            "ASLEEP",
                                                            id="wake-status-badge",
                                                            className="badge text-bg-dark fs-6 px-3 py-2",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="mb-4",
                                                    children=[
                                                        html.Span(
                                                            "OpenLibrary API",
                                                            className="badge rounded-pill text-bg-info me-2 px-3 py-2",
                                                        ),
                                                        html.Span(
                                                            "Open-Meteo API",
                                                            className="badge rounded-pill text-bg-info px-3 py-2",
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="mt-4",
                                                    children=[
                                                        html.P("Timer", className="text-muted mb-2"),
                                                        html.Div(
                                                            className="d-inline-block px-4 py-2 rounded bg-light border",
                                                            children=[
                                                                html.H2(
                                                                    id="timer-display",
                                                                    className="fw-semibold text-dark mb-0",
                                                                    children="00:00",
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        )
                                    ],
                                )
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="row g-4 mb-4",
                    children=[
                        html.Div(
                            className="col-12",
                            children=[
                                html.Div(
                                    className="card shadow-sm border-0",
                                    children=[
                                        html.Div(
                                            className="card-body",
                                            children=[
                                                html.H4("Book Search Matches", className="mb-3"),
                                                html.Div(
                                                    id="book-candidate-panel",
                                                    className="text-muted",
                                                    children="No book disambiguation needed",
                                                ),
                                            ],
                                        )
                                    ],
                                )
                            ],
                        )
                    ],
                ),
                html.Div(
                    className="row",
                    children=[
                        html.Div(
                            className="col-md-4",
                            children=[
                                html.Div(
                                    className="card shadow-sm border-0 h-100",
                                    children=[
                                        html.Div(
                                            className="card-body",
                                            children=[
                                                html.H4("Available Books", className="mb-3"),
                                                html.Div(id="book-list", children="No books loaded"),
                                            ],
                                        )
                                    ],
                                )
                            ],
                        ),
                        html.Div(
                            className="col-md-8",
                            children=[
                                html.Div(
                                    className="card shadow-sm border-0",
                                    children=[
                                        html.Div(
                                            className="card-body",
                                            children=[
                                                html.Div(
                                                    className="d-flex justify-content-between align-items-center mb-3 flex-wrap gap-2",
                                                    children=[
                                                        html.Div(id="reader-title", className="fw-semibold", children="E-Reader"),
                                                        html.Div(
                                                            className="d-flex gap-2 flex-wrap",
                                                            children=[
                                                                html.Button("A-", id="font-decrease", className="btn btn-outline-secondary btn-sm"),
                                                                html.Button("A+", id="font-increase", className="btn btn-outline-secondary btn-sm"),
                                                                html.Button("Brightness -", id="brightness-decrease", className="btn btn-outline-secondary btn-sm"),
                                                                html.Button("Brightness +", id="brightness-increase", className="btn btn-outline-secondary btn-sm"),
                                                                html.Button("Toggle Theme", id="toggle-theme", className="btn btn-outline-secondary btn-sm"),
                                                                html.Button("Close Book", id="close-book-btn", className="btn btn-outline-danger btn-sm"),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                html.Div(
                                                    className="d-flex justify-content-between align-items-center mb-3",
                                                    children=[
                                                        html.Button("Previous Page", id="prev-page", className="btn btn-outline-primary btn-sm"),
                                                        html.Div(id="page-indicator", className="text-muted", children="Page 0 / 0"),
                                                        html.Button("Next Page", id="next-page", className="btn btn-outline-primary btn-sm"),
                                                    ],
                                                ),
                                                html.Div(
                                                    id="ereader-panel",
                                                    className="p-4 rounded border",
                                                    children="No book open",
                                                    style={
                                                        "height": "420px",
                                                        "overflow": "hidden",
                                                        "lineHeight": "1.9",
                                                        "fontSize": "18px",
                                                        "backgroundColor": "#f6f3ea",
                                                        "color": "#2e2e2e",
                                                        "fontFamily": "Georgia, 'Times New Roman', serif",
                                                    },
                                                ),
                                            ],
                                        )
                                    ],
                                )
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

@app.callback(
    Output("system-status-badge", "children"),
    Output("system-status-badge", "className"),
    Input("system-state", "data"),
)
def render_system_badge(system_state):
    locked = (system_state or {}).get("locked", False)
    if locked:
        return "SYSTEM LOCKED", "badge text-bg-danger fs-6 px-3 py-2"
    return "SYSTEM UNLOCKED", "badge text-bg-success fs-6 px-3 py-2"


@app.callback(
    Output("listening-status-badge", "children"),
    Output("listening-status-badge", "className"),
    Input("listening-state", "data"),
)
def render_listening_badge(listening_state):
    listening = (listening_state or {}).get("listening", False)
    if listening:
        return "LISTENING", "badge text-bg-warning fs-6 px-3 py-2"
    return "NOT LISTENING", "badge text-bg-secondary fs-6 px-3 py-2"


@app.callback(
    Output("wake-status-badge", "children"),
    Output("wake-status-badge", "className"),
    Input("wake-state", "data"),
)
def render_wake_badge(wake_state):
    awake = (wake_state or {}).get("awake", False)
    if awake:
        return "AWAKE", "badge text-bg-info fs-6 px-3 py-2"
    return "ASLEEP", "badge text-bg-dark fs-6 px-3 py-2"

@app.callback(
    Output("book-candidate-panel", "children"),
    Input("book-candidate-state", "data"),
)
def render_book_candidate_panel(candidate_state):
    candidate_state = candidate_state or {}
    active = candidate_state.get("active", False)

    if not active:
        return html.Div(
            "No book disambiguation needed",
            className="text-muted",
        )

    query = candidate_state.get("query", "")
    options = candidate_state.get("options", [])
    page = candidate_state.get("page", 0)
    total = candidate_state.get("total", 0)

    if not options:
        return html.Div(
            "No candidate options available",
            className="text-muted",
        )

    children = [
        html.P(
            f"I found multiple matches for '{query}'. Please choose one.",
            className="mb-3",
        )
    ]

    start_number = page * 5 + 1

    for i, option in enumerate(options):
        title = option.get("title", "Unknown title")
        author = option.get("author", "Unknown author")
        year = option.get("year", "Unknown year")

        children.append(
            html.Button(
                f"{start_number + i}. {title} — {author} — {year}",
                id={"type": "candidate-select", "index": i},
                n_clicks=0,
                className="btn btn-outline-primary w-100 text-start mb-2",
            )
        )

    controls = [
        html.Button(
            "None of these — Show more",
            id="candidate-next-page",
            n_clicks=0,
            className="btn btn-outline-secondary me-2",
        ),
        html.Button(
            "Cancel",
            id="candidate-cancel",
            n_clicks=0,
            className="btn btn-outline-danger",
        ),
    ]

    children.append(html.Div(controls, className="mt-3"))
    children.append(
        html.Div(
            f"Showing page {page + 1} of {max(1, (total + 4) // 5)}",
            className="text-muted mt-2",
        )
    )

    return children

@app.callback(
    Output("input-text", "value", allow_duplicate=True),
    Input({"type": "candidate-select", "index": ALL}, "n_clicks"),
    Input("candidate-next-page", "n_clicks"),
    Input("candidate-cancel", "n_clicks"),
    State("book-candidate-state", "data"),
    prevent_initial_call=True,
)
def handle_book_candidate_actions(candidate_clicks, next_page_clicks, cancel_clicks, candidate_state):
    candidate_state = candidate_state or {}
    if not candidate_state.get("active", False):
        return no_update
    
    triggered = ctx.triggered_id

    if triggered is None:
        return no_update

    # Candidate button clicked
    if isinstance(triggered, dict) and triggered.get("type") == "candidate-select":
        idx = triggered["index"]

        if not candidate_clicks:
            return no_update

        if idx >= len(candidate_clicks):
            return no_update

        if not candidate_clicks[idx]:
            return no_update

        send_text_to_pipeline(f"__select_candidate__:{idx}")
        return ""

    # Next page clicked
    if triggered == "candidate-next-page":
        if not next_page_clicks:
            return no_update

        send_text_to_pipeline("__next_candidate_page__")
        return ""

    # Cancel clicked
    if triggered == "candidate-cancel":
        if not cancel_clicks:
            return no_update

        send_text_to_pipeline("__cancel_candidate_selection__")
        return ""

    return no_update

# @app.callback(
#     Output("output-history", "children"),
#     Input("history-state", "data"),
# )
# def render_output_history(history):
#     history = history or []

#     if len(history) == 0:
#         return html.Div(
#             className="h-100 d-flex justify-content-center align-items-center text-muted",
#             children="No history yet",
#         )

#     items = []
#     for msg in history:
#         role = msg.get("role", "").lower()
#         text = msg.get("text", "")

#         if role == "user":
#             role_label = html.Div("User", className="fw-semibold text-primary")
#         else:
#             role_label = html.Div("Assistant", className="fw-semibold text-success")

#         items.append(
#             html.Div(
#                 className="mb-3",
#                 children=[
#                     role_label,
#                     html.Div(text, className="mb-2"),
#                 ],
#             )
#         )

#     return items


def build_output_history_children(history):
    history = history or []

    if len(history) == 0:
        return html.Div(
            className="h-100 d-flex justify-content-center align-items-center text-muted",
            children="No history yet",
        )

    items = []
    for msg in history:
        role = msg.get("role", "").lower()
        text = msg.get("text", "")

        if role == "user":
            role_label = html.Div("User", className="fw-semibold text-primary")
        else:
            role_label = html.Div("Assistant", className="fw-semibold text-success")

        items.append(
            html.Div(
                className="mb-3",
                children=[
                    role_label,
                    html.Div(text, className="mb-2"),
                ],
            )
        )

    return items


@app.callback(
    Output("book-list", "children"),
    Input("reader-state", "data"),
)
def render_book_list(reader_state):
    reader_state = reader_state or close_book_data()
    current_title = reader_state.get("title", "")
    is_open = reader_state.get("is_open", False)

    items = []

    for title in sorted(BOOKS_DB.keys()):
        active = is_open and title == current_title
        items.append(
            html.Button(
                title,
                id={"type": "book-select", "title": title},
                n_clicks=0,
                className="btn w-100 text-start mb-2 " + (
                    "btn-primary" if active else "btn-outline-secondary"
                ),
            )
        )

    return items

@app.callback(
    Output("timer-display", "children"),
    Input("timer-state", "data"),
    Input("timer-interval", "n_intervals"),
)
def render_timer(timer_state, _):
    return get_timer_display(timer_state)


@app.callback(
    Output("page-load-trigger", "children"),
    Input("page-url", "pathname"),
    prevent_initial_call=False,
)
def reset_on_page_load(_pathname):
    reset_pipeline_state()
    return ""


@app.callback(
    Output("ereader-panel", "children"),
    Output("ereader-panel", "style"),
    Output("reader-title", "children"),
    Output("page-indicator", "children"),
    Input("reader-state", "data"),
)
def render_ereader(reader_state):
    reader_state = reader_state or close_book_data()

    base_style = {
        "height": "420px",
        "overflowY": "auto",
        "lineHeight": "1.9",
        "borderRadius": "0.5rem",
        "fontFamily": "Georgia, 'Times New Roman', serif",
    }

    dark_mode = reader_state.get("dark_mode", False)
    font_size = reader_state.get("font_size", 18)
    brightness = reader_state.get("brightness", 100)

    style = {
        **base_style,
        "backgroundColor": "#1e1e1e" if dark_mode else "#f6f3ea",
        "color": "#f5f5f5" if dark_mode else "#2e2e2e",
        "fontSize": f"{font_size}px",
        "filter": f"brightness({brightness}%)",
    }

    if not reader_state.get("is_open", False):
        return (
            html.Div(
                "No book open",
                className="h-100 d-flex justify-content-center align-items-center text-muted",
            ),
            style,
            "E-Reader",
            "Page 0 / 0",
        )

    pages = reader_state.get("pages", [])
    if not pages:
        return (
            html.Div(
                "No pages available",
                className="h-100 d-flex justify-content-center align-items-center text-muted",
            ),
            style,
            reader_state.get("title", "Untitled"),
            "Page 0 / 0",
        )

    page_index = reader_state.get("page_index", 0)
    page_index = max(0, min(page_index, len(pages) - 1))

    current_page_text = pages[page_index]
    content = html.Div([
        html.P(p.strip(), className="mb-3")
        for p in current_page_text.split("\n")
        if p.strip()
    ])

    return (
        content,
        style,
        reader_state.get("title", "Untitled"),
        f"Page {page_index + 1} / {len(pages)}",
    )

@app.callback(
    Output("submit-button", "disabled", allow_duplicate=True),
    Input("speak-button", "n_clicks"),
    prevent_initial_call=True,
)
def handle_speak(_n_clicks):
    threading.Thread(
        target=handle_live_voice_pipeline,
        daemon=True,
    ).start()
    return False


@app.callback(
    Output("input-text", "value", allow_duplicate=True),
    Input("submit-button", "n_clicks"),
    State("input-text", "value"),
    prevent_initial_call=True,
)
def handle_submit(_n_clicks, input_text):
    if not input_text or not input_text.strip():
        return ""

    cleaned = input_text.strip()

    with ui_state_lock:
        server_ui_state["input_text"] = ""


    send_text_to_pipeline(cleaned)
    return ""


@app.callback(
    Output("reader-state", "data", allow_duplicate=True),
    Input({"type": "book-select", "title": ALL}, "n_clicks"),
    Input("prev-page", "n_clicks"),
    Input("next-page", "n_clicks"),
    Input("font-decrease", "n_clicks"),
    Input("font-increase", "n_clicks"),
    Input("brightness-decrease", "n_clicks"),
    Input("brightness-increase", "n_clicks"),
    Input("toggle-theme", "n_clicks"),
    Input("close-book-btn", "n_clicks"),
    State("reader-state", "data"),
    prevent_initial_call=True,
)
def update_reader_state(
    book_clicks,
    prev_clicks,
    next_clicks,
    fd_clicks,
    fi_clicks,
    bd_clicks,
    bi_clicks,
    theme_clicks,
    close_book_clicks,
    reader_state,
):
    reader_state = dict(reader_state or close_book_data())
    triggered = ctx.triggered_id

    if isinstance(triggered, dict) and triggered.get("type") == "book-select":
        if not book_clicks or max([c or 0 for c in book_clicks]) == 0:
            return no_update

        title = triggered["title"]
        book = BOOKS_DB[title]
        return open_book_data(
            title,
            book["pages"],
            page_index=0,
            current_state=reader_state,
        )

    if triggered == "close-book-btn":
        return close_book_data(current_state=reader_state)

    if not reader_state.get("is_open", False):
        return no_update

    if triggered == "prev-page":
        reader_state["page_index"] = max(0, reader_state["page_index"] - 1)
    elif triggered == "next-page":
        reader_state["page_index"] = min(len(reader_state["pages"]) - 1, reader_state["page_index"] + 1)
    elif triggered == "font-decrease":
        reader_state["font_size"] = max(12, reader_state.get("font_size", 18) - 2)
    elif triggered == "font-increase":
        reader_state["font_size"] = min(36, reader_state.get("font_size", 18) + 2)
    elif triggered == "brightness-decrease":
        reader_state["brightness"] = max(40, reader_state.get("brightness", 100) - 10)
    elif triggered == "brightness-increase":
        reader_state["brightness"] = min(140, reader_state.get("brightness", 100) + 10)
    elif triggered == "toggle-theme":
        reader_state["dark_mode"] = not reader_state.get("dark_mode", False)

    return reader_state


@app.callback(
    Output("system-state", "data"),
    Output("listening-state", "data"),
    Output("wake-state", "data"),
    Output("history-state", "data"),
    Output("timer-state", "data"),
    Output("reader-state", "data"),
    Output("input-text", "value", allow_duplicate=True),
    Output("output-history", "children"),
    Output("book-candidate-state", "data"),
    Input("bridge-poll", "n_intervals"),
    prevent_initial_call=True,
)
def process_bridge_events(_n_intervals):
    updated = False

    with ui_state_lock:
        while True:
            try:
                event = ui_queue.get_nowait()
            except Empty:
                break

            event_type = event.get("type")
            payload = event.get("payload") or {}

            if event_type == "set_locked":
                server_ui_state["system_state"] = set_system_locked_data(payload.get("locked", False))
                updated = True

            elif event_type == "set_listening":
                server_ui_state["listening_state"] = set_listening_data(payload.get("listening", False))
                updated = True

            elif event_type == "set_awake":
                server_ui_state["wake_state"] = set_wake_state_data(payload.get("awake", False))
                updated = True

            elif event_type == "add_user_message":
                server_ui_state["history"] = add_history_message(
                    server_ui_state["history"], "user", payload.get("text", "")
                )
                updated = True

            elif event_type == "add_assistant_message":
                server_ui_state["history"] = add_history_message(
                    server_ui_state["history"], "assistant", payload.get("text", "")
                )
                updated = True

            elif event_type == "clear_history":
                server_ui_state["history"] = []
                updated = True

            elif event_type == "start_timer":
                duration = payload.get("duration")
                server_ui_state["timer_state"] = start_timer_data(server_ui_state["timer_state"], duration)
                updated = True

            elif event_type == "pause_timer":
                server_ui_state["timer_state"] = pause_timer_data(server_ui_state["timer_state"])
                updated = True

            elif event_type == "stop_timer":
                server_ui_state["timer_state"] = stop_timer_data(server_ui_state["timer_state"])
                updated = True

            elif event_type == "reset_timer":
                server_ui_state["timer_state"] = reset_timer_data(server_ui_state["timer_state"])
                updated = True

            elif event_type == "open_book":
                title = payload.get("title", "Untitled")
                if title in BOOKS_DB:
                    server_ui_state["reader_state"] = open_book_data(
                        title,
                        BOOKS_DB[title]["pages"],
                        page_index=payload.get("page_index", 0),
                        current_state=server_ui_state["reader_state"],
                    )
                else:
                    server_ui_state["reader_state"] = open_book_data(
                        title,
                        payload.get("pages", []) or [payload.get("content", "")],
                        page_index=payload.get("page_index", 0),
                        current_state=server_ui_state["reader_state"],
                    )
                updated = True

            elif event_type == "close_book":
                server_ui_state["reader_state"] = close_book_data(server_ui_state["reader_state"])
                updated = True

            elif event_type == "next_page":
                reader_state = dict(server_ui_state["reader_state"])
                if reader_state.get("is_open") and reader_state.get("pages"):
                    last_page = len(reader_state["pages"]) - 1
                    reader_state["page_index"] = min(last_page, reader_state["page_index"] + 1)
                    server_ui_state["reader_state"] = reader_state
                    updated = True

            elif event_type == "prev_page":
                reader_state = dict(server_ui_state["reader_state"])
                if reader_state.get("is_open") and reader_state.get("pages"):
                    reader_state["page_index"] = max(0, reader_state["page_index"] - 1)
                    server_ui_state["reader_state"] = reader_state
                    updated = True

            elif event_type == "increase_font_size":
                reader_state = dict(server_ui_state["reader_state"])
                reader_state["font_size"] = min(36, reader_state.get("font_size", 18) + 2)
                server_ui_state["reader_state"] = reader_state
                updated = True

            elif event_type == "decrease_font_size":
                reader_state = dict(server_ui_state["reader_state"])
                reader_state["font_size"] = max(12, reader_state.get("font_size", 18) - 2)
                server_ui_state["reader_state"] = reader_state
                updated = True

            elif event_type == "increase_brightness":
                reader_state = dict(server_ui_state["reader_state"])
                reader_state["brightness"] = min(140, reader_state.get("brightness", 100) + 10)
                server_ui_state["reader_state"] = reader_state
                updated = True

            elif event_type == "decrease_brightness":
                reader_state = dict(server_ui_state["reader_state"])
                reader_state["brightness"] = max(40, reader_state.get("brightness", 100) - 10)
                server_ui_state["reader_state"] = reader_state
                updated = True

            elif event_type == "toggle_reader_theme":
                reader_state = dict(server_ui_state["reader_state"])
                reader_state["dark_mode"] = not reader_state.get("dark_mode", False)
                server_ui_state["reader_state"] = reader_state
                updated = True

            elif event_type == "set_input_text":
                server_ui_state["input_text"] = payload.get("text", "")
                updated = True

            elif event_type == "show_book_candidates":
                server_ui_state["book_candidate_state"] = {
                    "active": True,
                    "query": payload.get("query", ""),
                    "options": payload.get("options", []),
                    "page": payload.get("page", 0),
                    "total": payload.get("total", 0),
                }
                updated = True

            elif event_type == "clear_book_candidates":
                server_ui_state["book_candidate_state"] = {
                    "active": False,
                    "query": "",
                    "options": [],
                    "page": 0,
                    "total": 0,
                }
                updated = True

        if not updated:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )

        history_copy = list(server_ui_state["history"])

        return (
            dict(server_ui_state["system_state"]),
            dict(server_ui_state["listening_state"]),
            dict(server_ui_state["wake_state"]),
            history_copy,
            dict(server_ui_state["timer_state"]),
            dict(server_ui_state["reader_state"]),
            server_ui_state["input_text"],
            build_output_history_children(history_copy),
            dict(server_ui_state["book_candidate_state"]),
        )


if __name__ == "__main__":
    reset_pipeline_state()
    app.run(debug=True, use_reloader=False)
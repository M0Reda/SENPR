"""
app.py -- Stage 8: Gradio Chat UI
===================================
A local chat interface for the EduBot academic advisor.

Run:
    cd edubot
    python src/app.py

Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str((ROOT / "../EduBot_Dataset_v1.2/scripts").resolve()))

cfg: dict[str, Any] = yaml.safe_load((ROOT / "config.yaml").read_text())
DATA_DIR = (ROOT / cfg["data_dir"]).resolve()

# Lazy bot instance per session
_active_bots: dict[str, Any] = {}


def _get_bot(sid: str):
    from bot import _get_db
    _get_db(sid)  # reload data if this is a newly created student
    if sid not in _active_bots:
        from bot import EduBot
        _active_bots[sid] = EduBot(session_sid=sid)
    return _active_bots[sid]


def _prefill_async(sid: str) -> None:
    """After login, warm Ollama's cache with this student's prompt in the background."""
    import threading

    def _run():
        try:
            _get_bot(sid).prefill()
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True, name=f"prefill-{sid}").start()


def _sidebar_info(sid: str) -> str:
    """Return HTML for the student info sidebar."""
    try:
        from bot import _get_db
        db = _get_db(sid)
        if sid not in db.students:
            return "<p style='color:red'>Student ID not found.</p>"
        from tools import StudentTools
        t = StudentTools(db, session_sid=sid)
        p = t.get_student_profile()
        a = t.get_advisor()
        gpa = p.get("cumulative_gpa") or "N/A"
        standing = p.get("standing", "")
        gpa_color = (
            "#4CAF50" if float(gpa or 0) >= 3.0
            else "#FF9800" if float(gpa or 0) >= 2.0
            else "#f44336"
        )
        return f"""
        <div style="font-family: 'Inter', sans-serif; padding: 12px;">
          <h3 style="margin:0 0 8px; color:#e0e0e0;">{p['name']}</h3>
          <p style="margin:2px 0; color:#aaa;">ID: {p['student_id']}</p>
          <hr style="border-color:#333; margin:8px 0;">
          <table style="width:100%; font-size:13px; color:#ccc;">
            <tr><td>Semester Level</td><td><b>{p['semester_level']}</b></td></tr>
            <tr><td>Credits</td><td><b>{p['credits_completed']} / 180</b></td></tr>
            <tr><td>GPA</td>
                <td><b style="color:{gpa_color}">{gpa}</b></td></tr>
            <tr><td>Standing</td><td><b>{standing}</b></td></tr>
            <tr><td>Max Load</td><td><b>{p['max_load_hours']} cr/term</b></td></tr>
          </table>
          <hr style="border-color:#333; margin:8px 0;">
          <p style="margin:2px 0; font-size:12px; color:#aaa;">
            <b>Advisor:</b> {a['name']}<br>
            <b>Office:</b> {a['office']}
          </p>
        </div>
        """
    except Exception as e:
        return f"<p style='color:orange'>Could not load profile: {e}</p>"


def login(student_id: str):
    """Validate the student ID and return the main chat interface."""
    sid = str(student_id).strip()
    try:
        from bot import _get_db
        db = _get_db(sid)
        if sid not in db.students:
            return (
                gr.update(visible=True),   # login panel
                gr.update(visible=False),  # chat panel
                gr.update(value="❌ Student ID not found. Please check and try again.",
                          visible=True),
                sid,
                "",
            )
        sidebar = _sidebar_info(sid)
        return (
            gr.update(visible=False),  # login panel
            gr.update(visible=True),   # chat panel
            gr.update(visible=False),  # error
            sid,
            sidebar,
        )
    except Exception as e:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=f"⚠️ Error: {e}", visible=True),
            "",
            "",
        )


def chat_fn(message: str, history: list[dict[str, str]], session_sid: str):
    """Process one chat message and yield the response."""
    history = history or []
    if not session_sid:
        new_hist = history + [{"role": "assistant", "content": "Please log in first."}]
        yield new_hist, new_hist
        return
    if not message.strip():
        yield history, history
        return

    bot = _get_bot(session_sid)
    new_hist = history + [{"role": "user", "content": message}, {"role": "assistant", "content": "Thinking..."}]
    yield new_hist, new_hist
    try:
        for partial in bot.chat_stream(message):
            new_hist[-1]["content"] = partial
            yield new_hist, new_hist
    except Exception as e:
        new_hist[-1]["content"] = (
            f"I'm having some technical trouble right now — please try again in a moment. "
            f"If this keeps happening, contact your advisor. (Error: {e})"
        )
        yield new_hist, new_hist


def clear_fn(session_sid: str):
    """Clear the conversation history."""
    if session_sid and session_sid in _active_bots:
        _active_bots[session_sid].clear_history()
    return [], []


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: #0d0d0d !important;
    color: #e0e0e0 !important;
}

.main-header {
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #006064 100%);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 4px 32px rgba(0,100,200,0.3);
}

.main-header h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 600;
    letter-spacing: -0.5px;
    color: #ffffff;
}

.main-header p {
    margin: 6px 0 0;
    color: rgba(255,255,255,0.75);
    font-size: 0.95rem;
}

.sidebar-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 0;
    overflow: hidden;
}

.chat-box {
    background: #111111 !important;
    border: 1px solid #222 !important;
    border-radius: 12px !important;
}

.message.user {
    background: linear-gradient(135deg, #1a237e, #0d47a1) !important;
    border-radius: 12px 12px 4px 12px !important;
    color: #fff !important;
}

.message.bot {
    background: #1e1e2e !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 12px 12px 12px 4px !important;
    color: #e0e0e0 !important;
}

.login-box {
    max-width: 480px;
    margin: 60px auto;
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 40px;
    box-shadow: 0 8px 48px rgba(0,0,0,0.5);
}

button.primary-btn {
    background: linear-gradient(135deg, #1a237e, #006064) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    cursor: pointer;
    transition: opacity 0.2s;
}

button.primary-btn:hover { opacity: 0.85; }

.send-btn {
    background: linear-gradient(135deg, #0d47a1, #006064) !important;
}

footer { display: none !important; }
"""


# ── Smart auto-scroll ────────────────────────────────────────────────────────
# Follow the streaming answer only while the user is at (or near) the bottom.
# If they scroll up to read earlier messages, leave the scroll position alone;
# scrolling back down re-enables following.

SMART_SCROLL_JS = """
() => {
  const NEAR = 80;                       // px from bottom that still counts as "at bottom"
  const follow = new WeakMap();          // scroll container -> should we follow?
  const inChat = (el) => el && el.closest && el.closest('.chat-box');

  document.addEventListener('scroll', (e) => {
    const el = e.target;
    if (el && el.nodeType === 1 && inChat(el)) {
      follow.set(el, el.scrollHeight - el.scrollTop - el.clientHeight < NEAR);
    }
  }, true);

  let queued = false;
  const tick = () => {
    queued = false;
    document.querySelectorAll('.chat-box *').forEach((el) => {
      if (el.scrollHeight > el.clientHeight + 5 &&
          /(auto|scroll)/.test(getComputedStyle(el).overflowY) &&
          follow.get(el) !== false) {
        el.scrollTop = el.scrollHeight;
      }
    });
  };
  new MutationObserver(() => {
    if (!queued) { queued = true; requestAnimationFrame(tick); }
  }).observe(document.body, { childList: true, subtree: true, characterData: true });
}
"""


# ── Build Gradio app ──────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="EduBot — Academic Advisor") as app:

        session_sid = gr.State("")

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="main-header">
          <h1>🎓 EduBot Academic Advisor</h1>
          <p>Computer Engineering Department · AASTMT · Your AI Academic Assistant</p>
        </div>
        """)

        # ── Login panel ─────────────────────────────────────────────────────
        with gr.Group(visible=True, elem_classes=["login-box"]) as login_panel:
            gr.Markdown("### Welcome — please enter your Student ID to begin.")
            sid_input = gr.Textbox(
                label="Student ID",
                placeholder="e.g. 201003465",
                lines=1,
                max_lines=1,
            )
            login_btn = gr.Button("Sign In", elem_classes=["primary-btn"])
            login_err = gr.HTML(visible=False)

        # ── Chat panel ──────────────────────────────────────────────────────
        with gr.Group(visible=False) as chat_panel:
            with gr.Row():
                # Sidebar
                with gr.Column(scale=1, min_width=240):
                    sidebar_html = gr.HTML(elem_classes=["sidebar-card"])
                    clear_btn = gr.Button("🗑️ Clear History", size="sm")

                # Chat
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="",
                        height=520,
                        elem_classes=["chat-box"],
                        autoscroll=False,   # we follow the stream ourselves (see SMART_SCROLL_JS)
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask about your GPA, courses, graduation plan...",
                            lines=1,
                            show_label=False,
                            scale=5,
                        )
                        send_btn = gr.Button("Send ➤", scale=1,
                                             elem_classes=["primary-btn", "send-btn"])

                    gr.HTML("""
                    <p style="font-size:11px; color:#555; margin-top:4px;">
                    EduBot may make mistakes. Always verify important decisions with your
                    academic advisor.
                    </p>
                    """)

        # ── Wire events ──────────────────────────────────────────────────────
        history_state = gr.State([])

        login_btn.click(
            login,
            inputs=[sid_input],
            outputs=[login_panel, chat_panel, login_err, session_sid, sidebar_html],
        )
        sid_input.submit(
            login,
            inputs=[sid_input],
            outputs=[login_panel, chat_panel, login_err, session_sid, sidebar_html],
        )
        send_btn.click(
            chat_fn,
            inputs=[msg_input, history_state, session_sid],
            outputs=[chatbot, history_state],
        ).then(lambda: "", outputs=[msg_input])

        msg_input.submit(
            chat_fn,
            inputs=[msg_input, history_state, session_sid],
            outputs=[chatbot, history_state],
        ).then(lambda: "", outputs=[msg_input])

        clear_btn.click(
            clear_fn,
            inputs=[session_sid],
            outputs=[chatbot, history_state],
        )

    return app


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    print("\nStarting EduBot UI ...")
    print(f"Open http://localhost:{server_port} in your browser.\n")
    import threading
    from bot import warmup
    threading.Thread(target=warmup, daemon=True, name="edubot-warmup").start()
    app = build_app()
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        show_error=True,
        css=CSS,
        js=SMART_SCROLL_JS,
    )

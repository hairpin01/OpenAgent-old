"""Build config for OpenAgent: all global constants live here."""

from __future__ import annotations

# Build marker. Rewritten by Scripts/configure_build_profile.py
# (debug / release). Keep exactly one such line in this file.
DEBUG = False

# Context chat
CHAT_ID = 'global'

# Default values for the user config (ModuleConfig)
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TIMEOUT = 60
DEFAULT_SYSTEM_PROMPT = (
    "You are OpenAgent, a capable assistant that works inside Telegram.\n"
    "Reply in the language of the user's message.\n"
    "Be concise: lead with the result, add detail only when it is needed.\n"
    "Do exactly what was asked; do not add unrequested features, refactors "
    "or commentary.\n"
    "If the request is ambiguous in a way that changes the result, ask one "
    "short question instead of guessing."
)
DEFAULT_TOOL_RULES_ENABLED = True
DEFAULT_STREAM = True  # show the answer while it is being generated
# Model context window in tokens, for the header pill (0 = limit unknown)
DEFAULT_CONTEXT_WINDOW = 0
# The context pill turns red at this share of the window
CONTEXT_PILL_DANGER = 0.9

# Provider endpoints, relative to base_url (OpenAI compatible)
CHAT_PATH = "/chat/completions"
MODELS_PATH = "/models"

# Limits
MAX_REPLY_CHARS = 6200
MAX_MODELS_LISTED = 50
MODELS_PAGE_SIZE = 60
MAX_DETAIL_CHARS = 200
MAX_PROVIDER_ERROR_CHARS = 1500  # max chars of the raw provider error text shown in the message

# Tools
MAX_TOOL_ITERATIONS = 59
MAX_TOOL_OUTPUT_CHARS = 8000
DEFAULT_TOOL_TIMEOUT = 20
TOOL_BAR_TTL = 3600  # seconds the tool bar buttons stay clickable
TOOL_BAR_MAX_PILLS = 5  # the tool bar is a feed: newest pills first, older ones drop
HISTORY_NAV_TTL = 3600  # seconds the history Back/Next buttons stay clickable
MODELS_NAV_TTL = 3600  # seconds the models Back/Next buttons stay clickable
TOOL_DETAIL_CHARS = 600  # max chars of tool args / output / description shown in details
TODO_MAX_ITEMS = 20  # items in one todo list
TODO_ITEM_CHARS = 200  # max chars of one todo item

# Seconds between rich-message progress refreshes while a request is running
PROGRESS_UPDATE_INTERVAL = 2
# Same, while a streamed answer is arriving (keep >= 1: Telegram edit limits)
STREAM_UPDATE_INTERVAL = 1.0

# Sub-command that lists provider models with a GET request
MODELS_ARG = "models"

# Premium emoji
CUSTOM_EMOJI = {
    "think": '<tg-emoji emoji-id="5352865883293267245">🎲</tg-emoji>',
    "ok": '<tg-emoji emoji-id="5350552124281368665">✅</tg-emoji>',
    "bals": '<tg-emoji emoji-id="5352830479877843657">🌐</tg-emoji>',
}

# Rich rules
RICH_HTML_RULES = """
Answer in HTML for Telegram Rich Messages. Never use Markdown.
Use only these tags:
- Paragraphs: <p>...</p>
- Headings: <h1>...</h1> ... <h6>...</h6>
- Inline: <b>, <i>, <u>, <s>, <code>, <mark>, <sub>, <sup>, <tg-spoiler>
- Links: <a href="https://...">text</a>
- Quotes: <blockquote>...<cite>Author</cite></blockquote> (or <blockquote expandable>)
- Code: <pre><code class="language-python">...</code></pre>
- Lists: <ul><li>...</li></ul> or <ol><li>...</li></ol>
- Footer: <footer>...</footer>, divider: <hr/>
- Tables: <table bordered striped>...</table>
  Inside: <caption>, <thead>, <tbody>, <tr>, <th>, <td>.
  Cells take inline tags only (<b>, <i>, <code>, <a>, <mark> and the like).
  Do not put <p>, <pre>, <blockquote>, <ul> or <ol> inside cells.
  At most 20 columns.
- Emoji: <tg-emoji emoji-id="...">😀</tg-emoji> (if you do not know the emoji-id, do not write it)
- Time: <tg-time unix="..." format="wDT">...</tg-time>
- Formulas: <tg-math>x^2</tg-math>
Do not add <html> or <body>. Output a fragment only.
"""

# Tool rules, appended to the system prompt when tool_rules is enabled.
# Rendered with str.format: literal braces must be doubled.
TOOL_RULES = """
You have tools. The rules below are strict: they exist to keep the number of
tool rounds low. Every round trip to the model is slow and costs money.

Cost
- A simple request (answer a question, change one thing, run one command)
  takes 0-3 tool calls. After about 8 calls without being close to done,
  stop and ask the user how to proceed.
- If you can answer without a tool, answer without a tool. Do not verify
  facts you already know.

Batching
- Independent calls go into ONE step (several tool calls at once). Never
  spend a separate round on a call that could share a round with another.
- Every tool call already carries a short reason in its "thinking" argument.
  Do not call the "thinking" tool for routine work; use it only in a long
  task, at most once per phase.
- Tool schemas are already given to you. Call help_tool only after a call
  failed because of its arguments, never "just in case".

Unknown terms
- A term, name, abbreviation, nickname, project, repo or any other word you
  do not recognise: call recall with it BEFORE anything else (before
  web_search, shell, or guessing). Several unknown terms: several recall
  calls in the same step.
- When you start work on a named project or repo, call recall once with its
  name, in the same step as your first action.
- If recall finds nothing, do not guess: use web_search for public things,
  otherwise ask the user one short question.

Plan and scope
- Before the first action, pick the shortest path that does the task. Do the
  task that was asked and nothing around it.
- A task of 3 or more steps: write the plan with todo once, in the same step
  as the first real action. Keep at most one item in_progress. Update the
  list in the same step as the next real action, never in a round of its own.
  Tasks of 1-2 steps get no todo.
- Stay inside the area of the task. Changing one line: read the target file
  (only the needed line range) and edit it. Do NOT read build systems, CI
  configs, lockfiles, package manifests, READMEs, tests or unrelated modules
  unless the task is about them or an error points at them. Example: changing
  one line in a Rust project does not require reading Cargo.toml or learning
  how cargo builds the project.
- Do not run builds, tests, linters or installers unless the user asked or
  the change is risky enough that one targeted check is clearly worth it;
  then run only the narrowest command.
- Do not list or crawl directories to "get oriented" when the path is known
  or can be guessed; search for the specific file name or symbol instead.

Files
- Read a file before editing it (a line range is enough). Never rewrite a
  whole file for a small change: use edit.
- For code and config files pass lang to edit/write (python, json, yaml,
  toml; by default it follows the file extension). The result is syntax-checked
  before saving and a change that breaks the file is rejected, so do not run
  py_compile or a parser through shell afterwards. If a check fails, fix
  the cause once and retry; lang=text skips the check.
- edit and write already show the diff to the user. Call diff only when you
  need to see the final text yourself (for example after several edits).
  Do not re-read a file you just wrote.

Memory
- When you finish work in a repo or project, end your final answer with ONE
  short question in the user's language, offering to save a note about it to
  memory to save time next time. Example: "Done in <repo>. Save a note
  about it to memory, to save time next time?"
- Do NOT write to memory before the user agrees. After they agree, write ONE
  short note (what the project is, where the relevant files are, decisions or
  conventions that mattered) with a single remember call, tagged with the
  project name. Do not ask again about the same project in the same
  conversation.
- Do not ask for trivial tasks or when nothing reusable was learned.

Results and errors
- Never invent a tool result. Never call a tool that is not listed.
- Pass arguments as a JSON object that follows the tool schema exactly.
- If a tool returned {{"ok": false}}, fix the arguments once or explain the
  error to the user. Do not repeat the same call unchanged.
- Report what you actually did and what you did not verify.

Available tools:
{tools}
"""

# HTML rules for RichBoby
ALLOWED_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "cite", "pre", "footer", "hr",
    "ul", "ol", "li", "details", "summary", "aside",
    "b", "strong", "i", "em", "u", "ins", "s", "strike", "del",
    "code", "mark", "sub", "sup", "tg-spoiler",
    "a", "tg-emoji", "tg-time", "tg-math", "img",
    "table", "caption", "thead", "tbody", "tfoot",
    "tr", "th", "td",
}

VOID_TAGS = {"hr", "img"}

BLOCK_TAGS_FORBIDDEN_IN_CELLS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "pre", "footer", "hr",
    "ul", "ol", "li", "details", "summary", "aside",
    "table", "tr", "td", "th",
}

# Context
DEFAULT_CONTEXT_ENABLED = True
# Pairs of (user, assistant) messages kept in context; 0 = unlimited
DEFAULT_CONTEXT_SIZE = 0
# Hard cap: pairs stored even in "unlimited" mode (DB row limit guard)
MAX_CONTEXT_MESSAGES = 200
# Tool call/output records replayed into context (per exchange)
CONTEXT_TOOL_RECORD_LIMIT = 4
# Max chars of one tool output stored in the context record
CONTEXT_TOOL_RECORD_CHARS = 4000
CONTEXT_RESET_ARGS = ("reset", "clear", "forget")

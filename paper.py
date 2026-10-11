import os
import re
import json
import shutil
import subprocess
import sys
import builtins
import secrets
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from groq import Groq

# Emoji-safe console output on every OS (Windows terminals default to a legacy encoding)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Load API key from the .env file beside this script
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY. Check the .env file beside paper.py")

client = Groq(api_key=GROQ_API_KEY)

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
# Change the model by setting PAPER_MODEL in .env (or edit the default here).
MODEL_NAME = os.environ.get("PAPER_MODEL", "openai/gpt-oss-20b").strip()
AI_NAME = "Paper by Silicon"
CUSTOM_IDENTITY = "Paper By Silicon (Custom)"

TERMINAL_TIMEOUT = 60             # seconds before a command is killed
TERMINAL_MAX_OUTPUT = 8000        # characters kept per stream
TERMINAL_REQUIRE_APPROVAL = True  # False = run every planned command without asking
HISTORY_WINDOW = 40               # how many past messages the planner can see

# Root workspace (where chat folders live)
ROOT_WORKSPACE = os.getcwd()
CHATS_DIR = os.path.join(ROOT_WORKSPACE, "chats")
SHARED_DIR = os.path.join(ROOT_WORKSPACE, "shared")
SHARED_MEMORY_FILE = os.path.join(SHARED_DIR, "shared_memory.json")
IDENTITY_CACHE_FILE = os.path.join(SHARED_DIR, "model_identity_cache.json")
os.makedirs(CHATS_DIR, exist_ok=True)
os.makedirs(SHARED_DIR, exist_ok=True)

# Set once a chat session is chosen; used for the chat's own memory
CURRENT_CHAT_DIR = None

# Absolute paths OUTSIDE the chat folder that the user approved this session.
# Approving a folder also approves what is inside it.
# (Applies to file operations only; terminal commands are not path-restricted.)
APPROVED_OUTSIDE = set()

print("=" * 55)
print("⚡ SILICON PAPER 1.0 (Ctrl+C or the ⏻ button in the page to power down) ⚡")
print(f"📂 Root workspace: {ROOT_WORKSPACE}")
print("=" * 55)


# ------------------------------------------------------------------
# Path resolution & safety
# ------------------------------------------------------------------
def normalize_rel(path):
    if path is None:
        return None
    return os.path.normpath(path).replace(os.sep, "/")


def within(path, base):
    """True if `path` is `base` or inside it (both absolute, already resolved)."""
    try:
        return os.path.commonpath([base, path]) == base
    except ValueError:
        return False


def is_approved(abs_path):
    return any(within(abs_path, a) for a in APPROVED_OUTSIDE)


def locate(path, chat_dir):
    """
    Resolve a planner path (relative to the chat folder).
    Returns (scope, key, abs_path):
      - ("inside",  chat-relative path, abs)  -> inside the chat folder
      - ("outside", abs, abs)                 -> elsewhere in the root workspace
      - ("invalid", None, None)               -> absolute, malformed, or beyond the root
    """
    if path is None or not isinstance(path, str) or os.path.isabs(path):
        return "invalid", None, None
    try:
        chat_real = os.path.realpath(os.path.abspath(chat_dir))
        root_real = os.path.realpath(os.path.abspath(ROOT_WORKSPACE))
        normalized = os.path.normpath(path or ".")
        abs_t = os.path.realpath(os.path.join(chat_real, normalized))
        if within(abs_t, chat_real):
            return "inside", normalize_rel(os.path.relpath(abs_t, chat_real)), abs_t
        if within(abs_t, root_real):
            return "outside", abs_t, abs_t
        return "invalid", None, None
    except Exception:
        return "invalid", None, None


def resolve_target(path, workspace, allow_root=False):
    """
    The single gatekeeper used at execution time for file operations.
    Returns (abs_path, error). Outside-chat paths need prior per-path approval.
    """
    if path is None:
        return None, "Path is None"
    scope, key, abs_t = locate(path, workspace)
    if scope == "invalid":
        return None, f"Path not allowed (absolute, invalid, or beyond the root workspace): '{path}'"
    if scope == "inside":
        if key == "." and not allow_root:
            return None, "Cannot target workspace root itself"
        return abs_t, None
    if not is_approved(abs_t):
        return None, f"'{path}' is outside the chat folder and has not been approved"
    return abs_t, None


def _is_allowed_abs(abs_path, workspace):
    chat_real = os.path.realpath(os.path.abspath(workspace))
    return within(abs_path, chat_real) or is_approved(abs_path)


def _revalidate(abs_path, workspace):
    """Re-check right before touching the disk. Returns an error string or None."""
    real = os.path.realpath(abs_path)
    if real != abs_path:
        return "Path changed while the operation was being prepared (possible symlink swap)"
    if not _is_allowed_abs(real, workspace):
        return "Path is no longer inside an allowed location"
    return None


def _prep(path, workspace, allow_root=False):
    abs_t, err = resolve_target(path, workspace, allow_root)
    if err:
        return None, err
    err = _revalidate(abs_t, workspace)
    if err:
        return None, err
    return abs_t, None


def _open_checked(abs_path, workspace, flags, mode=0o666, created=False):
    """
    Open without following a symlink in the final component, then verify that
    what we opened is still the allowed path BEFORE any write/truncate happens.
    If we just created the file and verification fails, remove it again.
    """
    fd = os.open(abs_path, flags | getattr(os, "O_NOFOLLOW", 0), mode)
    try:
        real_now = os.path.realpath(abs_path)
        if not _is_allowed_abs(real_now, workspace):
            raise PermissionError("Path was redirected outside the allowed area during the operation")
        if not os.path.samestat(os.fstat(fd), os.stat(real_now)):
            raise PermissionError("File was swapped during the operation")
    except BaseException:
        if created:
            try:
                if os.path.samestat(os.fstat(fd), os.lstat(abs_path)):
                    os.unlink(abs_path)
            except OSError:
                pass
        os.close(fd)
        raise
    return fd


def _read_text(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


# ------------------------------------------------------------------
# Filesystem inspection
# ------------------------------------------------------------------
def inspect_directory_recursive(root_path):
    """Recursively inspect directory structure, returning paths relative to root."""
    root_path = os.path.abspath(root_path)
    all_items = {"folders": [], "files": []}
    try:
        for root, dirs, files in os.walk(root_path, followlinks=False):
            for d in dirs:
                rel = os.path.relpath(os.path.join(root, d), root_path)
                all_items["folders"].append(rel)
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), root_path)
                all_items["files"].append(rel)
        return {
            "root": root_path,
            "folders": sorted(all_items["folders"]),
            "files": sorted(all_items["files"]),
        }
    except Exception as e:
        return {"error": str(e), "root": root_path, "folders": [], "files": []}


def snapshot_tree(abs_path):
    """
    Snapshot of a file or folder tree: {relative path: size or None for folders}.
    "." stands for the item itself. Used to verify moves, contents included.
    """
    snap = {}
    try:
        if os.path.isdir(abs_path) and not os.path.islink(abs_path):
            snap["."] = None
            for root, dirs, files in os.walk(abs_path, followlinks=False):
                for d in dirs:
                    snap[normalize_rel(os.path.relpath(os.path.join(root, d), abs_path))] = None
                for f in files:
                    p = os.path.join(root, f)
                    snap[normalize_rel(os.path.relpath(p, abs_path))] = os.lstat(p).st_size
        elif os.path.lexists(abs_path):
            snap["."] = os.lstat(abs_path).st_size
    except Exception:
        pass
    return snap


# ------------------------------------------------------------------
# Chat history
# ------------------------------------------------------------------
def find_existing_chat(name):
    """
    Case-insensitive lookup of an existing chat folder, so 'Notes', 'notes' and
    'NOTES' are the same chat on EVERY OS (Linux is normally case-sensitive).
    Returns (folder_name or None, duplicates_exist).
    An exact-case match wins; otherwise the first match alphabetically.
    """
    try:
        entries = sorted(e for e in os.listdir(CHATS_DIR)
                         if os.path.isdir(os.path.join(CHATS_DIR, e)))
    except OSError:
        return None, False
    matches = [e for e in entries if e.casefold() == name.casefold()]
    if not matches:
        return None, False
    if name in matches:
        return name, len(matches) > 1
    return matches[0], len(matches) > 1


def load_chat_history(chat_dir):
    history_path = os.path.join(chat_dir, "chat_history.json")
    if os.path.isfile(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "chat_name": os.path.basename(chat_dir),
        "created": datetime.now().isoformat(),
        "messages": [],
    }


def save_chat_history(chat_dir, history):
    history_path = os.path.join(chat_dir, "chat_history.json")
    tmp_path = history_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, history_path)


def append_chat_message(chat_dir, role, content, metadata=None):
    history = load_chat_history(chat_dir)
    msg = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
    if metadata:
        msg["metadata"] = metadata
    history.setdefault("messages", []).append(msg)
    save_chat_history(chat_dir, history)


def print_chat_history(history):
    """Print the entire saved conversation when an existing chat is opened."""
    messages = history.get("messages", [])
    print("\n" + "=" * 55)
    print(f"📖 Loaded chat '{history.get('chat_name', '?')}' "
          f"(created {history.get('created', '?')}) — {len(messages)} message(s)")
    print("=" * 55)
    labels = {"user": "User", "assistant": "Paper", "system": "System"}
    for m in messages:
        role = labels.get(m.get("role"), str(m.get("role")).title())
        ts = (m.get("timestamp") or "")[:19].replace("T", " ")
        print(f"\n[{ts}] {role}:")
        print(m.get("content", ""))
    if not messages:
        print("\n(this chat has no messages yet)")
    print("\n" + "=" * 55)
    print("— end of previous conversation —")
    print("=" * 55)


# ------------------------------------------------------------------
# Memory helpers (generic JSON-object store)
# ------------------------------------------------------------------
def _read_json_object(path, label):
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"⚠️ {label} file is malformed (not a JSON object). Resetting.")
            return {}
        return data
    except Exception as e:
        print(f"⚠️ Failed to read {label}: {e}. Resetting.")
        return {}


def _write_json_object(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


# ------------------------------------------------------------------
# Identity (derived from the CURRENT model, always present in shared memory)
# ------------------------------------------------------------------
_SELF_REPORTED_NAME = None  # No probing — just use Custom
IDENTITY_VALUE = f"{AI_NAME} (Custom)"

IDENTITY_RULE_VALUE = (
    f"When the user asks who or what you are, what your name is, or which model you run on, "
    f"say that you are '{IDENTITY_VALUE}'. (Model id in use: {MODEL_NAME}.) "
    f"If they ask follow-up questions like 'what do you mean?', explain naturally instead of "
    f"repeating the same line."
)
RESERVED_MEMORY = {
    "identity": IDENTITY_VALUE,
    "identity_rule": IDENTITY_RULE_VALUE,
}


# ------------------------------------------------------------------
# Shared Memory (external at root level, shared across all chats)
# ------------------------------------------------------------------
def save_shared_memory(memory):
    _write_json_object(SHARED_MEMORY_FILE, memory)


def load_shared_memory():
    """
    Load shared memory and ALWAYS make sure the reserved identity keys are
    present and correct for the CURRENT model. If they were missing or altered
    (e.g. you switched models), they are re-seeded and persisted automatically.
    """
    memory = _read_json_object(SHARED_MEMORY_FILE, "Shared memory")
    changed = False
    for k, v in RESERVED_MEMORY.items():
        if memory.get(k) != v:
            memory[k] = v
            changed = True
    if changed:
        try:
            save_shared_memory(memory)
        except Exception as e:
            print(f"⚠️ Could not persist identity to shared memory: {e}")
    return memory


def ensure_identity_in_shared_memory():
    """Guarantees identity is in shared memory (called every turn)."""
    load_shared_memory()


# ------------------------------------------------------------------
# Chat Memory (private to each chat, stored inside the chat folder)
# ------------------------------------------------------------------
def chat_memory_path(chat_dir=None):
    chat_dir = chat_dir or CURRENT_CHAT_DIR
    if not chat_dir:
        return None
    return os.path.join(chat_dir, "chat_memory.json")


def load_chat_memory(chat_dir=None):
    path = chat_memory_path(chat_dir)
    if not path:
        return {}
    return _read_json_object(path, "Chat memory")


def save_chat_memory(memory, chat_dir=None):
    path = chat_memory_path(chat_dir)
    if not path:
        raise RuntimeError("No active chat for chat memory")
    _write_json_object(path, memory)


def build_memory_context():
    """Both memories, formatted for inclusion in every prompt."""
    shared = load_shared_memory()
    chat = load_chat_memory()
    return (
        "CHAT MEMORY (private to this chat, current contents):\n"
        f"{json.dumps(chat, indent=2, ensure_ascii=False) if chat else '(empty)'}\n\n"
        "SHARED MEMORY (shared across all chats, current contents):\n"
        f"{json.dumps(shared, indent=2, ensure_ascii=False) if shared else '(empty)'}"
    )


# ------------------------------------------------------------------
# Memory operations (scope-aware)
# ------------------------------------------------------------------
def memory_operation(op):
    """
    Handles get/set/delete on 'chat' or 'shared' memory.
    - 'get' without a scope looks in chat memory first, then shared memory.
    - 'set'/'delete' without a scope default to 'shared'.
    Reserved identity keys cannot be changed or deleted.
    """
    action = op.get("memory_action")
    key = op.get("memory_key")
    value = op.get("memory_value")
    scope = (op.get("memory_scope") or "").strip().lower() or None

    if not action or not key:
        return {"success": False, "error": "Memory operations require 'memory_action' and 'memory_key'."}
    if not isinstance(key, str):
        return {"success": False, "error": "'memory_key' must be a string."}
    if scope not in (None, "chat", "shared"):
        return {"success": False, "error": f"Unknown memory_scope: '{scope}' (use 'chat' or 'shared')"}

    if action == "get":
        chat_mem = load_chat_memory()
        shared_mem = load_shared_memory()
        if scope == "chat":
            found, val, where = key in chat_mem, chat_mem.get(key), "chat"
        elif scope == "shared":
            found, val, where = key in shared_mem, shared_mem.get(key), "shared"
        else:
            if key in chat_mem:
                found, val, where = True, chat_mem[key], "chat"
            elif key in shared_mem:
                found, val, where = True, shared_mem[key], "shared"
            else:
                found, val, where = False, None, "chat+shared"
        result = {"success": True, "key": key, "value": val, "scope": where}
        if not found:
            result["warning"] = f"Key '{key}' not found in {where} memory."
        return result

    scope = scope or "shared"

    if scope == "shared" and key in RESERVED_MEMORY:
        return {"success": False, "error": f"'{key}' is a protected identity key and cannot be changed or deleted."}

    if scope == "chat":
        memory = load_chat_memory()
        saver, loader = save_chat_memory, load_chat_memory
    else:
        memory = load_shared_memory()
        saver, loader = save_shared_memory, load_shared_memory

    if action == "set":
        if value is None:
            return {"success": False, "error": "'memory_value' is required for 'set'."}
        try:
            json.dumps(value)
        except TypeError:
            return {"success": False, "error": "'memory_value' is not JSON serializable."}
        memory[key] = value
        saver(memory)
        verify = loader()
        if verify.get(key) == value:
            return {"success": True, "key": key, "value": value, "scope": scope, "verified": True}
        return {"success": False, "key": key, "value": value, "scope": scope,
                "error": "Write verification failed: value not persisted correctly"}

    if action == "delete":
        if key in memory:
            del memory[key]
            saver(memory)
            return {"success": True, "deleted": key, "scope": scope}
        return {"success": False, "error": f"Key '{key}' not found in {scope} memory."}

    return {"success": False, "error": f"Unknown memory action: '{action}'"}


# ------------------------------------------------------------------
# Terminal tool (ANY command, run through the system shell)
# ------------------------------------------------------------------
def _clip(text, limit=TERMINAL_MAX_OUTPUT):
    text = text or ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if len(text) > limit:
        return text[:limit] + "\n...[truncated]"
    return text


def run_terminal(op, workspace):
    """
    Run any command line through the system shell (pipes, &&, redirects all work),
    inside the chat folder. stdout/stderr/exit code are captured and returned.
    Only GROQ_API_KEY is removed from the environment.
    """
    command = op.get("command")
    if not isinstance(command, str) or not command.strip():
        return {"success": False, "error": "No command given"}

    env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
    env.setdefault("PYTHONIOENCODING", "utf-8")

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=os.path.realpath(workspace),
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TERMINAL_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        return {
            "success": False, "command": command, "exit_code": None,
            "stdout": _clip(e.stdout), "stderr": _clip(e.stderr),
            "error": f"Command timed out after {TERMINAL_TIMEOUT}s",
        }
    except Exception as e:
        return {"success": False, "command": command, "error": str(e)}

    stdout, stderr = _clip(proc.stdout), _clip(proc.stderr)
    result = {
        "success": proc.returncode == 0,
        "command": command,
        "exit_code": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }
    if proc.returncode != 0:
        result["error"] = f"Command exited with code {proc.returncode}: {(stderr or stdout).strip()}"
    return result


# ------------------------------------------------------------------
# Planning
# ------------------------------------------------------------------
IDENTITY_PROMPT_RULE = (
    f"IDENTITY: You are '{AI_NAME}'. Model id in use: {MODEL_NAME}. "
    f"When the user simply asks who/what you are, your name, or which model you are, "
    f"answer that you are '{IDENTITY_VALUE}'. Do not claim to be any other model or product."
)

HOW_IT_WORKS = (
    "HOW PAPER ACTUALLY WORKS (true facts about yourself, use them when asked to explain yourself):\n"
    f"- Paper is a Python program (paper.py) wrapped around a language model ({MODEL_NAME}). "
    "The model is the 'brain'; the Python script is the 'hands'. The model cannot touch files or run commands itself.\n"
    "- Every chat gets its own folder (chats/<name>/) holding chat_history.json, chat_memory.json and a files/ folder.\n"
    "- For each message you send: (1) Paper scans the chat folder, (2) it sends the model your request, the file list, "
    "recent chat history, and both memories (private chat memory + shared memory), (3) the model replies with a JSON "
    "plan, either a conversational answer or an ordered list of operations (create/write/append/read/list/move/delete/"
    "memory/terminal), (4) the script validates every path, keeping everything inside the chat folder unless you approve "
    "an outside path, (5) it asks for your approval before deleting/moving or running any terminal command, "
    "(6) Python executes each operation, (7) it verifies the result on the real disk (content, sizes, existence), "
    "(8) if something fails it asks the model for a recovery plan, up to 2 times, and (9) the model writes a short summary "
    "using only the real results.\n"
    "- Memory is just JSON files: chat_memory.json (this chat only) and shared/shared_memory.json (all chats). "
    "My identity is stored in shared memory and rebuilt from whichever model is currently configured.\n"
    "- Safety: symlink-swap checks, per-path approval for anything outside the chat folder, terminal commands need "
    "approval and have a timeout, and API keys are hidden from commands.\n"
    "- Limits: the model plans from text only, so it can make mistakes; that is why validation, approval and "
    "verification exist."
)

EXPLAIN_RULE = (
    "EXPLAINING YOURSELF: If the user asks you to explain yourself, how you work, what you can do, what you are made of, "
    "or says things like 'explain it' / 'what do you mean?' about your nature, give a real explanation based on "
    "HOW PAPER ACTUALLY WORKS below, in your own words, tailored to what they asked. Do NOT just repeat your identity "
    "line. Only state facts from that description; never invent abilities."
)

TERMINAL_RULES = (
    "- For \"terminal\" operations, set \"command\" to a command line for the system shell "
    f"(OS: {sys.platform}). Any command is allowed, including pipes, '&&' and redirects. "
    "It runs inside the chat folder"
    + (" and the user must approve it" if TERMINAL_REQUIRE_APPROVAL else "")
    + f". Commands are killed after {TERMINAL_TIMEOUT}s and have no keyboard input, so never run anything interactive. "
    "You will get the real output afterwards.\n"
    "- Example: to generate a UUID use: python -c \"import uuid; print(uuid.uuid4())\" "
    "(use 'python3' instead of 'python' on macOS/Linux if needed).\n"
    "- NEVER invent or guess command output (UUIDs, dates, random values, sizes, etc.). If the answer needs real output, "
    "plan a terminal operation instead of answering directly."
)

OP_SCHEMA = """{
      "operation": "create|delete|list|move|memory|append|write|read|terminal",
      "type": "folder|file",
      "source": "path/to/source" or null,
      "destination": "path/to/destination" or null,
      "target": "path/to/target" or null,
      "content": "file contents" or null,
      "command": "terminal command line" or null,
      "memory_action": "get|set|delete" or null,
      "memory_scope": "chat|shared" or null,
      "memory_key": "string key" or null,
      "memory_value": "any JSON value" or null,
      "explanation": "brief reason"
    }"""


def plan_operations(task, filesystem_state, chat_history, execution_errors=None,
                    last_execution_log=None):
    """
    Single AI call returning a structured task plan with one or more operations.
    Always receives BOTH the chat's own memory and the shared memory.
    If execution_errors is provided, the model is asked to recover from them.
    """
    ensure_identity_in_shared_memory()

    folders = filesystem_state.get("folders", [])
    files = filesystem_state.get("files", [])
    recent = chat_history.get("messages", [])[-HISTORY_WINDOW:]
    recent_clean = [{"role": m.get("role"), "content": m.get("content")} for m in recent]

    memory_context = build_memory_context()

    exec_log_summary = "(none)"
    if last_execution_log:
        exec_log_summary = json.dumps(
            [result_for_model(e, brief=True) for e in last_execution_log],
            indent=2, ensure_ascii=False,
        )

    if execution_errors:
        planning_prompt = f"""{IDENTITY_PROMPT_RULE}

TASK: {task}

CURRENT FILESYSTEM (paths relative to chat workspace):
Folders: {json.dumps(folders)}
Files: {json.dumps(files)}

{memory_context}

RECENT CHAT HISTORY (for context):
{json.dumps(recent_clean, indent=2, ensure_ascii=False) if recent_clean else "(none)"}

You previously attempted this task but encountered errors.
Please analyze the errors below and provide a RECOVERY PLAN to complete the task.

ERRORS:
{json.dumps(execution_errors, indent=2, default=str)}

RECOVERY GUIDELINES:
- If a 'create' operation failed because the file already exists, and the user wants to ADD content, use 'append' instead of 'delete + create'.
- If a 'create' operation failed because the file already exists, and the user wants to REPLACE content, use 'write' instead of 'delete + create'.
- NEVER delete an existing file just to recreate it with different content. Use 'append' or 'write' instead.
- If a 'move'/'rename' operation failed, check the actual filesystem state (the current files/folders listed above) and try again with correct paths.
- If a terminal command failed, fix the command; do not repeat it unchanged.
- If the file already has the correct name/content, report success — no operation needed.
- The identity keys in shared memory ('identity', 'identity_rule') are protected; never try to modify or delete them.
{TERMINAL_RULES}

Respond ONLY with valid JSON of this shape:
{{
  "is_question": false,
  "operations": [
    {OP_SCHEMA}
  ]
}}
- Respond with ONLY the JSON object, no prose around it.
- If no operations are needed because the task is already complete, return an empty operations list: {{ "is_question": false, "operations": [] }}
"""
    else:
        planning_prompt = f"""{IDENTITY_PROMPT_RULE}

{EXPLAIN_RULE}

{HOW_IT_WORKS}

TASK: {task}

CURRENT FILESYSTEM (paths relative to chat workspace):
Folders: {json.dumps(folders)}
Files: {json.dumps(files)}

{memory_context}

RECENT CHAT HISTORY (for context):
{json.dumps(recent_clean, indent=2, ensure_ascii=False) if recent_clean else "(none)"}

LAST EXECUTION LOG (what actually happened in the last operation sequence):
{exec_log_summary}

You have TWO memories and you must consult BOTH every time before answering or planning:
  * CHAT MEMORY: private to this chat.
  * SHARED MEMORY: shared across all chats (includes your identity).

Decide whether this is a filesystem/memory/terminal operation request or a casual question/conversation.

If it is FILESYSTEM/MEMORY/TERMINAL OPERATION(S), respond ONLY with valid JSON of this shape:
{{
  "is_question": false,
  "operations": [
    {OP_SCHEMA}
  ]
}}

If it is a QUESTION or CASUAL CONVERSATION (including questions answerable from chat memory, shared memory, or the chat history), respond with:
{{
  "is_question": true,
  "response": "natural conversational response"
}}

CRITICAL RULES:
- For "memory" operations, use "memory_action" (get, set, delete), "memory_scope" ("chat" or "shared"), "memory_key" (string), and "memory_value" (the actual value to store, e.g. a string for a name).
- Use memory_scope "chat" for things that only matter in this chat (this chat's topic, local notes, temporary context). Use memory_scope "shared" for things that should be remembered across all chats (facts about the user, global preferences). If the user says "remember this" with no further hint, use "shared".
- Answer questions using BOTH memories and the chat history. If the two memories disagree, prefer CHAT MEMORY for chat-specific matters and SHARED MEMORY for general matters.
- If the user simply asks who/what you are, what your name is, or which model you are, answer that you are "{IDENTITY_VALUE}". If they ask you to EXPLAIN yourself or how you work, follow the EXPLAINING YOURSELF rule instead of repeating the identity line.
- NEVER set or delete the shared keys 'identity' or 'identity_rule'; they are protected.
- All file paths are relative to the chat workspace. To reach outside the chat folder use '../' paths (e.g. '../shared'); the user is asked to approve each outside path individually.
- For "create" operations, ALWAYS specify "target" (the path of the new file/folder). This FAILS if the file already exists. Use this ONLY for new files.
- For "append" operations, specify "target" (existing file path) and "content" (text to append). The file MUST already exist. Use this to ADD content to an existing file.
- For "write" operations, specify "target" (file path) and "content" (full file contents). This creates or OVERWRITES the file. Use this to replace file contents entirely.
- For "read" operations, specify "target" (file path) to read file contents. Use this when the user asks what's in a file.
- NEVER delete and recreate a file just to modify its content. Use "append" or "write" instead.
- For "create file", include the proper extension when the user mentions one (e.g., "txt file" -> "name.txt").
- By default, place user-created files inside the "files/" subfolder (e.g., "files/hobbies.txt") unless the user explicitly specifies another location.
- For "move" operations, support BOTH:
  * RENAME: set "destination" to the NEW full path/name (e.g., rename "files/hobbies" -> destination="files/hobbies.txt").
  * MOVE INTO FOLDER: set "destination" to an existing folder path.
- For "list" operations, set "target" to the directory path. Use "." to list the current workspace root.
{TERMINAL_RULES}
- Plan ALL operations as an ordered SEQUENCE. Later operations may depend on paths created by earlier operations.
- When answering questions about what happened, use the LAST EXECUTION LOG above — do not invent explanations.
- Respond with ONLY the JSON object, no prose around it.
"""
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": planning_prompt}],
            model=MODEL_NAME,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        if "{" in text:
            js = text[text.find("{"): text.rfind("}") + 1]
            return json.loads(js)
        return {"is_question": True, "response": "I couldn't parse that. Could you rephrase?"}
    except Exception as e:
        return {"is_question": True, "response": f"Planning error: {str(e)}"}


# ------------------------------------------------------------------
# Result reporting (natural-language reply grounded in real results)
# ------------------------------------------------------------------
def result_for_model(entry, brief=False):
    """Compact, honest description of one executed operation, including real output."""
    op, res = entry["op"], entry["result"]
    out = {
        "operation": op.get("operation"),
        "target": op.get("target") or op.get("source"),
        "success": res.get("success"),
        "verified": entry["verified"],
    }
    if op.get("operation") == "move":
        out["destination"] = op.get("destination")
    if not res.get("success"):
        out["error"] = res.get("error")
    limit = 500 if brief else 4000
    operation = op.get("operation")
    if operation == "terminal":
        out["command"] = op.get("command")
        out["exit_code"] = res.get("exit_code")
        out["stdout"] = (res.get("stdout") or "")[:limit]
        out["stderr"] = (res.get("stderr") or "")[:limit]
    elif operation == "read" and res.get("success"):
        out["content"] = (res.get("content") or "")[:limit]
    elif operation == "list" and res.get("success"):
        out["folders"] = res.get("folders")
        out["files"] = res.get("files")
    elif operation == "memory" and res.get("success"):
        out["memory_key"] = op.get("memory_key")
        out["memory_scope"] = res.get("scope")
        if op.get("memory_action") == "get":
            out["value"] = res.get("value")
    elif operation in ("write", "create", "append") and op.get("content") is not None and not brief:
        out["content_length"] = len(op.get("content") or "")
    return out


def summarize_for_user(task, execution_log, sequence_complete, errors):
    """
    Natural-language reply built ONLY from what actually happened, so future turns
    (which read chat history) see an explanation instead of just a technical log.
    """
    if not execution_log:
        reasons = "; ".join(str(e.get("error")) for e in (errors or [])) or "unknown reason"
        return f"I didn't change anything: {reasons}"

    facts = [result_for_model(e) for e in execution_log]
    prompt = f"""{IDENTITY_PROMPT_RULE}

The user asked: {task}

Here is exactly what happened (real results from the operations that were executed):
{json.dumps(facts, indent=2, ensure_ascii=False)}

Sequence fully completed: {sequence_complete}

Write a brief, natural reply (1-4 sentences) telling the user what was done and, where relevant, the
actual results (command output, file contents, listings, remembered values).
Rules:
- Use ONLY the data above. Never invent or guess output.
- If something failed or was not verified, say so plainly.
- No JSON, no headings."""
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=MODEL_NAME,
            temperature=0.0,
        )
        text = (response.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        return None


# ------------------------------------------------------------------
# Projected filesystem state (validates a sequence before running it)
# ------------------------------------------------------------------
VALID_OPS = ("create", "delete", "list", "move", "memory", "append", "write", "read", "terminal")


def _parent_exists(key, p):
    parent = os.path.dirname(key)
    return parent in ("", ".") or parent in p


def apply_op_to_projected(op, p_state, chat_dir):
    """
    Apply an operation to the projected state: a set of chat-relative paths.
    Paths outside the chat folder are not tracked here; they are checked against
    the real filesystem when the operation executes.
    Terminal commands can change anything, so they are not projected.
    Returns (success, error, new_p_state).
    """
    p = set(p_state)
    operation = op.get("operation")
    if operation not in VALID_OPS:
        return False, f"Unknown operation: {operation}", p

    if operation in ("list", "memory", "read"):
        return True, None, p

    if operation == "terminal":
        if not isinstance(op.get("command"), str) or not op.get("command").strip():
            return False, "No command specified for terminal", p
        return True, None, p

    file_type = op.get("type", "folder")
    target = op.get("target")

    if operation in ("create", "append", "write", "delete"):
        if not target:
            return False, f"No target specified for {operation}", p
        scope, key, _ = locate(target, chat_dir)
        if scope == "invalid":
            return False, f"Invalid or unsafe path: '{target}'", p
        if scope == "outside":
            return True, None, p

        if operation == "create":
            if key in p:
                return False, f"'{key}' already exists", p
            if not _parent_exists(key, p):
                return False, f"Parent folder '{os.path.dirname(key)}' does not exist", p
            if file_type not in ("folder", "file"):
                return False, f"Unknown type: {file_type}", p
            p.add(key)
            return True, None, p

        if operation == "append":
            if key not in p:
                return False, f"File '{key}' does not exist (append requires an existing file)", p
            return True, None, p

        if operation == "write":
            if any(f.startswith(key + "/") for f in p):
                return False, f"'{key}' is a folder", p
            if key not in p:
                if not _parent_exists(key, p):
                    return False, f"Parent folder '{os.path.dirname(key)}' does not exist", p
                p.add(key)
            return True, None, p

        if operation == "delete":
            if key not in p:
                return False, f"'{key}' does not exist", p
            if file_type == "folder":
                prefix = key + "/"
                p = {f for f in p if f != key and not f.startswith(prefix)}
            elif file_type == "file":
                p.discard(key)
            else:
                return False, f"Unknown type: {file_type}", p
            return True, None, p

    if operation == "move":
        source, destination = op.get("source"), op.get("destination")
        if not source:
            return False, "No source for move", p
        if not destination:
            return False, "No destination for move", p
        s_scope, s_key, _ = locate(source, chat_dir)
        d_scope, d_key, _ = locate(destination, chat_dir)
        if s_scope == "invalid" or d_scope == "invalid":
            return False, "Invalid source or destination path", p
        if s_scope == "inside" and s_key not in p:
            return False, f"Source '{s_key}' does not exist", p

        expected = None
        if d_scope == "inside":
            expected = normalize_rel(os.path.join(d_key, os.path.basename(s_key))) if d_key in p else d_key
            if expected in p:
                return False, f"Target '{expected}' already exists", p
            if not _parent_exists(expected, p):
                return False, f"Parent folder '{os.path.dirname(expected)}' does not exist", p
            if s_scope == "inside" and (expected == s_key or expected.startswith(s_key + "/")):
                return False, "Cannot move into itself or a child folder", p

        if s_scope == "inside":
            prefix = s_key + "/"
            moved = {f for f in p if f == s_key or f.startswith(prefix)}
            p -= moved
            if expected is not None:
                for item in moved:
                    p.add(expected if item == s_key else normalize_rel(os.path.join(expected, item[len(prefix):])))
        elif expected is not None:
            p.add(expected)
        return True, None, p

    return False, "Unhandled operation", p


# ------------------------------------------------------------------
# Path validation & access approval for a plan
# ------------------------------------------------------------------
def collect_op_paths(op):
    operation = op.get("operation")
    if operation in ("create", "delete", "append", "write", "read"):
        return [op.get("target")]
    if operation == "move":
        return [op.get("source"), op.get("destination")]
    if operation == "list":
        return [op.get("target") or "."]
    return []


def validate_op_paths(op, workspace):
    """Validate all paths in op are allowed (inside the chat folder or individually approved)."""
    errors = []
    is_list = (op.get("operation") == "list")
    for t in collect_op_paths(op):
        if t is None:
            continue
        _, err = resolve_target(t, workspace, allow_root=is_list)
        if err:
            errors.append(f"Unsafe path '{t}': {err}")
    return errors


def ensure_access(ops, chat_dir):
    """
    Ask the user to approve each distinct path outside the chat folder, one by one.
    Approval covers only that path (and what's inside it, if it's a folder), for this session.
    Returns (ok, message).
    """
    needed = []
    for op in ops:
        for t in collect_op_paths(op):
            if t is None:
                continue
            scope, _, abs_t = locate(t, chat_dir)
            if scope == "invalid":
                return False, f"'{t}' is not allowed (absolute, invalid, or beyond the root workspace)."
            if scope == "outside" and not is_approved(abs_t) and abs_t not in needed:
                needed.append(abs_t)

    for abs_t in needed:
        print(f"\n⚠️  Paper wants to access a path OUTSIDE its chat folder:\n   {abs_t}")
        print("   (If it's a folder, everything inside it is included. Approval lasts this session only.)")
        grant = input("   Allow access to this path? (yes/no): ").strip().lower()
        if grant != "yes":
            return False, f"Access to '{abs_t}' was denied."
        APPROVED_OUTSIDE.add(abs_t)
        print("   🔓 Approved")
    return True, None


# ------------------------------------------------------------------
# Filesystem operations
# ------------------------------------------------------------------
def py_create_folder(name, workspace):
    try:
        abs_t, err = _prep(name, workspace)
        if err:
            return {"success": False, "error": err}
        if not os.path.isdir(os.path.dirname(abs_t)):
            return {"success": False, "error": f"Parent directory doesn't exist for '{name}'"}
        os.mkdir(abs_t)
        err = _revalidate(abs_t, workspace)
        if err:
            return {"success": False, "error": f"Folder created but location check failed: {err}"}
        return {"success": True, "created": name}
    except FileExistsError:
        return {"success": False, "error": f"Folder '{name}' already exists"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def py_create_file(path, content, workspace):
    """Create a file EXCLUSIVELY — never overwrite existing files."""
    try:
        abs_t, err = _prep(path, workspace)
        if err:
            return {"success": False, "error": err}
        if os.path.isdir(abs_t):
            return {"success": False, "error": f"Folder '{path}' already exists at this location. Delete it first."}
        if not os.path.isdir(os.path.dirname(abs_t)):
            return {"success": False, "error": f"Parent directory doesn't exist for '{path}'"}

        fd = _open_checked(abs_t, workspace, os.O_WRONLY | os.O_CREAT | os.O_EXCL, created=True)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        return {"success": True, "created": path, "content": content}
    except FileExistsError:
        return {"success": False, "error": f"File '{path}' already exists (use 'append' or 'write' to modify)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def py_append_file(path, content, workspace):
    """Append content to an existing file. Fails if file doesn't exist."""
    try:
        abs_t, err = _prep(path, workspace)
        if err:
            return {"success": False, "error": err}
        if not os.path.exists(abs_t):
            return {"success": False, "error": f"File '{path}' does not exist (use 'create' or 'write' for new files)"}
        if not os.path.isfile(abs_t):
            return {"success": False, "error": f"'{path}' is not a file"}

        fd = _open_checked(abs_t, workspace, os.O_WRONLY | os.O_APPEND)
        with os.fdopen(fd, "a", encoding="utf-8", newline="") as f:
            f.write(content)
        return {"success": True, "appended": path, "content_appended": content}
    except Exception as e:
        return {"success": False, "error": str(e)}


def py_write_file(path, content, workspace):
    """Write content to a file (creates or overwrites). Truncation happens only AFTER the path is re-verified."""
    try:
        abs_t, err = _prep(path, workspace)
        if err:
            return {"success": False, "error": err}
        if os.path.isdir(abs_t):
            return {"success": False, "error": f"Folder '{path}' already exists at this location"}
        if not os.path.isdir(os.path.dirname(abs_t)):
            return {"success": False, "error": f"Parent directory doesn't exist for '{path}'"}

        existed = os.path.exists(abs_t)
        fd = _open_checked(abs_t, workspace, os.O_WRONLY | os.O_CREAT, created=not existed)
        try:
            os.ftruncate(fd, 0)
        except Exception:
            os.close(fd)
            raise
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        return {"success": True, "written": path, "content": content}
    except Exception as e:
        return {"success": False, "error": str(e)}


def py_read_file(abs_path, workspace):
    """Read file contents (re-verified at open time)."""
    try:
        if not os.path.exists(abs_path):
            return {"success": False, "error": "File does not exist"}
        if not os.path.isfile(abs_path):
            return {"success": False, "error": "Path is not a file"}
        fd = _open_checked(abs_path, workspace, os.O_RDONLY)
        with os.fdopen(fd, "r", encoding="utf-8", newline="") as f:
            content = f.read()
        return {"success": True, "path": abs_path, "content": content}
    except Exception as e:
        return {"success": False, "error": str(e)}


def py_delete_folder(path, workspace):
    try:
        abs_t, err = _prep(path, workspace)
        if err:
            return {"success": False, "error": err}
        if not os.path.exists(abs_t):
            return {"success": False, "error": f"Path '{path}' does not exist"}
        if not os.path.isdir(abs_t):
            return {"success": False, "error": f"'{path}' is not a folder"}
        shutil.rmtree(abs_t)
        return {"success": True, "deleted": path}
    except Exception as e:
        return {"success": False, "error": str(e)}


def py_delete_file(path, workspace):
    try:
        abs_t, err = _prep(path, workspace)
        if err:
            return {"success": False, "error": err}
        if not os.path.exists(abs_t):
            return {"success": False, "error": f"File '{path}' does not exist"}
        if not os.path.isfile(abs_t):
            return {"success": False, "error": f"'{path}' is not a file"}
        os.remove(abs_t)
        return {"success": True, "deleted": path}
    except Exception as e:
        return {"success": False, "error": str(e)}


def py_move_item(source, destination, workspace):
    """
    Support BOTH move-into-folder AND rename.
    - If destination is an existing folder -> move source INTO it (preserving basename).
    - If destination does NOT exist -> rename source TO destination (new full path).
    """
    try:
        abs_src, err = _prep(source, workspace)
        if err:
            return {"success": False, "error": f"Unsafe source: {err}"}
        abs_dst, err = _prep(destination, workspace)
        if err:
            return {"success": False, "error": f"Unsafe destination: {err}"}

        if not os.path.exists(abs_src):
            return {"success": False, "error": f"Source '{source}' does not exist"}

        if os.path.isdir(abs_dst) and abs_dst != abs_src:
            target_path = os.path.join(abs_dst, os.path.basename(abs_src))
        else:
            target_path = abs_dst

        if not _is_allowed_abs(os.path.realpath(target_path), workspace):
            return {"success": False, "error": "Resolved target is outside the allowed area"}
        if os.path.exists(target_path):
            return {"success": False, "error": f"Target '{os.path.relpath(target_path, workspace)}' already exists"}
        if not os.path.isdir(os.path.dirname(target_path)):
            return {"success": False, "error": f"Parent directory doesn't exist for '{destination}'"}
        if within(target_path, abs_src):
            return {"success": False, "error": "Cannot move into itself or child directory"}

        shutil.move(abs_src, target_path)

        if not os.path.exists(abs_src) and os.path.exists(target_path):
            return {"success": True, "moved_from": source, "moved_to": target_path, "verified": True}
        return {
            "success": False,
            "error": f"Move check failed: source_exists={os.path.exists(abs_src)}, dest_exists={os.path.exists(target_path)}",
            "moved_from": source,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def py_list_items(path):
    """List items in a specific (already-validated) directory."""
    try:
        if os.path.isfile(path):
            return {"success": True, "is_file": True, "path": path}
        items = os.listdir(path)
        return {
            "success": True,
            "path": path,
            "folders": sorted([i for i in items if os.path.isdir(os.path.join(path, i))]),
            "files": sorted([i for i in items if os.path.isfile(os.path.join(path, i))]),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def execute_op(op, workspace):
    operation = op.get("operation")

    if operation == "memory":
        return memory_operation(op)

    if operation == "terminal":
        return run_terminal(op, workspace)

    if operation == "list":
        abs_t, err = resolve_target(op.get("target") or ".", workspace, allow_root=True)
        if err:
            return {"success": False, "error": err}
        return py_list_items(abs_t)

    if operation == "read":
        abs_t, err = resolve_target(op.get("target"), workspace)
        if err:
            return {"success": False, "error": err}
        return py_read_file(abs_t, workspace)

    file_type = op.get("type", "folder")

    if operation == "create":
        target = op.get("target")
        if file_type == "file":
            return py_create_file(target, op.get("content") or "", workspace)
        return py_create_folder(target, workspace)

    if operation == "append":
        return py_append_file(op.get("target"), op.get("content") or "", workspace)

    if operation == "write":
        return py_write_file(op.get("target"), op.get("content") or "", workspace)

    if operation == "delete":
        target = op.get("target")
        if file_type == "file":
            return py_delete_file(target, workspace)
        return py_delete_folder(target, workspace)

    if operation == "move":
        return py_move_item(op.get("source"), op.get("destination"), workspace)

    return {"success": False, "error": f"Unknown operation: {operation}"}


# ------------------------------------------------------------------
# Per-operation verification (against the REAL filesystem)
# ------------------------------------------------------------------
def gather_pre_info(op, workspace):
    """Facts captured right before execution so verification can compare before/after."""
    info = {}
    operation = op.get("operation")
    if operation in ("create", "append", "write", "delete"):
        abs_t, _ = resolve_target(op.get("target"), workspace)
        info["abs"] = abs_t
        info["existed"] = bool(abs_t) and os.path.lexists(abs_t)
        if abs_t and os.path.isfile(abs_t):
            info["size"] = os.path.getsize(abs_t)
    elif operation == "move":
        src, _ = resolve_target(op.get("source"), workspace)
        dst, _ = resolve_target(op.get("destination"), workspace)
        if src and dst:
            dest_was_dir = os.path.isdir(dst) and dst != src
            info["src_abs"] = src
            info["expected_abs"] = os.path.join(dst, os.path.basename(src)) if dest_was_dir else dst
            info["tree"] = snapshot_tree(src)   # the source folder AND everything inside it
    return info


def verify_op(op, exec_result, pre_info, workspace):
    """Confirm an operation really did what it claims, by looking at the disk."""
    if not exec_result.get("success"):
        return {"completed": False, "evidence": exec_result.get("error", "Execution failed")}

    operation = op.get("operation")

    if operation == "memory":
        scope = exec_result.get("scope", op.get("memory_scope") or "shared")
        if exec_result.get("verified"):
            return {"completed": True, "evidence": f"{scope} memory '{op.get('memory_key')}' set and verified"}
        return {"completed": True, "evidence": f"{scope} memory op: {op.get('memory_action')} {op.get('memory_key')}"}

    if operation == "terminal":
        return {"completed": True, "evidence": f"Command finished with exit code {exec_result.get('exit_code')}"}

    if operation == "list":
        if exec_result.get("is_file"):
            return {"completed": True, "evidence": f"Listed file: {exec_result.get('path')}"}
        return {"completed": True,
                "evidence": f"Listed {len(exec_result.get('folders', []))} folders, {len(exec_result.get('files', []))} files"}

    if operation == "read":
        return {"completed": True, "evidence": f"Read file: {op.get('target')}"}

    if operation in ("create", "append", "write", "delete"):
        abs_t = pre_info.get("abs")
        label = op.get("target")
        if not abs_t:
            return {"completed": False, "evidence": f"Could not resolve '{label}' for verification"}

        if operation == "create":
            if pre_info.get("existed"):
                return {"completed": False, "evidence": f"'{label}' already existed before create"}
            if op.get("type", "folder") == "file":
                if not os.path.isfile(abs_t):
                    return {"completed": False, "evidence": f"File '{label}' not found after create"}
                expected = op.get("content") or ""
                if _read_text(abs_t) != expected:
                    return {"completed": False, "evidence": f"File '{label}' created but content mismatch"}
                return {"completed": True, "evidence": f"File '{label}' created with correct content"}
            if os.path.isdir(abs_t):
                return {"completed": True, "evidence": f"Folder '{label}' created"}
            return {"completed": False, "evidence": f"Folder '{label}' not found after create"}

        if operation == "append":
            if not os.path.isfile(abs_t):
                return {"completed": False, "evidence": f"File '{label}' missing after append"}
            added = op.get("content") or ""
            if os.path.getsize(abs_t) != pre_info.get("size", 0) + len(added.encode("utf-8")):
                return {"completed": False, "evidence": f"Append to '{label}': size does not match expectation"}
            if added and not _read_text(abs_t).endswith(added):
                return {"completed": False, "evidence": f"Append to '{label}': content verification failed"}
            return {"completed": True, "evidence": f"Content appended to '{label}'"}

        if operation == "write":
            if not os.path.isfile(abs_t):
                return {"completed": False, "evidence": f"File '{label}' missing after write"}
            if _read_text(abs_t) != (op.get("content") or ""):
                return {"completed": False, "evidence": f"File '{label}' written but content mismatch"}
            return {"completed": True, "evidence": f"File '{label}' written with correct content"}

        if operation == "delete":
            if pre_info.get("existed") and not os.path.lexists(abs_t):
                return {"completed": True, "evidence": f"'{label}' deleted"}
            return {"completed": False, "evidence": f"'{label}' not removed as expected"}

    if operation == "move":
        src, expected = pre_info.get("src_abs"), pre_info.get("expected_abs")
        tree = pre_info.get("tree") or {}
        if not src or not expected:
            return {"completed": False, "evidence": "Could not resolve move paths for verification"}
        if os.path.lexists(src):
            return {"completed": False, "evidence": f"Source '{op.get('source')}' still exists after move"}
        if not os.path.lexists(expected):
            return {"completed": False, "evidence": f"Destination '{os.path.relpath(expected, workspace)}' not found after move"}

        # Verify the destination AND everything that was inside the source
        problems = []
        for rel, size in tree.items():
            p = expected if rel == "." else os.path.join(expected, rel)
            if size is None:
                if not os.path.isdir(p):
                    problems.append(f"missing folder {rel}")
            else:
                if not os.path.isfile(p):
                    problems.append(f"missing file {rel}")
                elif os.path.getsize(p) != size:
                    problems.append(f"size changed for {rel}")
            if len(problems) >= 5:
                break
        if problems:
            return {"completed": False, "evidence": "Move incomplete at destination: " + "; ".join(problems)}
        count = max(len(tree) - 1, 0) if tree.get(".") is None and "." in tree else len(tree)
        return {"completed": True,
                "evidence": f"'{op.get('source')}' moved to '{os.path.relpath(expected, workspace)}' ({count} item(s) inside verified)"}

    return {"completed": False, "evidence": f"Unknown operation: {operation}"}


# ------------------------------------------------------------------
# Unified plan runner
# ------------------------------------------------------------------
def run_plan(operations, chat_dir):
    """
    Validate + (confirm destructive ops and terminal commands) + execute + verify a sequence.
    Returns (sequence_complete, execution_log, errors).
    """
    # --- Validate against projected state ---
    print("\n✔️  Validating operation sequence...")
    inspect = inspect_directory_recursive(chat_dir)
    projected = {normalize_rel(p) for p in inspect.get("folders", []) + inspect.get("files", [])}

    validation_errors = []
    for i, op in enumerate(operations, 1):
        problem = None
        path_errs = validate_op_paths(op, chat_dir)
        if path_errs:
            problem = " | ".join(path_errs)
        if problem is None:
            ok, err, projected = apply_op_to_projected(op, projected, chat_dir)
            if not ok:
                problem = err
        if problem:
            print(f"   ❌ Op {i} validation: {problem}")
            validation_errors.append({"operation": op, "error": problem})
            return False, [], validation_errors
        print(f"   ✅ Op {i}: OK")

    # --- Confirm destructive operations (delete/move) ---
    if any(op.get("operation") in ("delete", "move") for op in operations):
        print("\n⚠️  Plan includes destructive operations (delete/move).")
        for op in operations:
            if op.get("operation") == "delete":
                print(f"   - delete {op.get('type', 'folder')}: {op.get('target')}")
            elif op.get("operation") == "move":
                print(f"   - move: {op.get('source')} → {op.get('destination')}")
        if input("   Proceed? (yes/no): ").strip().lower() != "yes":
            print("   ❌ Cancelled")
            return False, [], [{"operation": None, "error": "User cancelled", "cancelled": True}]

    # --- Approve terminal commands ---
    if TERMINAL_REQUIRE_APPROVAL:
        for op in operations:
            if op.get("operation") == "terminal":
                print(f"\n🖥️  Paper wants to run a command in: {chat_dir}")
                print(f"   $ {op.get('command')}")
                if op.get("explanation"):
                    print(f"   Reason: {op.get('explanation')}")
                if input("   Run it? (yes/no): ").strip().lower() != "yes":
                    print("   ❌ Cancelled")
                    return False, [], [{"operation": None, "error": "User declined to run the command", "cancelled": True}]

    # --- Execute ---
    print("\n⚙️  Executing operations...")
    execution_log = []
    sequence_complete = True
    exec_errors = []

    for i, op in enumerate(operations, 1):
        label = op.get("target") or op.get("source") or op.get("memory_key") or op.get("command") or "?"
        print(f"\n   [{i}/{len(operations)}] {op.get('operation')} {op.get('type', '')}: {label}")

        pre_info = gather_pre_info(op, chat_dir)
        exec_result = execute_op(op, chat_dir)

        if not exec_result.get("success"):
            print(f"   ❌ Failed: {exec_result.get('error')}")
            if op.get("operation") == "terminal":
                if exec_result.get("stdout"):
                    print(f"   📝 stdout:\n{exec_result.get('stdout')}")
                if exec_result.get("stderr"):
                    print(f"   ⚠️  stderr:\n{exec_result.get('stderr')}")
            execution_log.append({"op": op, "result": exec_result, "verified": False})
            exec_errors.append({"operation": op, "error": exec_result.get("error")})
            sequence_complete = False
            break

        operation = op.get("operation")
        if operation == "list":
            if exec_result.get("is_file"):
                print(f"   📄 File target: {exec_result.get('path')}")
            else:
                print(f"   📁 Folders: {exec_result.get('folders', [])}")
                print(f"   📄 Files:   {exec_result.get('files', [])}")
        elif operation == "read":
            content = exec_result.get("content", "")
            print(f"   📄 Read: {op.get('target')}")
            print(f"   📝 Content:\n{content[:500] + ('...' if len(content) > 500 else '')}")
        elif operation == "memory":
            mscope = exec_result.get("scope", op.get("memory_scope") or "shared")
            mval = exec_result.get("value", "")
            print(f"   💾 Memory Op [{mscope}]: {op.get('memory_action')} {op.get('memory_key')}"
                  + (f" = {mval}" if op.get("memory_action") == "get" else ""))
        elif operation == "terminal":
            print(f"   🖥️  Exit code: {exec_result.get('exit_code')}")
            if exec_result.get("stdout"):
                print(f"   📝 stdout:\n{exec_result.get('stdout')}")
            if exec_result.get("stderr"):
                print(f"   ⚠️  stderr:\n{exec_result.get('stderr')}")

        verification = verify_op(op, exec_result, pre_info, chat_dir)
        verified = verification["completed"]
        print(f"   {'✅' if verified else '⚠️ '} {verification['evidence']}")
        execution_log.append({"op": op, "result": exec_result, "verified": verified})

        if not verified:
            sequence_complete = False
            exec_errors.append({"operation": op, "error": verification["evidence"]})
            break

    return sequence_complete, execution_log, exec_errors


# ------------------------------------------------------------------
# Web interface
# Serves ONLY index.html (from the folder next to paper.py) plus a small
# JSON API that index.html talks to. Nothing else on disk is reachable.
#
# The old terminal prompts (chat name, "Run it? (yes/no)", ...) become buttons
# in the page. All the logic above is unchanged: input() is simply routed to
# the browser while a message is being processed, and every print() is
# streamed into the page's "Activity" log (and still shows in the terminal).
# ------------------------------------------------------------------
APPROVAL_TIMEOUT = 600   # seconds to wait for a Yes/No click before treating it as "no"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(SCRIPT_DIR, "index.html")
IN_CODESPACES = os.environ.get("CODESPACES", "").lower() == "true"
SESSION_TOKEN = secrets.token_urlsafe(24)   # injected into index.html; required by the API
LOOPBACK_ONLY = True                        # set in serve()
SERVER = None

CHAT_STATE = {}                 # chat folder name -> {"last_log": ...}  (was `last_execution_log`)
JOBS = {}                       # job id -> Job
ACTIVE = {"job": None}
RUN_LOCK = threading.Lock()     # one message at a time (paper.py keeps session-wide state)

_tls = threading.local()
_real_stdout = sys.stdout
_builtin_input = builtins.input


class _Tee:
    """stdout that still prints to the terminal AND feeds the running job's log."""
    encoding = "utf-8"

    def write(self, s):
        try:
            _real_stdout.write(s)
        except Exception:
            pass
        job = getattr(_tls, "job", None)
        if job is not None:
            job.feed(s)
        return len(s)

    def flush(self):
        try:
            _real_stdout.flush()
        except Exception:
            pass

    def isatty(self):
        return False


sys.stdout = _Tee()


def input(prompt=""):  # noqa: A001 - intentionally shadows the builtin
    """While a job is running, questions go to the browser instead of the terminal."""
    job = getattr(_tls, "job", None)
    if job is None:
        return _builtin_input(prompt)
    return job.ask(prompt)


class Job:
    """One user message being processed in the background."""

    def __init__(self, chat):
        self.id = uuid.uuid4().hex[:12]
        self.chat = chat
        self.lines = []
        self._partial = ""
        self.done = False
        self.result = None
        self.error = None
        self.approval = None
        self.answer = None
        self.created = time.time()
        self.cond = threading.Condition()

    def feed(self, text):
        with self.cond:
            self._partial += text
            parts = self._partial.split("\n")
            self._partial = parts.pop()
            self.lines.extend(parts)

    def ask(self, prompt):
        with self.cond:
            recent = [l for l in self.lines if l.strip()][-6:]
            approval_id = uuid.uuid4().hex[:8]
            self.approval = {"id": approval_id, "prompt": str(prompt).strip(), "context": recent}
            self.answer = None
        print("   ⏳ Waiting for your answer in the browser...")
        with self.cond:
            deadline = time.time() + APPROVAL_TIMEOUT
            while self.answer is None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self.cond.wait(remaining)
            answer = self.answer or "no"
            self.approval = None
            self.answer = None
        print("   👍 Approved" if answer == "yes" else "   🚫 Declined")
        return answer

    def respond(self, approval_id, answer):
        with self.cond:
            if not self.approval or self.approval["id"] != approval_id:
                return False
            self.answer = "yes" if answer == "yes" else "no"
            self.cond.notify_all()
            return True

    def finish(self, result=None, error=None):
        with self.cond:
            if self._partial:
                self.lines.append(self._partial)
                self._partial = ""
            self.result = result
            self.error = error
            self.approval = None
            self.done = True

    def snapshot(self, since):
        with self.cond:
            return {
                "lines": self.lines[since:],
                "next": len(self.lines),
                "done": self.done,
                "approval": self.approval,
                "result": self.result,
                "error": self.error,
            }


# ------------------------------------------------------------------
# One full turn (this is what the old terminal loop did per typed line)
# ------------------------------------------------------------------
def handle_message(chat_dir, user_input, state):
    """Returns {"text": reply, "kind": "ok"|"warn"|"error", "technical": str|None}."""
    global CURRENT_CHAT_DIR
    CURRENT_CHAT_DIR = chat_dir

    ensure_identity_in_shared_memory()
    append_chat_message(chat_dir, "user", user_input)

    # STEP 1: Inspect filesystem
    print("\n🔍 Step 1: Inspecting filesystem...")
    initial_state = inspect_directory_recursive(chat_dir)
    if initial_state.get("error"):
        msg = f"Inspection failed: {initial_state['error']}"
        print(f"❌ {msg}")
        return {"text": msg, "kind": "error", "technical": None}
    print(f"   📁 Folders: {initial_state.get('folders', [])}")
    print(f"   📄 Files:   {initial_state.get('files', [])}")

    # STEP 2: Plan (consults BOTH chat memory and shared memory)
    print("\n🤖 Step 2: Planning (chat memory + shared memory loaded)...")
    history = load_chat_history(chat_dir)
    plan = plan_operations(user_input, initial_state, history,
                           last_execution_log=state.get("last_log"))

    if plan.get("is_question"):
        response = plan.get("response", "(no response)")
        print(f"\n🤖 {response}")
        append_chat_message(chat_dir, "assistant", response)
        state["last_log"] = None
        return {"text": response, "kind": "ok", "technical": None}

    operations = plan.get("operations") or []
    if not operations:
        print("ℹ️  No operations needed — the task appears to be already complete.")
        msg = "No operations were needed — the requested state already exists."
        append_chat_message(chat_dir, "assistant", msg)
        return {"text": msg, "kind": "ok", "technical": None}

    print(f"   Planned {len(operations)} operation(s):")
    for i, op in enumerate(operations, 1):
        label = op.get("target") or op.get("source") or op.get("memory_key") or op.get("command")
        extra = f" [{op.get('memory_scope')}]" if op.get("operation") == "memory" and op.get("memory_scope") else ""
        print(f"   {i}. {op.get('operation')} ({op.get('type', '')}){extra} → {label}")

    # STEP 3: Access check — each path outside the chat folder needs its own approval
    print("\n🔒 Step 3: Checking workspace boundaries...")
    ok, msg = ensure_access(operations, chat_dir)
    if not ok:
        print(f"   ❌ {msg}")
        text = f"I couldn't do that: {msg}"
        append_chat_message(chat_dir, "assistant", text)
        return {"text": text, "kind": "warn", "technical": None}

    # STEPS 4-6: Validate + Execute + Verify
    sequence_complete, execution_log, errors = run_plan(operations, chat_dir)
    state["last_log"] = execution_log if execution_log else None

    # Recovery loop (never retries something the user declined)
    recovery_attempts = 0
    MAX_RECOVERY_ATTEMPTS = 2
    while (not sequence_complete and errors
           and not any(e.get("cancelled") for e in errors)
           and recovery_attempts < MAX_RECOVERY_ATTEMPTS):
        recovery_attempts += 1
        print(f"\n🔄 Recovery attempt {recovery_attempts}/{MAX_RECOVERY_ATTEMPTS}...")

        current_inspect = inspect_directory_recursive(chat_dir)
        recovery_plan = plan_operations(user_input, current_inspect, history,
                                        execution_errors=errors,
                                        last_execution_log=execution_log)

        if recovery_plan.get("is_question") or not recovery_plan.get("operations"):
            if not recovery_plan.get("operations") and not recovery_plan.get("is_question"):
                print("   ℹ️  Recovery determined no operations are needed (task may already be complete).")
                sequence_complete = True
                break
            print("   ❌ AI could not formulate a recovery plan.")
            break

        recovery_ops = recovery_plan.get("operations", [])
        print(f"   🤖 Recovery plan received with {len(recovery_ops)} operation(s):")
        for i, op in enumerate(recovery_ops, 1):
            label = op.get("target") or op.get("source") or op.get("memory_key") or op.get("command")
            print(f"   {i}. {op.get('operation')} ({op.get('type', '')}) → {label}")

        ok, msg = ensure_access(recovery_ops, chat_dir)
        if not ok:
            print(f"   ❌ Recovery plan rejected: {msg}")
            break

        recovery_complete, recovery_log, errors = run_plan(recovery_ops, chat_dir)
        execution_log.extend(recovery_log)
        sequence_complete = recovery_complete
        state["last_log"] = execution_log

    # STEP 7: Summary
    print("\n📊 Step 7: Sequence summary")
    succeeded = sum(1 for e in execution_log if e["verified"])
    print(f"   Operations executed: {len(execution_log)}")
    print(f"   Verified successful: {succeeded}")

    final_state = inspect_directory_recursive(chat_dir)
    print(f"\n📁 Final folders: {final_state.get('folders', [])}")
    print(f"📄 Final files:   {final_state.get('files', [])}")

    summary_lines = []
    if sequence_complete:
        summary_lines.append(f"✅ ALL OPERATIONS COMPLETED ({len(execution_log)} op(s)).")
    else:
        summary_lines.append(f"⚠️  Sequence incomplete — stopped after {len(execution_log)} op(s).")

    for e in execution_log:
        op = e["op"]
        label = op.get("target") or op.get("source") or op.get("memory_key") or op.get("command") or "?"
        status = "✅" if e["verified"] else "❌"
        detail = ""
        if not e["result"].get("success"):
            detail = f" — ERROR: {e['result'].get('error', 'unknown')}"
        scope_tag = ""
        if op.get("operation") == "memory":
            scope_tag = f" [{e['result'].get('scope') or op.get('memory_scope') or 'shared'}]"
        summary_lines.append(f"   {status} {op.get('operation', '?')} {op.get('type', '')}{scope_tag} → {label}{detail}")

    technical_summary = "\n".join(summary_lines)
    print(f"\n{technical_summary}")
    print("-" * 55)

    # Natural-language reply from the REAL results (this is what future turns will read)
    natural = summarize_for_user(user_input, execution_log, sequence_complete, errors)
    if natural:
        print(f"\n🤖 {natural}")
    stored_reply = natural or technical_summary

    append_chat_message(chat_dir, "assistant", stored_reply, metadata={
        "technical_summary": technical_summary,
        "operations": [
            {
                "operation": e["op"].get("operation"),
                "type": e["op"].get("type"),
                "target": e["op"].get("target"),
                "source": e["op"].get("source"),
                "destination": e["op"].get("destination"),
                "command": e["op"].get("command"),
                "memory_scope": e["op"].get("memory_scope"),
                "success": e["result"].get("success"),
                "verified": e["verified"],
                "exit_code": e["result"].get("exit_code"),
                "error": e["result"].get("error") if not e["result"].get("success") else None,
            }
            for e in execution_log
        ],
        "approved_outside_paths": sorted(APPROVED_OUTSIDE),
        "sequence_complete": sequence_complete,
    })
    return {"text": stored_reply, "kind": "ok" if sequence_complete else "warn",
            "technical": technical_summary}


# ------------------------------------------------------------------
# Chat management (what the old "Enter chat name" prompt did)
# ------------------------------------------------------------------
def _clean_chat_name(raw):
    if not isinstance(raw, str):
        return None
    name = re.sub(r"[^\w\-. ]+", "_", raw.strip()).strip()
    if not name or name in (".", ".."):
        return None
    return name[:80]


def chat_path(name):
    """Exact-name lookup of an existing chat folder (also blocks path tricks)."""
    if not isinstance(name, str) or not name:
        return None
    try:
        for e in os.listdir(CHATS_DIR):
            p = os.path.join(CHATS_DIR, e)
            if e == name and os.path.isdir(p):
                return p
    except OSError:
        pass
    return None


def chat_paths(chat_dir):
    return {
        "chat_dir": chat_dir,
        "files_dir": os.path.join(chat_dir, "files"),
        "history": os.path.join(chat_dir, "chat_history.json"),
        "memory": chat_memory_path(chat_dir),
    }


def chat_payload(chat_dir):
    return {
        "name": os.path.basename(chat_dir),
        "history": load_chat_history(chat_dir),
        "paths": chat_paths(chat_dir),
    }


def list_chats():
    out = []
    try:
        entries = os.listdir(CHATS_DIR)
    except OSError:
        return out
    for e in entries:
        p = os.path.join(CHATS_DIR, e)
        if not os.path.isdir(p):
            continue
        hist_file = os.path.join(p, "chat_history.json")
        try:
            mtime = os.path.getmtime(hist_file if os.path.isfile(hist_file) else p)
        except OSError:
            mtime = 0
        hist = load_chat_history(p)
        out.append({"name": e, "messages": len(hist.get("messages", [])),
                    "created": hist.get("created"), "_m": mtime})
    out.sort(key=lambda c: c["_m"], reverse=True)
    for c in out:
        c.pop("_m", None)
    return out


def open_or_create_chat(raw_name):
    """Same rules as before: case-insensitive match on every OS, else create. Returns (payload, error)."""
    safe_name = _clean_chat_name(raw_name)
    if not safe_name:
        return None, "Invalid chat name."

    notes = []
    match, has_duplicates = find_existing_chat(safe_name)
    existed = False
    if match:
        if match != safe_name:
            notes.append(f"Matched existing chat '{match}' (chat names are case-insensitive).")
        if has_duplicates:
            notes.append("Several chat folders differ only by letter case; using the best match. "
                         "Consider merging or renaming them.")
        safe_name = match
        existed = True

    chat_dir = os.path.join(CHATS_DIR, safe_name)
    if os.path.exists(chat_dir) and not os.path.isdir(chat_dir):
        return None, "A file with that name exists in the chats folder. Pick another name."

    os.makedirs(chat_dir, exist_ok=True)
    os.makedirs(os.path.join(chat_dir, "files"), exist_ok=True)
    if not os.path.isfile(chat_memory_path(chat_dir)):
        save_chat_memory({}, chat_dir)
    history = load_chat_history(chat_dir)
    if not history.get("messages"):
        history["chat_name"] = safe_name
        history["created"] = datetime.now().isoformat()
        save_chat_history(chat_dir, history)

    payload = chat_payload(chat_dir)
    payload["existed"] = existed
    payload["notes"] = notes
    return payload, None


def rename_chat(old_name, raw_new):
    old_dir = chat_path(old_name)
    if not old_dir:
        return None, "Chat not found."
    new_name = _clean_chat_name(raw_new)
    if not new_name:
        return None, "Invalid chat name."
    if new_name == old_name:
        return {"name": old_name}, None
    try:
        for e in os.listdir(CHATS_DIR):
            if e != old_name and e.casefold() == new_name.casefold():
                return None, f"A chat named '{e}' already exists (chat names are case-insensitive)."
    except OSError as e:
        return None, str(e)
    new_dir = os.path.join(CHATS_DIR, new_name)
    try:
        os.rename(old_dir, new_dir)
        history = load_chat_history(new_dir)
        history["chat_name"] = new_name
        save_chat_history(new_dir, history)
    except OSError as e:
        return None, str(e)
    if old_name in CHAT_STATE:
        CHAT_STATE[new_name] = CHAT_STATE.pop(old_name)
    return {"name": new_name}, None


# ------------------------------------------------------------------
# Jobs
# ------------------------------------------------------------------
def _run_job(job, chat_dir, text):
    _tls.job = job
    result, error = None, None
    try:
        state = CHAT_STATE.setdefault(os.path.basename(chat_dir), {"last_log": None})
        result = handle_message(chat_dir, text, state)
    except BaseException as e:  # keep the server alive no matter what the model returned
        import traceback
        traceback.print_exc(file=_real_stdout)
        error = f"{type(e).__name__}: {e}"
        print(f"❌ {error}")
    finally:
        _tls.job = None
        ACTIVE["job"] = None
        RUN_LOCK.release()          # release first so the page can send again the moment it sees "done"
        job.finish(result=result, error=error)


def start_job(chat_name, text):
    """Returns (job, error_code). error_code: 'nochat' or 'busy'."""
    chat_dir = chat_path(chat_name)
    if not chat_dir:
        return None, "nochat"
    if not RUN_LOCK.acquire(blocking=False):
        return None, "busy"
    for jid in [j for j, jb in JOBS.items() if jb.done and time.time() - jb.created > 3600]:
        JOBS.pop(jid, None)
    job = Job(os.path.basename(chat_dir))
    JOBS[job.id] = job
    ACTIVE["job"] = job
    threading.Thread(target=_run_job, args=(job, chat_dir, text), daemon=True).start()
    return job, None


# ------------------------------------------------------------------
# HTTP server
# ------------------------------------------------------------------
def state_payload():
    active = ACTIVE["job"]
    return {
        "version": "1.0",
        "ai_name": AI_NAME,
        "model": MODEL_NAME,
        "identity": IDENTITY_VALUE,
        "root": ROOT_WORKSPACE,
        "chats_dir": CHATS_DIR,
        "shared_memory": SHARED_MEMORY_FILE,
        "terminal": {"approval": TERMINAL_REQUIRE_APPROVAL, "timeout": TERMINAL_TIMEOUT},
        "approved_outside": sorted(APPROVED_OUTSIDE),
        "chats": list_chats(),
        "active": {"job": active.id, "chat": active.chat} if active else None,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "SiliconPaper/1.0"

    def log_message(self, fmt, *args):   # polling is chatty; keep the terminal readable
        pass

    # ---- helpers ----
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else str(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def _err(self, code, msg):
        self._json(code, {"error": msg})

    def _host_ok(self):
        """Blocks DNS-rebinding when only listening on this machine."""
        if not LOOPBACK_ONLY:
            return True
        host = (self.headers.get("Host") or "").lower()
        name = host.split("]")[0] + "]" if host.startswith("[") else host.rsplit(":", 1)[0]
        return name in ("localhost", "127.0.0.1", "[::1]")

    def _authed(self):
        tok = self.headers.get("X-Paper-Token") or ""
        return secrets.compare_digest(tok, SESSION_TOKEN)

    def _body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if n > 1_000_000:
            return None
        raw = self.rfile.read(n) if n else b""
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    # ---- routes ----
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._host_ok():
            return self._err(403, "Forbidden host")
        url = urlparse(self.path)
        path, q = url.path, parse_qs(url.query)

        if path in ("/", "/index.html"):
            try:
                with open(INDEX_FILE, "r", encoding="utf-8") as f:
                    html = f.read()
            except OSError:
                return self._send(500, "index.html was not found next to paper.py.",
                                  "text/plain; charset=utf-8")
            html = html.replace("__PAPER_TOKEN__", SESSION_TOKEN)
            return self._send(200, html, "text/html; charset=utf-8")

        if path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")

        if not path.startswith("/api/"):
            return self._send(404, "Not found", "text/plain; charset=utf-8")
        if not self._authed():
            return self._err(403, "Missing or wrong session token. Reload the page.")

        try:
            if path == "/api/state":
                return self._json(200, state_payload())

            if path == "/api/chat":
                chat_dir = chat_path((q.get("name") or [""])[0])
                if not chat_dir:
                    return self._err(404, "Chat not found.")
                return self._json(200, chat_payload(chat_dir))

            if path == "/api/job":
                job = JOBS.get((q.get("id") or [""])[0])
                if not job:
                    return self._err(404, "Unknown job.")
                try:
                    since = max(0, int((q.get("since") or ["0"])[0]))
                except ValueError:
                    since = 0
                return self._json(200, job.snapshot(since))
        except Exception as e:
            return self._err(500, f"{type(e).__name__}: {e}")

        return self._err(404, "Unknown endpoint.")

    def do_POST(self):
        if not self._host_ok():
            return self._err(403, "Forbidden host")
        path = urlparse(self.path).path
        if not path.startswith("/api/"):
            return self._send(404, "Not found", "text/plain; charset=utf-8")
        if not self._authed():
            return self._err(403, "Missing or wrong session token. Reload the page.")
        body = self._body()
        if body is None:
            return self._err(400, "Bad request body.")

        try:
            if path == "/api/chat":                      # open or create (case-insensitive)
                payload, err = open_or_create_chat(body.get("name"))
                if err:
                    return self._err(400, err)
                return self._json(200, payload)

            if path == "/api/rename":
                if RUN_LOCK.locked():
                    return self._err(409, "Paper is busy. Wait for the current task to finish.")
                res, err = rename_chat(body.get("name"), body.get("new_name"))
                if err:
                    return self._err(400, err)
                return self._json(200, res)

            if path == "/api/message":
                text = body.get("text")
                if not isinstance(text, str) or not text.strip():
                    return self._err(400, "Message is empty.")
                if len(text) > 20000:
                    return self._err(400, "Message is too long (20,000 characters max).")
                job, err = start_job(body.get("chat"), text.strip())
                if err == "nochat":
                    return self._err(404, "Chat not found. Create or open a chat first.")
                if err == "busy":
                    return self._err(409, "Paper is still working on the previous message.")
                return self._json(200, {"job": job.id})

            if path == "/api/approve":
                job = JOBS.get(body.get("job") or "")
                if not job or not job.respond(body.get("approval"), body.get("answer")):
                    return self._err(409, "That approval request is no longer waiting.")
                return self._json(200, {"ok": True})

            if path == "/api/shutdown":
                if RUN_LOCK.locked():
                    return self._err(409, "Paper is busy. Answer or wait for the current task first.")
                self._json(200, {"ok": True})
                if SERVER is not None:
                    threading.Thread(target=SERVER.shutdown, daemon=True).start()
                return
        except Exception as e:
            return self._err(500, f"{type(e).__name__}: {e}")

        return self._err(404, "Unknown endpoint.")


class PaperServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets two programs share a port, which would defeat the port search.
    allow_reuse_address = (os.name != "nt")

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def _should_open_browser():
    if os.environ.get("PAPER_NO_BROWSER"):
        return False
    if sys.platform.startswith("linux") and not (
            os.environ.get("BROWSER") or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False    # headless Linux: webbrowser could launch a text browser in this terminal
    return True


def serve():
    global SERVER, LOOPBACK_ONLY
    ensure_identity_in_shared_memory()

    if not os.path.isfile(INDEX_FILE):
        print(f"⚠️  index.html not found at {INDEX_FILE} — put it next to paper.py.")

    # Codespaces forwards ports from outside the container, so listen on all interfaces there.
    # On a normal computer listen on this machine only (other devices can't reach it).
    host = os.environ.get("PAPER_HOST") or ("0.0.0.0" if IN_CODESPACES else "127.0.0.1")
    LOOPBACK_ONLY = host in ("127.0.0.1", "localhost", "::1")
    try:
        first_port = int(os.environ.get("PAPER_PORT", "8000"))
    except ValueError:
        first_port = 8000

    server = None
    for port in range(first_port, first_port + 20):
        try:
            server = PaperServer((host, port), Handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"❌ No free port in {first_port}-{first_port + 19}. "
                         "Set PAPER_PORT in .env to another number.")
    SERVER = server
    port = server.server_address[1]
    local_url = f"http://localhost:{port}"

    print(f"\n📂 Chats folder   : {CHATS_DIR}")
    print(f"💾 Shared memory  : {SHARED_MEMORY_FILE}")
    print(f"🤖 Model          : {MODEL_NAME}")
    print(f"🪪 Identity       : {IDENTITY_VALUE}")
    print(f"🔒 File access    : chat folder only; outside paths need per-path approval")
    print(f"🖥️  Terminal       : ANY command ({'approval required' if TERMINAL_REQUIRE_APPROVAL else 'NO approval'}, "
          f"{TERMINAL_TIMEOUT}s timeout)")
    print("=" * 55)
    print(f"🌐 Serving index.html (and nothing else) on port {port}")
    print(f"   Open: {local_url}")
    if IN_CODESPACES:
        cs_name = os.environ.get("CODESPACE_NAME")
        cs_domain = os.environ.get("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN", "app.github.dev")
        if cs_name:
            print(f"   Codespaces link: https://{cs_name}-{port}.{cs_domain}")
        print("   (Or open the PORTS tab and click the globe icon. Keep the port Private.)")
    print("   Ctrl+C or the ⏻ button in the page powers down.")
    print("=" * 55)

    if _should_open_browser():
        threading.Timer(0.8, lambda: webbrowser.open(local_url)).start()

    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("\n⚡ Silicon shutting down")


if __name__ == "__main__":
    serve()
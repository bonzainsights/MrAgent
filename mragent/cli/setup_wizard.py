"""Interactive setup wizard for MRAgent — configures NVIDIA NIM key and model.

Runs entirely in the same terminal window using prompt_toolkit (no new windows).
Arrow-key navigation for all selections.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import ValidationError, Validator
from rich.console import Console

console = Console()

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"

# Model family prefixes — used for filtering
FAMILIES: list[tuple[str, str]] = [
    ("all", "All families"),
    ("meta/", "Meta / Llama"),
    ("qwen/", "Qwen"),
    ("nvidia/", "NVIDIA"),
    ("mistral/", "Mistral"),
    ("google/", "Google"),
    ("microsoft/", "Microsoft"),
]

# ---------------------------------------------------------------------------
# Arrow-key UI helpers (prompt_toolkit custom app)
# ---------------------------------------------------------------------------

async def arrow_select(title: str, choices: list[tuple[str, str]], add_back: bool = False) -> str | None:
    """Simple up/down arrow selector that returns immediately on Enter."""
    if add_back:
        choices = [("BACK", "⬅  Go Back")] + choices

    current_index = 1 if (add_back and len(choices) > 1) else 0

    kb = KeyBindings()

    @kb.add("up")
    def _(event):
        nonlocal current_index
        current_index = max(0, current_index - 1)

    @kb.add("down")
    def _(event):
        nonlocal current_index
        current_index = min(len(choices) - 1, current_index + 1)

    @kb.add("enter")
    def _(event):
        event.app.exit(result=choices[current_index][0])

    @kb.add("escape")
    @kb.add("c-c")
    def _(event):
        event.app.exit(result=None)

    def get_text():
        fragments = [("class:title", f"  {title}\n\n")]
        for i, (val, label) in enumerate(choices):
            if i == current_index:
                fragments.append(("class:selected", f"  ❯ {label}\n"))
            else:
                fragments.append(("", f"    {label}\n"))
        
        fragments.append(("class:footer", "\n  (Use ↑↓ arrows to move, Enter to select, Esc to cancel)\n"))
        return fragments

    style = Style.from_dict({
        "title": "bold cyan",
        "selected": "bold green",
        "footer": "dim",
    })

    layout = Layout(Window(FormattedTextControl(get_text)))
    app = Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        full_screen=False,
    )
    # Print empty line before rendering so it sits cleanly
    print()
    return await app.run_async()


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


async def fetch_nvidia_models(api_key: str) -> list[dict]:
    """Fetch available models from NVIDIA NIM API.

    Returns list of model dicts with at least an 'id' field.
    Raises httpx.HTTPStatusError on API errors.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{NVIDIA_API_BASE}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        # Sort by id for consistent display
        return sorted(models, key=lambda m: m.get("id", "").lower())


def filter_models_by_prefix(models: list[dict], prefix: str | None) -> list[dict]:
    """Filter models by org prefix (e.g. 'meta/', 'qwen/'). None = all."""
    if prefix is None:
        return models
    return [m for m in models if m.get("id", "").startswith(prefix)]


def _count(models: list[dict], prefix: str) -> int:
    return sum(1 for m in models if m.get("id", "").startswith(prefix))


async def arrow_select_family(models: list[dict]) -> str | None:
    """Show family filter picker. Returns prefix string (or 'all'), or None if cancelled."""
    choices = []
    for prefix, label in FAMILIES:
        if prefix == "all":
            choices.append(("all", f"{label}  ({len(models)} models)"))
        else:
            count = _count(models, prefix)
            if count > 0:
                choices.append((prefix, f"{label}  ({count} models)"))

    return await arrow_select("Step 2/3 — Select Model Family", choices)


async def arrow_select_model(models: list[dict], family_label: str) -> str | None:
    """Show model picker from filtered list. Returns model id or None if cancelled."""
    choices = [(m["id"], m["id"]) for m in models]
    return await arrow_select(f"Step 3/3 — Select Model [{family_label}]", choices, add_back=True)


# ---------------------------------------------------------------------------
# Key validation
# ---------------------------------------------------------------------------


class _NvidiaKeyValidator(Validator):
    def validate(self, document) -> None:
        text = document.text.strip()
        if not text:
            raise ValidationError(message="API key cannot be empty")
        if not text.startswith("nvapi-"):
            raise ValidationError(message="NVIDIA NIM keys start with 'nvapi-' — get one at build.nvidia.com")


# ---------------------------------------------------------------------------
# Main wizard
# ---------------------------------------------------------------------------


async def run_setup_wizard() -> bool:
    """Run the full interactive NVIDIA NIM setup wizard.

    Returns True on success, False if the user cancelled or an error occurred.
    """
    console.print()
    console.print("[bold cyan]🤖  MRAgent Setup Wizard[/bold cyan]")
    console.print("[dim]Configure NVIDIA NIM — free credits at [link=https://build.nvidia.com]https://build.nvidia.com[/link][/dim]")
    console.print()

    # ── Step 1: API key ──────────────────────────────────────────────────────
    console.print("[bold dim]Step 1/3[/bold dim] — Enter your NVIDIA NIM API key")
    console.print("[dim]  The key will be saved to ~/.mragent/config.json (local only, never sent anywhere)[/dim]")
    console.print()

    try:
        session = PromptSession()
        api_key = await session.prompt_async(
            HTML("<ansicyan><b>  API Key</b></ansicyan> (nvapi-...): "),
            is_password=True,
            validator=_NvidiaKeyValidator(),
            validate_while_typing=False,
        )
        api_key = api_key.strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[yellow]Setup cancelled.[/yellow]")
        return False

    console.print()

    # ── Step 2a: Fetch models ──────────────────────────────────────────────
    console.print("[bold dim]Step 2/3[/bold dim] — Fetching available models from NVIDIA...")
    models: list[dict] = []
    with console.status("[dim]  Connecting to https://integrate.api.nvidia.com ...[/dim]", spinner="dots"):
        try:
            models = await fetch_nvidia_models(api_key)
        except httpx.HTTPStatusError as e:
            console.print(f"\n[red]✗ API error {e.response.status_code}[/red] — please check your key and try again.")
            return False
        except httpx.ConnectError:
            console.print("\n[red]✗ Could not connect to NVIDIA API.[/red] Check your internet connection.")
            return False
        except Exception as e:  # noqa: BLE001
            console.print(f"\n[red]✗ Error: {e}[/red]")
            return False

    console.print(f"[green]  ✓[/green] Found [bold]{len(models)}[/bold] available models")
    console.print()

    while True:
        # ── Step 2b: Family filter ────────────────────────────────────────────
        family_key = await arrow_select_family(models)
        if family_key is None:
            console.print("[yellow]Setup cancelled.[/yellow]")
            return False

        prefix = None if family_key == "all" else family_key
        family_label = next((lbl for k, lbl in FAMILIES if k == family_key), "All")
        filtered = filter_models_by_prefix(models, prefix)

        if not filtered:
            console.print(f"[yellow]No models found for family '{family_label}'.[/yellow]")
            return False

        # ── Step 3: Model selection ──────────────────────────────────────────
        selected_model = await arrow_select_model(filtered, family_label)
        if selected_model is None:
            console.print("[yellow]Setup cancelled.[/yellow]")
            return False
            
        if selected_model == "BACK":
            continue
            
        break

    console.print()

    # ── Save config ───────────────────────────────────────────────────────
    from mragent.config.loader import get_config_path, load_config, save_config

    config = load_config()
    config.providers.nvidia_nim.api_key = api_key
    config.agents.defaults.model = selected_model
    save_config(config)

    config_path = get_config_path()
    console.print(f"[green]✓[/green] Key + model saved to [cyan]{config_path}[/cyan]")
    console.print(f"[green]✓[/green] Provider : [bold]NVIDIA NIM[/bold]")
    console.print(f"[green]✓[/green] Model    : [bold cyan]{selected_model}[/bold cyan]")
    console.print()
    console.print("Next steps:")
    console.print("  [cyan]mragent agent[/cyan]   — chat in terminal")
    console.print("  [cyan]mragent web[/cyan]     — open web UI at [link=http://localhost:6326]http://localhost:6326[/link]")
    console.print()
    return True

"""ClipMind Command Line Interface & Interactive Startup Wizard.

Provides:
- Interactive setup wizard for first-time configuration:
  * Niche selection (Tech & AI, Finance, True Crime, Psychology, etc.)
  * Audience timezone selection
  * Upload frequency interval (default 12 hours)
  * Multi-service API key configuration (YouTube, OpenAI, SerpAPI, ElevenLabs)
  * Live health ping verification of all keys
- Operational commands:
  * --cycle: Execute a single autonomous production and publish cycle
  * --daemon: Run continuous scheduler loop (APScheduler)
  * --status: View active key pools, quotas, and recent cycle logs
  * --verify-keys: Test live connectivity for all configured API keys
  * --wizard: Launch interactive configuration wizard
  * --reset-quotas: Reset daily quota usage to 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from clipmind.core.config import Config, get_config
from clipmind.core.key_rotator import KeyRotator, get_key_rotator, mask_key
from clipmind.core.logger import get_logger
from clipmind.orchestrator import MasterOrchestrator

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    console = Console()
except ImportError:
    console = None  # type: ignore

logger = get_logger("clipmind.cli")

NICHE_CHOICES = [
    "Tech & AI",
    "Personal Finance & Crypto",
    "True Crime & Mystery",
    "Psychology & Human Behavior",
    "Productivity & Stoicism",
    "Custom Niche",
]

TIMEZONE_CHOICES = [
    "America/New_York",
    "America/Los_Angeles",
    "UTC",
    "Europe/London",
    "Asia/Kolkata",
    "Asia/Tokyo",
]


def print_banner() -> None:
    banner_text = """
   ____ _ _       __  __ _           _ 
  / ___| (_)_ __ |  \\/  (_)_ __   __| |
 | |   | | | '_ \\| |\\/| | | '_ \\ / _` |
 | |___| | | |_) | |  | | | | | | (_| |
  \\____|_|_| .__/|_|  |_|_|_| |_|\\__,_|
           |_|                         
  Autonomous YouTube Shorts Engine v1.0.0
    """
    if console:
        console.print(Panel(banner_text.strip(), style="bold cyan", expand=False))
    else:
        print(banner_text)


def run_setup_wizard(config: Optional[Config] = None) -> Config:
    """Interactive wizard to guide user through first-time startup configuration."""
    cfg = config or get_config()
    print_banner()

    if console:
        console.print("\n[bold yellow]=== Step 1: Channel Niche ===[/bold yellow]")
        for idx, niche_opt in enumerate(NICHE_CHOICES, 1):
            console.print(f"  [cyan]{idx}[/cyan]. {niche_opt}")
        choice = Prompt.ask("\nSelect your channel niche", default="1")
    else:
        print("\n=== Step 1: Channel Niche ===")
        for idx, niche_opt in enumerate(NICHE_CHOICES, 1):
            print(f"  {idx}. {niche_opt}")
        choice = input("\nSelect your channel niche [1]: ") or "1"

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(NICHE_CHOICES) - 1:
            cfg.niche = NICHE_CHOICES[idx]
        elif idx == len(NICHE_CHOICES) - 1:
            custom = Prompt.ask("Enter custom niche") if console else input("Enter custom niche: ")
            cfg.niche = custom.strip() or "Tech & AI"
    except Exception:
        cfg.niche = "Tech & AI"

    if console:
        console.print(f"[green]✓ Niche set to: {cfg.niche}[/green]\n")
        console.print("[bold yellow]=== Step 2: Target Audience Timezone ===[/bold yellow]")
        for idx, tz_opt in enumerate(TIMEZONE_CHOICES, 1):
            console.print(f"  [cyan]{idx}[/cyan]. {tz_opt}")
        tz_choice = Prompt.ask("Select audience timezone", default="1")
    else:
        print("\n=== Step 2: Target Audience Timezone ===")
        for idx, tz_opt in enumerate(TIMEZONE_CHOICES, 1):
            print(f"  {idx}. {tz_opt}")
        tz_choice = input("Select audience timezone [1]: ") or "1"

    try:
        tz_idx = int(tz_choice) - 1
        if 0 <= tz_idx < len(TIMEZONE_CHOICES):
            cfg.audience_timezone = TIMEZONE_CHOICES[tz_idx]
            cfg.scheduling.audience_timezone = TIMEZONE_CHOICES[tz_idx]
    except Exception:
        cfg.audience_timezone = "America/New_York"

    if console:
        console.print(f"[green]✓ Timezone set to: {cfg.audience_timezone}[/green]\n")
        console.print("[bold yellow]=== Step 3: Upload Frequency (Hours) ===[/bold yellow]")
        freq_input = Prompt.ask("Enter upload interval in hours (e.g. 12 for twice daily)", default="12")
    else:
        print("\n=== Step 3: Upload Frequency (Hours) ===")
        freq_input = input("Enter upload interval in hours [12]: ") or "12"

    try:
        cfg.upload_frequency_hours = float(freq_input)
        cfg.scheduling.frequency_hours = float(freq_input)
    except ValueError:
        cfg.upload_frequency_hours = 12.0

    # Step 4: API Keys Entry & Health Ping Verification
    rotator = get_key_rotator()
    if console:
        console.print(f"[green]✓ Upload frequency set to: {cfg.upload_frequency_hours} hours[/green]\n")
        console.print("[bold yellow]=== Step 4: Multi-Provider API Keys Configuration ===[/bold yellow]")
        console.print("Press Enter to keep existing environment keys, or paste new keys below.\n")
    else:
        print(f"\n✓ Upload frequency set to: {cfg.upload_frequency_hours} hours\n")
        print("=== Step 4: Multi-Provider API Keys Configuration ===")

    services = [
        ("youtube", "YouTube Data API v3 Key"),
        ("openai", "OpenAI API Key (Scripting & Moderation)"),
        ("serpapi", "SerpAPI Key (Search & PAA Mining)"),
        ("elevenlabs", "ElevenLabs API Key (Voiceovers)"),
    ]

    for s_name, label in services:
        current_active = rotator.get_active_key(s_name)
        masked_curr = mask_key(current_active) if current_active and not current_active.startswith("sim_") else "<not configured>"
        prompt_msg = f"{label} [Current: {masked_curr}]"
        
        if console:
            val = Prompt.ask(prompt_msg, default="")
        else:
            val = input(f"{prompt_msg}: ")

        if val.strip():
            for k in val.replace(";", ",").split(","):
                k_clean = k.strip().strip("'\"")
                if k_clean:
                    rotator.add_key(s_name, k_clean)

    # Health Ping Verification
    if console:
        console.print("\n[bold cyan]Verifying API keys with live health pings...[/bold cyan]")
    else:
        print("\nVerifying API keys with live health pings...")

    verify_results = rotator.verify_all_keys()
    display_key_table(verify_results)

    # Persist configurations
    cfg.save_persistent_overrides()
    if console:
        console.print("\n[bold green]Configuration successfully saved to disk![/bold green]\n")
    else:
        print("\nConfiguration successfully saved to disk!\n")

    return cfg


def display_key_table(verify_results: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> None:
    """Render a formatted table of key health and quota status."""
    rotator = get_key_rotator()
    status_report = rotator.get_pool_status()

    if console:
        table = Table(title="ClipMind API Key Pools & Quotas", style="cyan")
        table.add_column("Service", style="bold")
        table.add_column("Key (Masked)")
        table.add_column("Quota Remaining")
        table.add_column("Pre-Flight (>15%)")
        table.add_column("Blacklist Status")
        table.add_column("Health Ping")

        services = status_report.get("services", {})
        for s_name, s_data in services.items():
            keys = s_data.get("keys", [])
            if not keys:
                table.add_row(s_name.upper(), "<none configured>", "N/A", "N/A", "OK", "Unverified")
                continue

            for k in keys:
                rem_pct = k.get("remaining_percent", 100.0)
                quota_str = f"{k.get('remaining_units', 0)} ({rem_pct:.1f}%)"
                pre_flight = "[green]PASS[/green]" if k.get("has_sufficient_quota") else "[bold red]FAIL (<15%)[/bold red]"
                bl_status = "[bold red]RATE LIMITED[/bold red]" if k.get("is_rate_limited") else "[green]ACTIVE[/green]"
                
                # Check health from verify_results if available
                health_str = "[green]HEALTHY[/green]" if k.get("is_valid") else f"[red]FAIL: {k.get('last_error', 'Invalid')}[/red]"
                table.add_row(s_name.upper(), k.get("key_masked", ""), quota_str, pre_flight, bl_status, health_str)

        console.print(table)
    else:
        print("\n=== ClipMind API Key Pools & Quotas ===")
        for s_name, s_data in status_report.get("services", {}).items():
            print(f"Service: {s_name.upper()} ({s_data.get('healthy_keys', 0)}/{s_data.get('total_keys', 0)} healthy)")
            for k in s_data.get("keys", []):
                print(f"  Key: {k.get('key_masked')} | Quota: {k.get('remaining_percent')}% | Valid: {k.get('is_valid')}")


def show_status() -> None:
    """Print comprehensive diagnostics: keys, scheduler config, and recent cycles."""
    print_banner()
    cfg = get_config()

    if console:
        console.print(Panel(
            f"[bold]Active Niche:[/bold] {cfg.niche}\n"
            f"[bold]Audience Timezone:[/bold] {cfg.audience_timezone}\n"
            f"[bold]Upload Frequency:[/bold] Every {cfg.upload_frequency_hours} hours\n"
            f"[bold]Dry-Run Simulation Mode:[/bold] {cfg.dry_run}\n"
            f"[bold]Project Directory:[/bold] {cfg.paths.project_root}",
            title="System Configuration",
            style="yellow",
        ))
    else:
        print(f"Niche: {cfg.niche} | Timezone: {cfg.audience_timezone} | Frequency: {cfg.upload_frequency_hours}h")

    display_key_table()

    # List recent cycle logs
    logs_dir = cfg.paths.cycle_logs_dir
    if logs_dir.exists():
        cycle_files = sorted(logs_dir.glob("cycle_*.json"), key=os.path.getmtime, reverse=True)[:5]
        if cycle_files:
            if console:
                console.print("\n[bold cyan]Recent Execution Cycles:[/bold cyan]")
                for cf in cycle_files:
                    try:
                        import json
                        with open(cf, "r", encoding="utf-8") as f:
                            c_data = json.load(f)
                        st = c_data.get("status", "UNKNOWN")
                        color = "green" if st == "COMPLETED" else "red"
                        console.print(
                            f"  • [bold]{c_data.get('cycle_id')}[/bold] | "
                            f"Topic: '{c_data.get('topic')[:40]}...' | "
                            f"Status: [{color}]{st}[/{color}] | "
                            f"Duration: {c_data.get('total_duration_seconds', 0)}s"
                        )
                    except Exception:
                        console.print(f"  • {cf.name}")
            else:
                print("\nRecent Execution Cycles:")
                for cf in cycle_files:
                    print(f"  • {cf.name}")


def main() -> None:
    """Main CLI entrypoint handling command flags and interactive execution."""
    parser = argparse.ArgumentParser(
        description="ClipMind: Autonomous YouTube Shorts Generator & Uploader"
    )
    parser.add_argument("--cycle", action="store_true", help="Execute a single autonomous cycle and exit")
    parser.add_argument("--daemon", action="store_true", help="Run the continuous 12-hour scheduler loop")
    parser.add_argument("--wizard", action="store_true", help="Launch interactive first-time setup wizard")
    parser.add_argument("--status", action="store_true", help="Display system status, key quotas, and diagnostics")
    parser.add_argument("--verify-keys", action="store_true", help="Run health pings against all registered API keys")
    parser.add_argument("--reset-quotas", action="store_true", help="Reset all key quota counters to 0")
    parser.add_argument("--niche", type=str, help="Override channel niche for this run")
    parser.add_argument("--frequency", type=float, help="Override upload frequency in hours")
    parser.add_argument("--dry-run", action="store_true", help="Run in headless simulation mode without external API mutations")

    args = parser.parse_args()
    cfg = get_config()

    if args.niche:
        cfg.niche = args.niche
    if args.frequency:
        cfg.upload_frequency_hours = args.frequency
        cfg.scheduling.frequency_hours = args.frequency
    if args.dry_run:
        cfg.dry_run = True

    if args.reset_quotas:
        rotator = get_key_rotator()
        for states in rotator.pools.values():
            for s in states:
                s.used_units = 0
                s.blacklisted_until = 0.0
                s.failure_count = 0
                s.last_error = ""
                s.is_valid = True
        rotator.save_state()
        if console:
            console.print("[bold green]All daily key quotas and blacklists have been successfully reset![/bold green]")
        else:
            print("All daily key quotas and blacklists have been successfully reset!")
        return

    if args.verify_keys:
        print_banner()
        rotator = get_key_rotator()
        results = rotator.verify_all_keys()
        display_key_table(results)
        return

    if args.status:
        show_status()
        return

    if args.wizard:
        run_setup_wizard(cfg)
        if Confirm.ask("\nWould you like to start the autonomous cycle now?") if console else input("Start cycle now? (y/n): ").lower().startswith("y"):
            orch = MasterOrchestrator(cfg)
            orch.run_cycle()
        return

    if args.cycle:
        print_banner()
        orch = MasterOrchestrator(cfg)
        res = orch.run_cycle()
        if res.get("status") == "COMPLETED":
            sys.exit(0)
        else:
            sys.exit(1)

    if args.daemon:
        print_banner()
        orch = MasterOrchestrator(cfg)
        orch.start_scheduler(cfg.upload_frequency_hours)
        return

    # Default action if no arguments provided:
    # If config file doesn't exist, launch wizard, otherwise show status and prompt
    override_file = cfg.paths.data_dir / "clipmind_config.json"
    if not override_file.exists():
        run_setup_wizard(cfg)
        orch = MasterOrchestrator(cfg)
        orch.run_cycle()
    else:
        show_status()
        if console:
            console.print("\n[cyan]Tip: Use --cycle to run a single cycle, or --daemon to start the 12-hour scheduler loop.[/cyan]\n")
        else:
            print("\nTip: Use --cycle to run a single cycle, or --daemon to start the 12-hour scheduler loop.\n")


if __name__ == "__main__":
    main()

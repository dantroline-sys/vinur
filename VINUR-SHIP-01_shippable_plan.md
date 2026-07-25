# VINUR-SHIP-01 — Shippable Vinur: OS lifecycle + a consumer GUI

**Status:** S1 SHIPPED (machine seam + OS service layer) · **Doc version:** 1.0 · **Date:** 2026-07-25
**Goal:** every aspect of managing a vinur box works BOTH headlessly (CLI + the web panel,
already the primary surface) and through a consumer-grade native GUI following Apple/macOS
design principles — installable, start-at-login, understandable by someone who has never
opened a terminal.

## Principles (normative for every stage)

1. **Headless-first parity.** Every capability lands as CLI + machine-readable seam (and,
   where it belongs, a panel surface) BEFORE the GUI touches it.  The GUI calls the same
   seams; nothing is ever GUI-only.
2. **Thin shell.** The GUI is a native window around the EXISTING web panel plus only what a
   web page cannot do for itself: starting/stopping the host process, tray/menu-bar presence,
   OS-service registration, first-run setup.  (Precedent: Vinkona's Tauri launcher and its
   `status --json` seam.)
3. **Single implementation.** No management logic is duplicated in the shell.  The panel's
   seven tabs stay the one implementation of knowledge/serving/network management.
4. **One design language.** Apple HIG–derived, applied to BOTH the shell chrome and (S4) the
   panel itself, so they read as one application: clarity over density, an 8-pt spacing grid,
   a restrained type scale (system font stack), sidebar navigation, progressive disclosure
   (advanced knobs behind "details", never deleted), system light/dark followed
   automatically, motion used only to explain state changes.

## Stages

### S1 — the machine seam + OS service layer — SHIPPED
- `./vinur.sh status --json`: `supervisor.status_data()` — one collection feeding the text
  renderer, scripts, and the future shell (running/stale, per-service state incl. standby/
  held/failed with notes, swap state, panel URL, repo path).
- `supervisor run`: foreground mode — the OS service manager is the daemonizer (journald/
  launchd own the log stream); same preflight as `start`.
- `./vinur.sh service install|uninstall|status [--dry-run]` (`knowledgehost/service.py`):
  systemd USER unit on Linux, launchd agent on macOS, `--dry-run` prints the file and exact
  commands before anything acts; Windows refuses naming the interim (Task Scheduler over
  `supervisor run`) and its stage (S6).  The OS now revives the supervisor; the supervisor
  keeps reviving its children.
- Tests: `tests/ship_test.py`; gates entry "ship battery".

### S2 — first-run, headless: `setup` + `doctor`
- `./vinur.sh setup`: interactive first-run (uv env; config.toml from the example with
  prompted paths/bind/token; offer an embed-model pull through the egress broker; offer
  `service install`).  Mirrors Vinkona's on-ramp so the two feel like one family.
- `./vinur.sh doctor`: named, remedied checks — python/uv, config parses, ports free, GPU
  driver actually loaded (nvidia-smi AND the /dev/nvidia* + CDI story we debugged live),
  llama-server present when [serving] needs it, egress posture, service-layer state.
- Acceptance: a fresh clone on a clean box reaches a serving panel with only
  `./vinur.sh setup && ./vinur.sh start`.

### S3 — the GUI shell (`launcher/`, Tauri 2, compiled on Dan's box)
- Native window embedding the panel URL (token injected); the shell adds ONLY: Start/Stop/
  Restart of the box, a tray / menu-bar icon driven by a `status --json` poll (ok / attention
  / down), the S2 wizard as native panes, the S1 service toggle, and "Open logs".
- Mirrors `vinkona/launcher/`'s structure so both shells stay maintainable together.
- Acceptance: quit the shell → the box keeps serving (the shell is a REMOTE CONTROL, not the
  process owner — the OS service layer owns lifetime).

### S4 — panel visual pass to the same design language
- CSS-level restyle of the existing panel to the §Principles.4 language: type scale, spacing
  grid, control styling, light/dark.  Tab structure and every behavior unchanged — this is
  paint, deliberately after the shell exists so the two are tuned together.

### S5 — packaging & distribution
- Tauri bundler artifacts (.AppImage / .dmg; .msi waits for S6), install.sh hardening,
  versioned releases; update story stays `git pull + uv sync` documented in-app, tauri
  self-update deferred until releases exist to point it at.

### S6 — Windows pass
- Service layer (Task Scheduler or NSSM over `supervisor run`), path/console semantics in
  the supervisor, bundler `.msi`.  Rides the existing platform-independence thread.

## Non-goals
- A second management UI (the panel remains the management surface; the shell is chrome).
- Multi-box fleet management, remote administration beyond the existing LAN+token story.
- Cloud/telemetry anything.

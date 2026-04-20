# next-plaid-browser agent guide

<!-- effect-solutions:start -->
## Effect Best Practices

**IMPORTANT:** Always consult effect-solutions before writing Effect code.

1. Run `effect-solutions list` to see available guides
2. Run `effect-solutions show <topic>...` for relevant patterns (supports multiple topics)
3. Search `~/.local/share/effect-solutions/effect` for real implementations

Topics: quick-start, project-setup, tsconfig, basics, services-and-layers, data-modeling, error-handling, config, testing, cli.

Never guess at Effect patterns - check the guide first.

## Local Effect Source

The Effect v4 repository (`Effect-TS/effect-smol`, matching the installed
`effect@4.0.0-beta.x`) is cloned to `~/.local/share/effect-solutions/effect` for
reference. Use this to explore APIs, find usage examples, and understand
implementation details when the documentation isn't enough.

This project tracks Effect v4 beta — consult the `effect-smol` source for any
API not covered by `effect-solutions show`, since v3 docs will not match.

## Workers

Use `BrowserWorker.layer(spawn)` from `@effect/platform-browser` for all Web
Worker wiring. Do not call `new Worker(...)` directly. Worker correlation,
decoding, and error classification live on top in
`browser-src/effect/worker-transport.ts`. Source to reference:
`~/.local/share/effect-solutions/effect/packages/platform-browser/src/BrowserWorker.ts`.

## Typecheck

- `bun run typecheck` — authoritative. Runs patched `tsc`, emits Effect
  language-service diagnostics (missing services, unhandled errors, pipeline
  issues). Run this before commit and in CI.
- `bun run typecheck:fast` — `tsgo` (native-preview). ~10× faster but does
  NOT load the Effect language-service plugin. Use for inner-loop feedback
  only. A green `typecheck:fast` does not mean Effect diagnostics are clean.

## Testing (Effect code)

Use `@effect/vitest`. Entry points:

- `bun run test` — one-shot (`vitest run`).
- `bun run test:watch` — watch mode.
- `test:keyword` remains the Rust/Wasm kernel lane (`cargo test`).
- `smoke:chromium` remains the real-browser integration lane.

Prefer `it.effect` (auto-provided `TestContext` with `TestClock`, logs
suppressed) over `it.live`. Use `TestClock.adjust(duration)` from
`effect/testing` for timeout and readiness-gate tests. Use `it.layer(layer)`
to share a constructed layer across a `describe` block.
<!-- effect-solutions:end -->

## Package manager

This package uses **bun** (see `bun.lock`). Install with `bun install`, run
scripts with `bun run <script>`. Do not reintroduce `package-lock.json`.

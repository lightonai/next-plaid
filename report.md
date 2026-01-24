# Detecting and Redirecting Claude Code Search Tool Usage with Hooks

This report explains how to detect when Claude Code uses built-in search tools (Grep, Glob) and send custom messages to encourage using `colgrep` instead.

## Current Implementation in Colgrep

The colgrep plugin uses two hook types:

1. **SessionStart hook** - Injects comprehensive colgrep usage instructions at session start
2. **PreToolUse hook** - Adds a gentle reminder when Grep/Glob is used

```json
{
  "description": "Colgrep hooks - inject context and redirect search tools to colgrep",
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume",
        "hooks": [
          {
            "type": "command",
            "command": "colgrep --session-hook",
            "timeout": 10
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Grep|Glob",
        "hooks": [
          {
            "type": "command",
            "command": "echo '{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"additionalContext\":\"Reminder: colgrep is available for semantic code search. Consider using colgrep for better results.\"}}'",
            "timeout": 5
          }
        ]
      }
    ]
  }
}
```

### How It Works

1. **SessionStart** runs `colgrep --session-hook` which outputs detailed usage instructions
2. **PreToolUse** intercepts Grep/Glob calls and injects a reminder via `additionalContext`
3. The tools still execute (non-blocking), but Claude sees the reminder

This approach allows Claude to explore with built-in tools when needed while being nudged toward colgrep.

## Hook Events and Matchers

| Hook Event | When it Fires | Can Block? |
|------------|---------------|------------|
| `PreToolUse` | Before tool execution | Yes (exit code 2 or JSON) |
| `PostToolUse` | After tool completes | No (already ran) |
| `SessionStart` | Session begins/resumes | No |

### Matcher Patterns

The `matcher` field uses regex to match tool names:

| Pattern | Matches |
|---------|---------|
| `Grep` | Only the Grep tool |
| `Glob` | Only the Glob tool |
| `Grep\|Glob` | Both Grep and Glob |
| `Task` | Task tool (spawns subagents) |
| `*` | All tools |

## Alternative Approaches

### Option 1: Hard Block with Exit Code 2

If you want to completely prevent Grep/Glob usage:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Grep|Glob",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'STOP: Use colgrep instead. Run: colgrep \"<query>\" [path]' >&2 && exit 2"
          }
        ]
      }
    ]
  }
}
```

- Exit code 2 = blocking error
- stderr content is shown to Claude as feedback
- The tool call is prevented from executing

### Option 2: JSON Output with Permission Deny

For more control, return structured JSON:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Grep|Glob",
        "hooks": [
          {
            "type": "command",
            "command": "echo '{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"permissionDecision\":\"deny\",\"permissionDecisionReason\":\"Use colgrep instead\"}}'"
          }
        ]
      }
    ]
  }
}
```

### Option 3: Soft Reminder with additionalContext (Current Implementation)

Allows the tool to run but adds context for Claude:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Grep|Glob",
        "hooks": [
          {
            "type": "command",
            "command": "echo '{\"hookSpecificOutput\":{\"hookEventName\":\"PreToolUse\",\"additionalContext\":\"Reminder: colgrep is available for semantic code search.\"}}'"
          }
        ]
      }
    ]
  }
}
```

## JSON Response Fields

| Field | Value | Effect |
|-------|-------|--------|
| `permissionDecision` | `"deny"` | Blocks the tool call |
| `permissionDecision` | `"allow"` | Auto-approves (bypasses permission) |
| `permissionDecision` | `"ask"` | Shows permission dialog to user |
| `permissionDecisionReason` | string | Message shown to Claude (on deny) or user (on allow/ask) |
| `additionalContext` | string | Context added for Claude without blocking |

## Hook Input Schema Reference

### PreToolUse Input for Grep

```json
{
  "session_id": "abc123",
  "hook_event_name": "PreToolUse",
  "tool_name": "Grep",
  "tool_input": {
    "pattern": "async fn.*->.*Result",
    "path": "./src",
    "glob": "*.rs",
    "output_mode": "content"
  },
  "tool_use_id": "toolu_01ABC123..."
}
```

### PreToolUse Input for Glob

```json
{
  "session_id": "abc123",
  "hook_event_name": "PreToolUse",
  "tool_name": "Glob",
  "tool_input": {
    "pattern": "**/*.rs",
    "path": "./src"
  },
  "tool_use_id": "toolu_01ABC123..."
}
```

## Hook Behavior Matrix

| Exit Code | stderr | stdout | Effect |
|-----------|--------|--------|--------|
| 0 | ignored | shown to user (verbose) | Tool proceeds |
| 0 | ignored | JSON with `additionalContext` | Tool proceeds, context added for Claude |
| 0 | ignored | JSON with `permissionDecision: "deny"` | Tool blocked, reason shown to Claude |
| 2 | shown to Claude | ignored | Tool blocked, stderr is the message |
| other | shown to user | ignored | Non-blocking error, tool proceeds |

## Testing Hooks

1. Run `/hooks` in Claude Code to verify configuration
2. Use `claude --debug` to see hook execution logs
3. Test with simple commands: "search for error handling in this codebase"
4. Verify Claude receives the reminder when using Grep/Glob:
   ```
   PreToolUse:Grep hook additional context: Reminder: colgrep is available...
   ```

## References

- [Claude Code Hooks Guide](https://code.claude.com/docs/en/hooks-guide)
- [Claude Code Hooks Reference](https://code.claude.com/docs/en/hooks)
- Colgrep source: `colgrep/src/install/claude_code.rs`

# mgrep Installation Mechanism Analysis

This report analyzes how mgrep implements automatic installation for various AI coding tools (Claude Code, OpenCode, Codex, Factory Droid) to help you implement similar functionality for `next-plaid-cli`.

## Overview

mgrep uses four different installation strategies depending on the target tool's extension mechanism:

| Tool | Strategy | Config Location |
|------|----------|-----------------|
| Claude Code | Plugin Marketplace CLI | Uses `claude plugin` CLI commands |
| OpenCode | Tool file + JSON config | `~/.config/opencode/` |
| Codex | MCP CLI + AGENTS.md | `~/.codex/AGENTS.md` |
| Factory Droid | Hooks + Skills + Settings | `~/.factory/` |

---

## 1. Claude Code Installation

**File:** `src/install/claude-code.ts`

### Mechanism

Claude Code uses a **plugin marketplace system**. The installation leverages the `claude` CLI commands:

```typescript
// Add plugin to marketplace
await execAsync("claude plugin marketplace add mixedbread-ai/mgrep", { shell, env: process.env });

// Install the plugin
await execAsync("claude plugin install mgrep", { shell, env: process.env });
```

### Uninstall

```typescript
await execAsync("claude plugin uninstall mgrep", { shell, env: process.env });
await execAsync("claude plugin marketplace remove mixedbread-ai/mgrep", { shell, env: process.env });
```

### Required Assets

mgrep ships with a `.claude-plugin/` directory containing:

**`marketplace.json`** - Plugin registry metadata:
```json
{
  "$schema": "https://anthropic.com/claude-code/marketplace.schema.json",
  "name": "Mixedbread-Grep",
  "owner": { "name": "Mixedbread", "email": "support@mixedbread.ai" },
  "plugins": [{
    "name": "mgrep",
    "source": "./plugins/mgrep",
    "description": "Search your local files using Mixedbread",
    "version": "0.0.0",
    "author": { "name": "Joel Dierkes" },
    "skills": ["./skills/mgrep"]
  }]
}
```

**`plugins/mgrep/.claude-plugin/plugin.json`** - Plugin definition:
```json
{
  "name": "mgrep",
  "description": "Search your local files using Mixedbread",
  "version": "0.0.0",
  "author": { "name": "Joel Dierkes" },
  "hooks": "./hooks/hook.json"
}
```

**`plugins/mgrep/hooks/hook.json`** - Session lifecycle hooks:
```json
{
  "hooks": {
    "SessionStart": [{
      "matcher": "startup|resume",
      "hooks": [{
        "type": "command",
        "command": "python3 ${CLAUDE_PLUGIN_ROOT}/hooks/mgrep_watch.py",
        "timeout": 10
      }]
    }],
    "SessionEnd": [{
      "hooks": [{
        "type": "command",
        "command": "python3 ${CLAUDE_PLUGIN_ROOT}/hooks/mgrep_watch_kill.py",
        "timeout": 10
      }]
    }]
  }
}
```

### Key Requirements for Claude Code

- Requires Claude Code version 2.0.36+
- Uses `${CLAUDE_PLUGIN_ROOT}` environment variable for paths
- Plugin assets must be in `dist/plugins/` after build
- Build script copies plugins: `"postbuild": "cp -r plugins dist/"`

---

## 2. OpenCode Installation

**File:** `src/install/opencode.ts`

### Mechanism

OpenCode uses a **dual approach**:
1. Write a tool definition file to `~/.config/opencode/tool/mgrep.ts`
2. Add MCP server config to `~/.config/opencode/opencode.json` (or `.jsonc`)

### Tool File

Creates `~/.config/opencode/tool/mgrep.ts` with embedded SKILL documentation:

```typescript
import { tool } from "@opencode-ai/plugin"

const SKILL = `
---
name: mgrep
description: A semantic grep-like search tool...
---
// Usage instructions...
`;

export default tool({
  description: SKILL,
  args: {
    q: tool.schema.string().describe("The semantic search query."),
    m: tool.schema.number().default(10).describe("Number of chunks to return."),
    a: tool.schema.boolean().default(false).describe("Generate answer."),
  },
  async execute(args) {
    const result = await Bun.$`mgrep search -m ${args.m} ${args.a ? '-a ' : ''}${args.q}`.text()
    return result.trim()
  },
})
```

### Config File Modification

Adds MCP entry to `opencode.json`:

```typescript
const configJson = parseConfigFile(configPath, configContent);
configJson.mcp = configJson.mcp || {};
configJson.mcp.mgrep = {
  type: "local",
  command: ["mgrep", "mcp"],
  enabled: true,
};
fs.writeFileSync(configPath, stringify(configJson, null, 2));
```

### Uses `comment-json` for Safe Parsing

Preserves comments in JSONC files:
```typescript
import { parse, stringify } from "comment-json";
```

---

## 3. Codex Installation

**File:** `src/install/codex.ts`

### Mechanism

Codex uses a **CLI command + AGENTS.md file**:
1. Register MCP server via `codex mcp add` command
2. Append skill documentation to `~/.codex/AGENTS.md`

### MCP Registration

```typescript
await execAsync("codex mcp add mgrep mgrep mcp", { shell, env: process.env });
```

### AGENTS.md Skill Injection

Appends a SKILL markdown block to `~/.codex/AGENTS.md`:

```typescript
const SKILL = `
---
name: mgrep
description: A semantic grep-like search tool...
---

## When to use this skill
Whenever you need to search your local files...

## How to use this skill
Use \`mgrep\` to search your local files...
`;

const destPath = path.join(os.homedir(), ".codex", "AGENTS.md");
// Check if skill already exists
if (!existingContent.includes(SKILL) && !existingContent.includes(skillTrimmed)) {
  fs.appendFileSync(destPath, SKILL);
}
```

### Uninstall

Removes skill content with repeated string replacement:
```typescript
while (updatedContent !== previousContent) {
  previousContent = updatedContent;
  updatedContent = updatedContent.replace(SKILL, "");
  updatedContent = updatedContent.replace(SKILL.trim(), "");
}
```

---

## 4. Factory Droid Installation

**File:** `src/install/droid.ts`

### Mechanism

Factory Droid is the most complex, using **hooks, skills, and settings**:
1. Copy Python hook scripts to `~/.factory/hooks/mgrep/`
2. Copy SKILL.md to `~/.factory/skills/mgrep/`
3. Modify `~/.factory/settings.json` with hook configuration

### Bundled Assets

mgrep ships plugin assets at `dist/plugins/mgrep/`:
```
plugins/mgrep/
├── .claude-plugin/plugin.json
├── hooks/
│   ├── hook.json
│   ├── mgrep_watch.py
│   └── mgrep_watch_kill.py
└── skills/mgrep/SKILL.md
```

### Hook Installation

Copies Python scripts and registers hooks in settings:

```typescript
const hookConfig: HooksConfig = {
  SessionStart: [{
    matcher: "startup|resume",
    hooks: [{
      type: "command",
      command: `python3 "${watchPy}"`,
      timeout: 10,
    }],
  }],
  SessionEnd: [{
    hooks: [{
      type: "command",
      command: `python3 "${killPy}"`,
      timeout: 10,
    }],
  }],
};
```

### Settings Modification

Updates `~/.factory/settings.json`:

```typescript
const settings = loadSettings(settingsPath);
settings.enableHooks = true;
settings.allowBackgroundProcesses = true;
settings.hooks = mergeHooks(settings.hooks, hookConfig);
saveSettings(settingsPath, settings);
```

### Smart Merge for Hooks

Prevents duplicates when merging:
```typescript
function mergeHooks(existingHooks, newHooks) {
  // Clone existing, check for duplicates by matcher + command
  const duplicate = current.some(item =>
    (item?.matcher ?? null) === matcher &&
    item?.hooks?.[0]?.command === command
  );
  if (!duplicate) current.push(entry);
}
```

---

## Implementation Recommendations for next-plaid-cli

### 1. Project Structure

```
src/
├── index.ts                    # CLI entry point with commander
├── install/
│   ├── claude-code.ts          # Claude Code installer
│   ├── opencode.ts             # OpenCode installer
│   ├── codex.ts                # Codex installer
│   └── droid.ts                # Factory Droid installer
└── lib/
    └── warning.ts              # Post-install warning message

plugins/plaid/
├── .claude-plugin/
│   └── plugin.json
├── hooks/
│   ├── hook.json
│   ├── plaid_watch.py          # Session start hook
│   └── plaid_watch_kill.py     # Session end hook
└── skills/plaid/
    └── SKILL.md                # Usage instructions for agents
```

### 2. Build Configuration

In `package.json`:
```json
{
  "bin": { "plaid": "dist/index.js" },
  "scripts": {
    "postbuild": "chmod +x dist/index.js && cp -r plugins dist/"
  },
  "files": ["dist", "README.md"]
}
```

### 3. Command Registration

Using `commander`:
```typescript
import { installClaudeCode, uninstallClaudeCode } from "./install/claude-code.js";
// ... other imports

program.addCommand(installClaudeCode);
program.addCommand(uninstallClaudeCode);
// ... add other install commands
```

### 4. Key Dependencies

```json
{
  "commander": "^14.0.0",      // CLI framework
  "comment-json": "^4.5.0",    // Preserve JSON comments
  "chalk": "^5.6.0"            // Colored output
}
```

### 5. Essential Patterns

1. **Always check authentication first:**
   ```typescript
   await ensureAuthenticated();
   ```

2. **Use shell detection for cross-platform:**
   ```typescript
   const shell = process.env.SHELL ||
     (process.platform === "win32" ? process.env.COMSPEC || "cmd.exe" : "/bin/sh");
   ```

3. **Print post-install warning:**
   ```typescript
   printInstallWarning("Claude Code", "plaid uninstall-claude-code");
   ```

4. **Idempotent writes:**
   ```typescript
   function writeFileIfChanged(filePath, content) {
     const existing = fs.existsSync(filePath) ? fs.readFileSync(filePath, "utf-8") : undefined;
     if (existing !== content) fs.writeFileSync(filePath, content);
   }
   ```

5. **Graceful error handling with helpful messages:**
   ```typescript
   catch (error) {
     console.error(`Error: ${error}`);
     console.error(`Do you have claude-code version 2.0.36 or higher installed?`);
   }
   ```

---

## SKILL.md Template

Create a compelling SKILL.md that tells agents when and how to use your tool:

```markdown
---
name: plaid
description: Semantic code search tool. ALWAYS use this instead of Grep for code searches.
license: Apache 2.0
---

## When to use this skill

Whenever you need to search code files. Use this instead of Grep.

## How to use this skill

Use `plaid` for semantic code search with natural language queries.

### Do

```bash
plaid "authentication flow" -k 10
plaid "error handling patterns" ./src
```

### Don't

```bash
plaid "function"  # Too vague
```

## Keywords
search, grep, code, semantic search
```

---

## Summary

| Aspect | Claude Code | OpenCode | Codex | Factory Droid |
|--------|-------------|----------|-------|---------------|
| Installation Method | CLI commands | File writes | CLI + file | File writes |
| Config Format | Plugin system | JSONC | MCP + MD | JSON + scripts |
| Hooks | JSON in plugin | N/A | N/A | Python scripts |
| Skills | SKILL.md | Embedded TS | AGENTS.md | SKILL.md |
| Requires Build | Yes (cp plugins) | No | No | Yes (cp plugins) |

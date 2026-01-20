# Installing Colgrep for Claude Code

Colgrep integrates with Claude Code as a plugin, enabling semantic code search directly within your AI coding sessions.

## Prerequisites

- **Claude Code** version 2.0.36 or higher
- **Colgrep CLI** installed and available in your PATH

```bash
# Install colgrep (if not already installed)
cargo install colgrep
```

## Quick Install

Once colgrep is published to a marketplace, installation is a single command:

```bash
colgrep install-claude-code
```

This will:
1. Add the colgrep marketplace to Claude Code
2. Install the colgrep plugin
3. Enable semantic search in all future sessions

## Manual Installation (Local Development)

For local testing or development, you can install the plugin from a local directory:

### Step 1: Validate the Plugin

```bash
claude plugin validate /path/to/colgrep
```

This checks that the plugin manifest is correctly formatted.

### Step 2: Add Local Marketplace

```bash
claude plugin marketplace add /path/to/colgrep
```

This registers your local directory as a plugin source.

### Step 3: Install the Plugin

```bash
claude plugin install colgrep
```

### Step 4: Verify Installation

```bash
claude plugin list
```

You should see:
```
Installed plugins:

  ❯ colgrep@lightonai-colgrep
    Version: 0.1.0
    Scope: user
    Status: ✔ enabled
```

### Step 5: Restart Claude Code

Start a new Claude Code session to load the plugin. The SKILL.md instructions will be available to Claude.

## Uninstalling

### Using the CLI

```bash
colgrep uninstall-claude-code
```

### Manual Uninstall

```bash
claude plugin uninstall colgrep
claude plugin marketplace remove lightonai-colgrep
```

## What Gets Installed

The plugin installs the following components:

```
~/.claude/plugins/cache/lightonai-colgrep/colgrep/0.1.0/
├── .claude-plugin/
│   └── plugin.json      # Plugin metadata
├── hooks/
│   └── hook.json        # Session lifecycle hooks (empty for colgrep)
└── skills/
    └── colgrep/
        └── SKILL.md     # Instructions for Claude on how to use colgrep
```

## How It Works

Once installed, Claude Code will:

1. **Load the SKILL.md** at session start, learning when and how to use colgrep
2. **Prefer colgrep over grep** for semantic/intent-based code searches
3. **Use grep for exact matches** when you need literal string searches

### Example Session

After installation, Claude will automatically use colgrep for appropriate searches:

```
You: Find the authentication handling code

Claude: [Uses colgrep "authentication handling" instead of grep]
```

## Troubleshooting

### Plugin not found after installation

Restart Claude Code to load the newly installed plugin.

### Case-sensitivity errors on macOS

If you see errors like `EINVAL: invalid argument, rename`, ensure the marketplace name doesn't conflict with the plugin name. The marketplace is named `lightonai-colgrep` (not `Colgrep`) to avoid this issue.

### Claude Code version too old

The plugin system requires Claude Code 2.0.36+. Update with:

```bash
npm update -g @anthropic-ai/claude-code
```

### Verify colgrep is in PATH

The plugin calls `colgrep` directly. Ensure it's accessible:

```bash
which colgrep
colgrep --version
```

## Configuration

Colgrep uses its own configuration for search behavior. See the main README for:

- Setting the default model (`colgrep set-model`)
- Index location (`~/.local/share/colgrep/`)
- Config location (`~/.config/colgrep/`)

## Updating the Plugin

To update to a new version:

```bash
# For published marketplace
claude plugin update colgrep

# For local development, re-add the marketplace
claude plugin marketplace remove lightonai-colgrep
claude plugin marketplace add /path/to/colgrep
claude plugin uninstall colgrep
claude plugin install colgrep
```

# Installing Plaid for Claude Code

Plaid integrates with Claude Code as a plugin, enabling semantic code search directly within your AI coding sessions.

## Prerequisites

- **Claude Code** version 2.0.36 or higher
- **Plaid CLI** installed and available in your PATH

```bash
# Install plaid (if not already installed)
cargo install next-plaid-cli
```

## Quick Install

Once plaid is published to a marketplace, installation is a single command:

```bash
plaid install-claude-code
```

This will:
1. Add the plaid marketplace to Claude Code
2. Install the plaid plugin
3. Enable semantic search in all future sessions

## Manual Installation (Local Development)

For local testing or development, you can install the plugin from a local directory:

### Step 1: Validate the Plugin

```bash
claude plugin validate /path/to/next-plaid-cli
```

This checks that the plugin manifest is correctly formatted.

### Step 2: Add Local Marketplace

```bash
claude plugin marketplace add /path/to/next-plaid-cli
```

This registers your local directory as a plugin source.

### Step 3: Install the Plugin

```bash
claude plugin install plaid
```

### Step 4: Verify Installation

```bash
claude plugin list
```

You should see:
```
Installed plugins:

  ❯ plaid@lightonai-plaid
    Version: 0.1.0
    Scope: user
    Status: ✔ enabled
```

### Step 5: Restart Claude Code

Start a new Claude Code session to load the plugin. The SKILL.md instructions will be available to Claude.

## Uninstalling

### Using the CLI

```bash
plaid uninstall-claude-code
```

### Manual Uninstall

```bash
claude plugin uninstall plaid
claude plugin marketplace remove lightonai-plaid
```

## What Gets Installed

The plugin installs the following components:

```
~/.claude/plugins/cache/lightonai-plaid/plaid/0.1.0/
├── .claude-plugin/
│   └── plugin.json      # Plugin metadata
├── hooks/
│   └── hook.json        # Session lifecycle hooks (empty for plaid)
└── skills/
    └── plaid/
        └── SKILL.md     # Instructions for Claude on how to use plaid
```

## How It Works

Once installed, Claude Code will:

1. **Load the SKILL.md** at session start, learning when and how to use plaid
2. **Prefer plaid over grep** for semantic/intent-based code searches
3. **Use grep for exact matches** when you need literal string searches

### Example Session

After installation, Claude will automatically use plaid for appropriate searches:

```
You: Find the authentication handling code

Claude: [Uses plaid "authentication handling" instead of grep]
```

## Troubleshooting

### Plugin not found after installation

Restart Claude Code to load the newly installed plugin.

### Case-sensitivity errors on macOS

If you see errors like `EINVAL: invalid argument, rename`, ensure the marketplace name doesn't conflict with the plugin name. The marketplace is named `lightonai-plaid` (not `Plaid`) to avoid this issue.

### Claude Code version too old

The plugin system requires Claude Code 2.0.36+. Update with:

```bash
npm update -g @anthropic-ai/claude-code
```

### Verify plaid is in PATH

The plugin calls `plaid` directly. Ensure it's accessible:

```bash
which plaid
plaid --version
```

## Configuration

Plaid uses its own configuration for search behavior. See the main README for:

- Setting the default model (`plaid set-model`)
- Index location (`~/.local/share/plaid/`)
- Config location (`~/.config/plaid/`)

## Updating the Plugin

To update to a new version:

```bash
# For published marketplace
claude plugin update plaid

# For local development, re-add the marketplace
claude plugin marketplace remove lightonai-plaid
claude plugin marketplace add /path/to/next-plaid-cli
claude plugin uninstall plaid
claude plugin install plaid
```

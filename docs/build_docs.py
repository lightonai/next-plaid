#!/usr/bin/env python3
"""Build simple HTML documentation from README.md files.

Generates minimal HTML pages suitable for LLM copy-paste.
"""

import argparse
import re
import shutil
from pathlib import Path


# Mapping of relative paths to display names
SECTION_NAMES = {
    ".": "Overview",
    "next-plaid": "next-plaid (Core Library)",
    "next-plaid-api": "next-plaid-api (REST Server)",
    "next-plaid-api/python-sdk": "Python SDK",
    "next-plaid-onnx": "next-plaid-onnx (ONNX Runtime)",
    "next-plaid-onnx/python": "pylate-onnx-export",
    "colgrep": "ColGREP (Code Search)",
}

# Directories to skip when scanning
SKIP_DIRS = {
    ".venv", "venv", ".pytest_cache", "__pycache__", "node_modules",
    "target", "models", "docs", "dist", "build", ".git",
}


def find_readmes(root: Path, include_paths: list[str] | None = None) -> list[tuple[str, Path]]:
    """Find README.md files and return (display_name, path) tuples.

    If include_paths is provided, only include READMEs from those paths.
    Otherwise, scan all directories excluding SKIP_DIRS.
    """
    readmes = []

    if include_paths:
        # Use explicit include list
        for rel_path in include_paths:
            readme = root / rel_path / "README.md"
            if readme.exists():
                display_name = SECTION_NAMES.get(rel_path, rel_path)
                readmes.append((display_name, readme))
    else:
        # Scan all directories
        for readme in sorted(root.rglob("README.md")):
            parts = readme.relative_to(root).parts

            # Skip if any part is in skip list or starts with .
            if any(p.startswith(".") or p in SKIP_DIRS for p in parts):
                continue

            # Get relative path of parent directory
            rel_parent = str(readme.parent.relative_to(root)) if readme.parent != root else "."

            # Use custom name if defined, otherwise use path
            display_name = SECTION_NAMES.get(rel_parent, rel_parent)
            readmes.append((display_name, readme))

    return readmes


def markdown_to_html(md_content: str) -> str:
    """Convert markdown to HTML with minimal processing."""
    html = md_content

    # Extract and preserve code blocks
    code_blocks = []
    def save_code_block(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

    # Fenced code blocks
    html = re.sub(r'```(\w*)\n(.*?)```', save_code_block, html, flags=re.DOTALL)

    # Inline code
    inline_codes = []
    def save_inline_code(match):
        inline_codes.append(match.group(1))
        return f"__INLINE_CODE_{len(inline_codes) - 1}__"

    html = re.sub(r'`([^`]+)`', save_inline_code, html)

    # Extract and preserve raw HTML blocks/tags
    html_blocks = []
    def save_html_block(match):
        html_blocks.append(match.group(0))
        return f"__HTML_BLOCK_{len(html_blocks) - 1}__"

    # Preserve HTML tags (div, span, h1, etc.)
    html = re.sub(r'<(\w+)[^>]*>.*?</\1>', save_html_block, html, flags=re.DOTALL)
    html = re.sub(r'<(\w+)[^>]*/>', save_html_block, html)  # Self-closing tags

    # Now escape remaining special chars (not in preserved blocks)
    html = html.replace("&", "&amp;")
    html = html.replace("<", "&lt;")
    html = html.replace(">", "&gt;")

    # Headers
    html = re.sub(r'^######\s+(.+)$', r'<h6>\1</h6>', html, flags=re.MULTILINE)
    html = re.sub(r'^#####\s+(.+)$', r'<h5>\1</h5>', html, flags=re.MULTILINE)
    html = re.sub(r'^####\s+(.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'^###\s+(.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^##\s+(.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^#\s+(.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

    # Bold and italic
    html = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', html)
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

    # Links
    html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)

    # Horizontal rules
    html = re.sub(r'^---+$', r'<hr>', html, flags=re.MULTILINE)

    # Tables (simple conversion)
    def convert_table(match):
        lines = match.group(0).strip().split('\n')
        if len(lines) < 2:
            return match.group(0)

        result = ['<table>']
        for i, line in enumerate(lines):
            if re.match(r'^\|[-:\s|]+\|$', line):  # Skip separator row
                continue
            cells = [c.strip() for c in line.strip('|').split('|')]
            tag = 'th' if i == 0 else 'td'
            result.append('<tr>' + ''.join(f'<{tag}>{c}</{tag}>' for c in cells) + '</tr>')
        result.append('</table>')
        return '\n'.join(result)

    html = re.sub(r'(\|.+\|[\n\r]+)+', convert_table, html)

    # Lists (simple handling)
    html = re.sub(r'^(\s*)-\s+(.+)$', r'\1<li>\2</li>', html, flags=re.MULTILINE)
    html = re.sub(r'^(\s*)\d+\.\s+(.+)$', r'\1<li>\2</li>', html, flags=re.MULTILINE)

    # Wrap consecutive <li> in <ul>
    html = re.sub(r'((?:<li>.*</li>\n?)+)', r'<ul>\1</ul>', html)

    # Paragraphs (lines not already wrapped)
    lines = html.split('\n')
    result = []
    in_para = False
    para_lines = []

    for line in lines:
        stripped = line.strip()
        is_block = (
            stripped.startswith('<h') or
            stripped.startswith('<ul') or
            stripped.startswith('<table') or
            stripped.startswith('<hr') or
            stripped.startswith('__CODE_BLOCK_') or
            stripped.startswith('__HTML_BLOCK_') or
            stripped == '' or
            stripped.startswith('<li>') or
            stripped.startswith('</ul>') or
            stripped.startswith('</table>')
        )

        if is_block:
            if para_lines:
                result.append('<p>' + ' '.join(para_lines) + '</p>')
                para_lines = []
            result.append(line)
        else:
            para_lines.append(stripped)

    if para_lines:
        result.append('<p>' + ' '.join(para_lines) + '</p>')

    html = '\n'.join(result)

    # Restore code blocks
    for i, block in enumerate(code_blocks):
        lang_match = re.match(r'```(\w*)\n', block)
        lang = lang_match.group(1) if lang_match else ''
        code = block[len(lang_match.group(0)) if lang_match else 3:-3]
        if lang == 'mermaid':
            # Mermaid blocks: use <pre class="mermaid"> without <code> wrapper or escaping
            html = html.replace(f"__CODE_BLOCK_{i}__", f'<pre class="mermaid">{code}</pre>')
        else:
            code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html = html.replace(f"__CODE_BLOCK_{i}__", f'<pre><code class="{lang}">{code}</code></pre>')

    # Restore inline code
    for i, code in enumerate(inline_codes):
        code = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = html.replace(f"__INLINE_CODE_{i}__", f'<code>{code}</code>')

    # Restore HTML blocks
    for i, block in enumerate(html_blocks):
        html = html.replace(f"__HTML_BLOCK_{i}__", block)

    # Clean up empty paragraphs and divs
    html = re.sub(r'<p>\s*</p>', '', html)
    html = re.sub(r'<p>\s*<div', '<div', html)
    html = re.sub(r'</div>\s*</p>', '</div>', html)

    return html


def generate_nav(readmes: list[tuple[str, Path]], current: Path, root: Path, output_dir: Path) -> str:
    """Generate navigation HTML."""
    nav_items = []
    for name, path in readmes:
        html_name = "index.html" if path.parent == root else str(path.parent.relative_to(root)).replace("/", "-") + ".html"
        current_html = "index.html" if current.parent == root else str(current.parent.relative_to(root)).replace("/", "-") + ".html"

        if html_name == current_html:
            nav_items.append(f'<strong>{name}</strong>')
        else:
            nav_items.append(f'<a href="{html_name}">{name}</a>')

    return ' | '.join(nav_items)


def build_page(title: str, nav: str, content: str) -> str:
    """Build a complete HTML page."""
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
body {{ font-family: monospace; max-width: 900px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
pre {{ background: #f4f4f4; padding: 10px; overflow-x: auto; }}
code {{ background: #f4f4f4; padding: 2px 4px; }}
pre code {{ background: none; padding: 0; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f4f4f4; }}
nav {{ background: #f4f4f4; padding: 10px; margin-bottom: 20px; }}
a {{ color: #0066cc; }}
</style>
</head>
<body>
<nav>{nav}</nav>
<main>
{content}
</main>
<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
mermaid.initialize({{ startOnLoad: true }});
</script>
</body>
</html>
'''


def main():
    parser = argparse.ArgumentParser(description="Build HTML docs from README.md files")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root")
    parser.add_argument("--output", type=Path, default=Path("docs/_site"), help="Output directory")
    parser.add_argument("--include", nargs="+", help="Specific paths to include (relative to root)")
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    readmes = find_readmes(root, args.include)

    if not readmes:
        print("No README.md files found")
        return 1

    print(f"Found {len(readmes)} README.md file(s):")
    for name, path in readmes:
        print(f"  - {name}: {path.relative_to(root)}")

    for name, readme_path in readmes:
        content = readme_path.read_text(encoding="utf-8")

        # Rewrite relative image/asset paths so they resolve from the flat output dir.
        # e.g. in colgrep/README.md, src="../docs/x.gif" -> src="docs/x.gif"
        if readme_path.parent != root:
            readme_dir = readme_path.parent
            def rewrite_src(match):
                prefix, path_val = match.group(1), match.group(2)
                if path_val.startswith(("http://", "https://", "/")):
                    return match.group(0)
                resolved = (readme_dir / path_val).resolve()
                try:
                    return f'{prefix}"{resolved.relative_to(root)}"'
                except ValueError:
                    return match.group(0)
            content = re.sub(r'(src=)"([^"]+)"', rewrite_src, content)

        html_content = markdown_to_html(content)
        nav = generate_nav(readmes, readme_path, root, output_dir)

        # Determine output filename
        if readme_path.parent == root:
            html_name = "index.html"
        else:
            html_name = str(readme_path.parent.relative_to(root)).replace("/", "-") + ".html"

        page = build_page(name, nav, html_content)
        output_path = output_dir / html_name
        output_path.write_text(page, encoding="utf-8")
        print(f"Generated: {output_path.relative_to(root)}")

    # Copy static assets (images) referenced in READMEs
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg", "*.gif"):
        for img in root.glob(f"docs/{ext}"):
            dest = output_dir / "docs" / img.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest)
            print(f"Copied: {img.relative_to(root)} -> docs/{img.name}")

    print(f"\nDocs built in: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())

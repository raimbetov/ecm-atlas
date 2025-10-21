#!/usr/bin/env python3
"""
Generate professional PDF from PCOLCE evidence document
Renders mermaid diagrams as SVG and includes them in the PDF
"""

import re
import subprocess
import tempfile
import os
from pathlib import Path

def extract_mermaid_diagrams(markdown_content):
    """Extract mermaid diagrams and replace with placeholders"""
    diagrams = []

    def replace_mermaid(match):
        diagram_code = match.group(1)
        diagrams.append(diagram_code)
        idx = len(diagrams) - 1
        # Return placeholder that will be replaced with image
        return f'<div class="mermaid-diagram" data-diagram-id="{idx}"></div>'

    # Extract mermaid code blocks
    pattern = r'```mermaid\n(.*?)\n```'
    content = re.sub(pattern, replace_mermaid, markdown_content, flags=re.DOTALL)

    return content, diagrams

def render_mermaid_diagram(diagram_code, output_path):
    """Render a single mermaid diagram to SVG using mermaid.cli if available"""
    # Try using mermaid-cli (mmdc)
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
            f.write(diagram_code)
            mmd_file = f.name

        # Try mmdc command
        result = subprocess.run(
            ['mmdc', '-i', mmd_file, '-o', output_path, '-b', 'transparent'],
            capture_output=True,
            text=True,
            timeout=10
        )

        os.unlink(mmd_file)

        if result.returncode == 0:
            return True

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: Create placeholder SVG
    placeholder_svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="800" height="400" viewBox="0 0 800 400">
  <rect width="800" height="400" fill="#f5f5f5" stroke="#ccc" stroke-width="2"/>
  <text x="400" y="180" font-family="Arial, sans-serif" font-size="16" fill="#666" text-anchor="middle">
    Mermaid Diagram
  </text>
  <text x="400" y="210" font-family="monospace" font-size="12" fill="#999" text-anchor="middle">
    (Install mermaid-cli for automatic rendering)
  </text>
  <foreignObject x="50" y="230" width="700" height="150">
    <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: monospace; font-size: 10px; color: #666; white-space: pre-wrap; overflow: hidden;">
{diagram_code[:500]}
    </div>
  </foreignObject>
</svg>'''

    with open(output_path, 'w') as f:
        f.write(placeholder_svg)

    return False

def markdown_to_html(markdown_file, output_html):
    """Convert markdown to HTML with professional styling"""

    # Read markdown
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract mermaid diagrams
    content, diagrams = extract_mermaid_diagrams(content)

    # Render mermaid diagrams to SVG
    diagram_files = []
    temp_dir = tempfile.mkdtemp()

    for idx, diagram_code in enumerate(diagrams):
        svg_path = os.path.join(temp_dir, f'diagram_{idx}.svg')
        rendered = render_mermaid_diagram(diagram_code, svg_path)
        diagram_files.append((idx, svg_path, rendered))

        # Replace placeholder with image tag
        content = content.replace(
            f'<div class="mermaid-diagram" data-diagram-id="{idx}"></div>',
            f'<div class="diagram-container"><img src="{svg_path}" alt="Diagram {idx+1}" class="mermaid-svg"/></div>'
        )

    # Convert markdown to HTML using markdown library
    import markdown
    from markdown.extensions.tables import TableExtension
    from markdown.extensions.fenced_code import FencedCodeExtension

    md = markdown.Markdown(extensions=[
        TableExtension(),
        FencedCodeExtension(),
        'markdown.extensions.nl2br',
        'markdown.extensions.sane_lists'
    ])

    html_content = md.convert(content)

    # Create professional HTML template
    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PCOLCE Context-Dependent Participation in Aging and Pathology: Evidence Document</title>
    <style>
        @page {{
            size: A4;
            margin: 2.5cm 2cm 2.5cm 2cm;
            @top-center {{
                content: "PCOLCE Evidence Document";
                font-size: 9pt;
                color: #666;
            }}
            @bottom-center {{
                content: "Page " counter(page) " of " counter(pages);
                font-size: 9pt;
                color: #666;
            }}
        }}

        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
            max-width: 100%;
            margin: 0;
            padding: 0;
        }}

        h1 {{
            font-size: 20pt;
            font-weight: bold;
            color: #1a1a1a;
            margin-top: 0;
            margin-bottom: 0.5em;
            page-break-after: avoid;
            border-bottom: 3px solid #2c5aa0;
            padding-bottom: 0.3em;
        }}

        h2 {{
            font-size: 16pt;
            font-weight: bold;
            color: #2c5aa0;
            margin-top: 1.5em;
            margin-bottom: 0.8em;
            page-break-after: avoid;
            border-bottom: 1px solid #ccc;
            padding-bottom: 0.2em;
        }}

        h3 {{
            font-size: 13pt;
            font-weight: bold;
            color: #2c5aa0;
            margin-top: 1.2em;
            margin-bottom: 0.6em;
            page-break-after: avoid;
        }}

        h4 {{
            font-size: 11pt;
            font-weight: bold;
            color: #333;
            margin-top: 1em;
            margin-bottom: 0.5em;
            page-break-after: avoid;
        }}

        p {{
            margin: 0.8em 0;
            text-align: justify;
        }}

        strong {{
            font-weight: bold;
            color: #1a1a1a;
        }}

        em {{
            font-style: italic;
        }}

        code {{
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            background-color: #f5f5f5;
            padding: 0.1em 0.3em;
            border-radius: 2px;
            border: 1px solid #ddd;
        }}

        pre {{
            font-family: 'Courier New', monospace;
            font-size: 9pt;
            background-color: #f5f5f5;
            padding: 1em;
            border-left: 4px solid #2c5aa0;
            overflow-x: auto;
            page-break-inside: avoid;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
            font-size: 9pt;
            page-break-inside: avoid;
        }}

        th {{
            background-color: #2c5aa0;
            color: white;
            font-weight: bold;
            padding: 0.6em 0.4em;
            text-align: left;
            border: 1px solid #1a4080;
        }}

        td {{
            padding: 0.5em 0.4em;
            border: 1px solid #ddd;
            vertical-align: top;
        }}

        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}

        .diagram-container {{
            text-align: center;
            margin: 1.5em 0;
            page-break-inside: avoid;
        }}

        .mermaid-svg {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            background-color: white;
            padding: 1em;
        }}

        blockquote {{
            border-left: 4px solid #2c5aa0;
            margin: 1em 0;
            padding: 0.5em 1em;
            background-color: #f5f7fa;
            font-style: italic;
        }}

        ul, ol {{
            margin: 0.8em 0;
            padding-left: 2em;
        }}

        li {{
            margin: 0.4em 0;
        }}

        hr {{
            border: none;
            border-top: 2px solid #ddd;
            margin: 2em 0;
        }}

        .correction-notice {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 1em;
            margin: 1em 0;
            page-break-inside: avoid;
        }}

        .page-break {{
            page-break-before: always;
        }}

        @media print {{
            body {{
                font-size: 10pt;
            }}

            h1 {{ font-size: 18pt; }}
            h2 {{ font-size: 14pt; }}
            h3 {{ font-size: 12pt; }}

            a {{
                color: #2c5aa0;
                text-decoration: none;
            }}
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>'''

    # Write HTML file
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_template)

    return output_html, diagram_files

def html_to_pdf(html_file, output_pdf):
    """Convert HTML to PDF using weasyprint"""
    try:
        from weasyprint import HTML, CSS

        # Convert with WeasyPrint
        HTML(filename=html_file).write_pdf(
            output_pdf,
            stylesheets=[CSS(string='@page { size: A4; margin: 2.5cm 2cm; }')]
        )

        return True

    except Exception as e:
        print(f"WeasyPrint error: {e}")
        return False

def main():
    """Main function to generate PDF"""

    base_dir = Path(__file__).parent
    markdown_file = base_dir / "01_EVIDENCE_DOCUMENT_PCOLCE_CONTEXT_DEPENDENCY.md"
    output_pdf = base_dir / "PCOLCE_Evidence_Document_v1.1.pdf"
    temp_html = base_dir / "temp_document.html"

    print("="*80)
    print("PCOLCE Evidence Document - PDF Generation")
    print("="*80)

    # Step 1: Convert markdown to HTML
    print("\n[1/2] Converting Markdown to HTML...")
    html_file, diagrams = markdown_to_html(markdown_file, temp_html)
    print(f"  ✓ HTML generated: {temp_html}")
    print(f"  ✓ Mermaid diagrams processed: {len(diagrams)}")

    # Step 2: Convert HTML to PDF
    print("\n[2/2] Converting HTML to PDF...")
    success = html_to_pdf(html_file, output_pdf)

    if success:
        print(f"  ✓ PDF generated: {output_pdf}")

        # Get file size
        size_mb = output_pdf.stat().st_size / (1024 * 1024)
        print(f"  ✓ File size: {size_mb:.2f} MB")

        # Clean up temp HTML
        if temp_html.exists():
            temp_html.unlink()
            print(f"  ✓ Cleaned up temporary files")

        print("\n" + "="*80)
        print("✅ SUCCESS! PDF ready for distribution")
        print("="*80)
        print(f"\nOutput file: {output_pdf}")

        return 0
    else:
        print("  ✗ PDF generation failed")
        print(f"\n  HTML file available at: {temp_html}")
        print("  You can manually convert it to PDF or install weasyprint")
        return 1

if __name__ == "__main__":
    exit(main())

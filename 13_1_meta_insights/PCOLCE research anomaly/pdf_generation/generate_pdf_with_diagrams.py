#!/usr/bin/env python3
"""
Generate professional PDF from PCOLCE evidence document
Uses pre-rendered PNG diagrams from diagrams/ folder
"""

import re
import os
from pathlib import Path
import base64

def extract_and_replace_mermaid_diagrams(markdown_content, diagrams_dir):
    """Extract mermaid diagrams and replace with rendered PNG images"""

    diagram_counter = [0]  # Use list for mutable counter in closure

    def replace_mermaid(match):
        diagram_counter[0] += 1
        idx = diagram_counter[0]

        # Path to rendered diagram
        diagram_path = diagrams_dir / f"diagram_{idx}.png"

        if diagram_path.exists():
            # Embed image as base64 data URI for self-contained HTML
            with open(diagram_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')

            return f'<div class="diagram-container"><img src="data:image/png;base64,{img_data}" alt="Diagram {idx}" class="diagram-image"/></div>'
        else:
            # Fallback to placeholder
            return f'<div class="diagram-placeholder">Diagram {idx} (not rendered)</div>'

    # Replace mermaid code blocks with images
    pattern = r'```mermaid\n(.*?)\n```'
    content = re.sub(pattern, replace_mermaid, markdown_content, flags=re.DOTALL)

    return content, diagram_counter[0]

def markdown_to_html(markdown_file, diagrams_dir, output_html):
    """Convert markdown to HTML with embedded diagram images"""

    # Read markdown
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace mermaid diagrams with rendered PNGs
    content, diagram_count = extract_and_replace_mermaid_diagrams(content, diagrams_dir)

    # Convert markdown to HTML
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
    <title>PCOLCE Context-Dependent Participation in Aging and Pathology: Evidence Document v1.1</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm 1.5cm 2cm 1.5cm;
            @top-center {{
                content: "PCOLCE Evidence Document v1.1";
                font-size: 9pt;
                color: #666;
                font-family: Arial, sans-serif;
            }}
            @bottom-center {{
                content: "Page " counter(page);
                font-size: 9pt;
                color: #666;
                font-family: Arial, sans-serif;
            }}
        }}

        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            font-size: 10pt;
            line-height: 1.5;
            color: #222;
            max-width: 100%;
            margin: 0;
            padding: 0;
        }}

        h1 {{
            font-size: 18pt;
            font-weight: bold;
            color: #1a1a1a;
            margin-top: 0;
            margin-bottom: 0.5em;
            page-break-after: avoid;
            border-bottom: 3px solid #2c5aa0;
            padding-bottom: 0.3em;
        }}

        h2 {{
            font-size: 14pt;
            font-weight: bold;
            color: #2c5aa0;
            margin-top: 1.2em;
            margin-bottom: 0.6em;
            page-break-after: avoid;
            border-bottom: 1px solid #ddd;
            padding-bottom: 0.2em;
        }}

        h3 {{
            font-size: 12pt;
            font-weight: bold;
            color: #2c5aa0;
            margin-top: 1em;
            margin-bottom: 0.5em;
            page-break-after: avoid;
        }}

        h4 {{
            font-size: 10pt;
            font-weight: bold;
            color: #333;
            margin-top: 0.8em;
            margin-bottom: 0.4em;
            page-break-after: avoid;
        }}

        p {{
            margin: 0.6em 0;
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
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 8.5pt;
            background-color: #f5f5f5;
            padding: 0.1em 0.3em;
            border-radius: 2px;
            border: 1px solid #ddd;
        }}

        pre {{
            font-family: 'Courier New', 'Consolas', monospace;
            font-size: 8.5pt;
            background-color: #f5f5f5;
            padding: 0.8em;
            border-left: 3px solid #2c5aa0;
            overflow-x: auto;
            page-break-inside: avoid;
            margin: 0.8em 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
            font-size: 8.5pt;
            page-break-inside: auto;
        }}

        thead {{
            display: table-header-group;
        }}

        tr {{
            page-break-inside: avoid;
        }}

        th {{
            background-color: #2c5aa0;
            color: white;
            font-weight: bold;
            padding: 0.5em 0.4em;
            text-align: left;
            border: 1px solid #1a4080;
        }}

        td {{
            padding: 0.4em 0.3em;
            border: 1px solid #ccc;
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

        .diagram-image {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            background-color: white;
            padding: 0.5em;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .diagram-placeholder {{
            background-color: #f5f5f5;
            border: 2px dashed #ccc;
            padding: 2em;
            text-align: center;
            color: #999;
            font-style: italic;
        }}

        blockquote {{
            border-left: 4px solid #2c5aa0;
            margin: 1em 0;
            padding: 0.5em 1em;
            background-color: #f5f7fa;
            font-style: italic;
        }}

        ul, ol {{
            margin: 0.6em 0;
            padding-left: 1.5em;
        }}

        li {{
            margin: 0.3em 0;
        }}

        hr {{
            border: none;
            border-top: 2px solid #ddd;
            margin: 1.5em 0;
        }}

        a {{
            color: #2c5aa0;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        .correction-notice {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 1em;
            margin: 1em 0;
            page-break-inside: avoid;
            font-size: 9pt;
        }}

        @media print {{
            body {{ font-size: 9pt; }}
            h1 {{ font-size: 16pt; }}
            h2 {{ font-size: 13pt; }}
            h3 {{ font-size: 11pt; }}
            table {{ font-size: 8pt; }}
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

    return output_html, diagram_count

def html_to_pdf(html_file, output_pdf):
    """Convert HTML to PDF using weasyprint"""
    try:
        from weasyprint import HTML, CSS

        # Convert with WeasyPrint
        HTML(filename=html_file).write_pdf(
            output_pdf,
            stylesheets=[CSS(string='''
                @page {
                    size: A4;
                    margin: 2cm 1.5cm;
                }
            ''')]
        )

        return True

    except Exception as e:
        print(f"WeasyPrint error: {e}")
        return False

def main():
    """Main function to generate PDF with rendered diagrams"""

    base_dir = Path(__file__).parent
    markdown_file = base_dir / "01_EVIDENCE_DOCUMENT_PCOLCE_CONTEXT_DEPENDENCY.md"
    diagrams_dir = base_dir / "diagrams"
    output_pdf = base_dir / "PCOLCE_Evidence_Document_v1.1.pdf"
    temp_html = base_dir / "temp_document_with_diagrams.html"

    print("="*80)
    print("PCOLCE Evidence Document - PDF Generation (with Rendered Diagrams)")
    print("="*80)

    # Check for rendered diagrams
    if not diagrams_dir.exists():
        print("\n⚠️  WARNING: diagrams/ folder not found")
        print("   Run: python render_diagrams.py")
        print("   to generate diagram PNGs first\n")
        return 1

    diagram_files = list(diagrams_dir.glob("diagram_*.png"))
    print(f"\n✓ Found {len(diagram_files)} rendered diagrams")

    # Step 1: Convert markdown to HTML with embedded diagrams
    print("\n[1/2] Converting Markdown to HTML (embedding diagrams)...")
    html_file, diagram_count = markdown_to_html(markdown_file, diagrams_dir, temp_html)
    print(f"  ✓ HTML generated: {temp_html.name}")
    print(f"  ✓ Diagrams embedded: {diagram_count}")

    # Step 2: Convert HTML to PDF
    print("\n[2/2] Converting HTML to PDF...")
    success = html_to_pdf(html_file, output_pdf)

    if success:
        print(f"  ✓ PDF generated: {output_pdf.name}")

        # Get file size
        size_mb = output_pdf.stat().st_size / (1024 * 1024)
        print(f"  ✓ File size: {size_mb:.2f} MB")

        # Clean up temp HTML
        if temp_html.exists():
            temp_html.unlink()
            print(f"  ✓ Cleaned up temporary files")

        print("\n" + "="*80)
        print("✅ SUCCESS! PDF with rendered diagrams ready for distribution")
        print("="*80)
        print(f"\nOutput file: {output_pdf}")
        print(f"Diagrams included: {diagram_count}")

        return 0
    else:
        print("  ✗ PDF generation failed")
        print(f"\n  HTML file available at: {temp_html}")
        return 1

if __name__ == "__main__":
    exit(main())

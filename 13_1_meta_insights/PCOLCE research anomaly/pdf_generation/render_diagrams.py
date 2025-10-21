#!/usr/bin/env python3
"""
Extract and render mermaid diagrams from the evidence document
Creates PNG images for inclusion in PDF
"""

import re
import subprocess
import tempfile
import os
from pathlib import Path

def extract_mermaid_diagrams(markdown_file):
    """Extract all mermaid diagrams from markdown file"""

    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all mermaid code blocks
    pattern = r'```mermaid\n(.*?)\n```'
    matches = re.findall(pattern, content, flags=re.DOTALL)

    return matches

def render_mermaid_to_png(diagram_code, output_path, width=1200):
    """Render mermaid diagram to PNG using mermaid-cli"""

    # Create temporary mermaid file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
        f.write(diagram_code)
        mmd_file = f.name

    try:
        # Render with mmdc
        cmd = [
            'mmdc',
            '-i', mmd_file,
            '-o', output_path,
            '-b', 'white',
            '-w', str(width),
            '--pdfFit'
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print(f"  ✓ Rendered: {os.path.basename(output_path)}")
            return True
        else:
            print(f"  ✗ Failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout rendering diagram")
        return False
    except FileNotFoundError:
        print(f"  ✗ mmdc command not found")
        return False
    finally:
        # Clean up temp file
        if os.path.exists(mmd_file):
            os.unlink(mmd_file)

def main():
    """Extract and render all diagrams"""

    base_dir = Path(__file__).parent
    markdown_file = base_dir / "01_EVIDENCE_DOCUMENT_PCOLCE_CONTEXT_DEPENDENCY.md"
    output_dir = base_dir / "diagrams"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    print("="*80)
    print("Mermaid Diagram Renderer")
    print("="*80)

    # Extract diagrams
    print(f"\n[1/2] Extracting diagrams from: {markdown_file.name}")
    diagrams = extract_mermaid_diagrams(markdown_file)
    print(f"  ✓ Found {len(diagrams)} mermaid diagrams")

    # Render each diagram
    print(f"\n[2/2] Rendering diagrams to PNG...")

    success_count = 0
    for idx, diagram_code in enumerate(diagrams, 1):
        output_path = output_dir / f"diagram_{idx}.png"

        # Determine diagram type from code
        diagram_type = "unknown"
        if "graph TD" in diagram_code:
            diagram_type = "hierarchical"
        elif "graph LR" in diagram_code:
            diagram_type = "workflow"

        print(f"\n  Diagram {idx}/{len(diagrams)} ({diagram_type}):")

        if render_mermaid_to_png(diagram_code, str(output_path)):
            success_count += 1
            # Check file size
            size_kb = output_path.stat().st_size / 1024
            print(f"    Size: {size_kb:.1f} KB")

    print("\n" + "="*80)
    print(f"✅ Rendered {success_count}/{len(diagrams)} diagrams successfully")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files created:")

    for img_file in sorted(output_dir.glob("*.png")):
        print(f"  - {img_file.name}")

    return 0 if success_count == len(diagrams) else 1

if __name__ == "__main__":
    exit(main())

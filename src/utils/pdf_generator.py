import markdown
from xhtml2pdf import pisa
import os
import re
from pathlib import Path
from typing import Optional


def _soft_wrap_table_text(text: str) -> str:
    raw = str(text or "")
    if not raw.strip():
        return raw

    def _wrap_token(token: str) -> str:
        if len(token) < 14:
            return token
        wrapped = token
        for marker in ("/", "_", "-", "."):
            wrapped = wrapped.replace(marker, marker + "<br/>")
        return wrapped

    parts = re.split(r"(\s+)", raw)
    return "".join(_wrap_token(part) if not part.isspace() else part for part in parts)


def _prepare_exec_tables_for_pdf(html_content: str) -> str:
    if not html_content or "exec-table" not in html_content:
        return html_content

    cell_pattern = re.compile(r"(<t[dh][^>]*>)(.*?)(</t[dh]>)", flags=re.IGNORECASE | re.DOTALL)
    text_node_pattern = re.compile(r"(?<=>)([^<>]+)(?=<)", flags=re.DOTALL)

    def _replace_cell(match: re.Match) -> str:
        open_tag, inner_html, close_tag = match.groups()
        if "<" not in inner_html and ">" not in inner_html:
            return f"{open_tag}{_soft_wrap_table_text(inner_html)}{close_tag}"

        def _replace_text_node(text_match: re.Match) -> str:
            return _soft_wrap_table_text(text_match.group(1))

        wrapped_inner = text_node_pattern.sub(_replace_text_node, inner_html)
        return f"{open_tag}{wrapped_inner}{close_tag}"

    return cell_pattern.sub(_replace_cell, html_content)


def resolve_image_path(img_path: str, base_dir_abs: Optional[str]) -> Optional[str]:
    if not img_path:
        return None

    if os.path.isabs(img_path) and os.path.exists(img_path):
        return img_path

    candidates = []
    if base_dir_abs:
        candidates.append(os.path.join(base_dir_abs, img_path))
        basename = os.path.basename(img_path)
        candidates.append(os.path.join(base_dir_abs, "artifacts", "plots", basename))
        candidates.append(os.path.join(base_dir_abs, "work", "artifacts", "plots", basename))
        candidates.append(os.path.join(base_dir_abs, "sandbox", "downloaded_artifacts", "plots", basename))
    else:
        candidates.append(os.path.abspath(img_path))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None

def convert_report_to_pdf(
    markdown_content: str,
    output_filename: str = "final_report.pdf",
    base_dir: Optional[str] = None,
) -> bool:
    """
    Converts markdown content to a PDF file using xhtml2pdf.
    Resolves image paths and adds basic styling.
    """
    try:
        base_dir_abs = os.path.abspath(base_dir) if base_dir else None
        # 1. Keep images inline to preserve narrative context.
        inline_image_count = 0

        def _replace_markdown_image(match):
            nonlocal inline_image_count
            alt_text = (match.group(1) or "").strip() or "plot"
            img_path = (match.group(2) or "").strip()
            abs_path = resolve_image_path(img_path, base_dir_abs)
            if not abs_path:
                return f"*Image not found: {alt_text}*"
            inline_image_count += 1
            img_src = Path(abs_path).resolve().as_posix()
            safe_alt = alt_text.replace("<", "").replace(">", "")
            return (
                f'<div class="inline-figure">'
                f'<img src="{img_src}" style="max-width: 95%; height: auto;" alt="{safe_alt}" />'
                f'<p class="inline-caption">{safe_alt}</p>'
                f"</div>"
            )

        markdown_text_with_inline_images = re.sub(
            r'!\[(.*?)\]\((.*?)\)',
            _replace_markdown_image,
            markdown_content,
        )

        # FALLBACK: If no inline images were resolved, scan known plot folders
        images = []
        if inline_image_count == 0:
            if base_dir_abs:
                candidate_dirs = [
                    os.path.join(base_dir_abs, "static", "plots"),
                    os.path.join(base_dir_abs, "artifacts", "plots"),
                    os.path.join(base_dir_abs, "work", "artifacts", "plots"),
                ]
            else:
                candidate_dirs = [
                    os.path.abspath(os.path.join("static", "plots")),
                    os.path.abspath(os.path.join("artifacts", "plots")),
                    os.path.abspath(os.path.join("work", "artifacts", "plots")),
                ]
            supported_ext = ('.png', '.jpg', '.jpeg')
            for plots_dir in candidate_dirs:
                if not os.path.exists(plots_dir):
                    continue
                plot_files = sorted([f for f in os.listdir(plots_dir) if f.lower().endswith(supported_ext)])
                for f in plot_files:
                    full_path = os.path.join(plots_dir, f)
                    images.append((f, full_path))
        
        # 2. Convert Markdown to HTML (keeping inline figures in-place)
        html_content = markdown.markdown(markdown_text_with_inline_images, extensions=['tables', 'fenced_code'])
        html_content = _prepare_exec_tables_for_pdf(html_content)
        
        # 3. Construct Image Grid (2-Column Table)
        image_grid_html = ""
        if images:
            image_grid_html += "<h3>Visual Analysis</h3>"

        resolved_images = []
        for alt_text, img_path in images:
            abs_path = resolve_image_path(img_path, base_dir_abs)
            if abs_path:
                resolved_images.append((alt_text, abs_path))

        if resolved_images:
            image_grid_html += '<table style="width: 100%; border: none;">'
            
            for i, (alt_text, abs_path) in enumerate(resolved_images):
                # Start row for even indices (0, 2, 4...)
                if i % 2 == 0:
                    image_grid_html += '<tr>'

                # Create Cell
                img_src = Path(abs_path).resolve().as_posix()
                image_grid_html += f'''
                    <td style="width: 50%; padding: 5px; vertical-align: top; border: none;">
                        <div style="text-align: center;">
                            <img src="{img_src}" style="width: 500px; height: auto;" />
                            <p style="font-size: 8pt; color: #666;">{alt_text}</p>
                        </div>
                    </td>
                '''
                
                # End row for odd indices (1, 3, 5...) OR if it's the last image
                if i % 2 == 1 or i == len(resolved_images) - 1:
                    image_grid_html += '</tr>'
            
            image_grid_html += '</table>'
        elif images:
            image_grid_html += '<p style="color: #666; font-style: italic;">No plots were generated.</p>'

        # 4. Add CSS Styling and Assemble
        styled_html = f"""
        <html>
        <head>
            <style>
                @page {{
                    size: A4;
                    margin: 1.5cm;
                }}
                body {{
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 11pt;
                    line-height: 1.4;
                    color: #333;
                }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }}
                h2 {{ color: #e74c3c; margin-top: 20px; }}
                h3 {{ color: #34495e; margin-top: 15px; border-bottom: 1px solid #eee; }}
                code {{ background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-family: Courier New, monospace; }}
                pre {{ background-color: #f8f8f8; border: 1px solid #ddd; padding: 10px; overflow-x: auto; }}
                
                /* Data Tables */
                table.data {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                table.data th, table.data td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
                table.data th {{ background-color: #f2f2f2; font-weight: bold; }}

                /* Executive HTML tables injected by translator */
                table.exec-table {{ width: 100%; border-collapse: collapse; margin: 12px 0 16px 0; font-size: 8.5pt; table-layout: fixed; }}
                table.exec-table th, table.exec-table td {{ border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; vertical-align: top; word-wrap: break-word; white-space: normal; }}
                table.exec-table th {{ background-color: #e2e8f0; color: #0f172a; font-weight: bold; }}
                table.exec-table tr:nth-child(even) td {{ background-color: #f8fafc; }}
                .status-badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 8pt; font-weight: bold; }}
                .status-ok {{ background: #dcfce7; color: #166534; }}
                .status-warn {{ background: #fef3c7; color: #92400e; }}
                .status-error {{ background: #fee2e2; color: #991b1b; }}
                .status-neutral {{ background: #e2e8f0; color: #334155; }}
                .inline-figure {{ text-align: center; margin: 14px 0; }}
                .inline-caption {{ font-size: 8pt; color: #666; margin-top: 4px; }}

                .footer {{ position: fixed; bottom: 0; width: 100%; text-align: center; font-size: 9pt; color: #aaa; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>The Insight Foundry - Executive Report</h1>
            </div>
            
            {html_content}
            
            <div class="visualizations">{image_grid_html}</div>

            <div class="footer">
                Generated by Gemini Agents | Automated Business Intelligence
            </div>
        </body>
        </html>
        """
        
        output_path = output_filename
        if base_dir_abs and not os.path.isabs(output_filename):
            output_path = os.path.join(base_dir_abs, output_filename)
        # 4. Write to PDF
        with open(output_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(
                styled_html, dest=pdf_file
            )
            
        if pisa_status.err:
            print(f"PDF generation error: {pisa_status.err}")
            return False
            
        print(f"PDF generated successfully: {output_path}")
        return True

    except Exception as e:
        print(f"Failed to generate PDF: {e}")
        return False

import re
from typing import List

from marker.logger import get_logger
from marker.output import json_to_html
from marker.processors.llm import BaseLLMSimpleBlockProcessor, BlockData, PromptData
from marker.schema import BlockTypes
from marker.schema.document import Document
from pydantic import BaseModel

logger = get_logger()


class LLMFormProcessor(BaseLLMSimpleBlockProcessor):
    block_types = (BlockTypes.Form,)
    form_rewriting_prompt = """You are a text correction expert specializing in accurately reproducing text from images.
You will receive an image of a text block and an html representation of the form in the image.
Your task is to correct any errors in the html representation, and format it properly.

**Instructions:**
1. Carefully examine the provided form block image.
2. Analyze the html representation of the form. If the html is empty or contains only empty tags (like `<p></p>`), extract all text directly from the image.
3. Preserve the EXACT order, line breaks, blank lines, horizontal alignment/spacing, AND TABLE STRUCTURES as they appear in the image. If content is arranged in columns, preserve that columnar structure using HTML table tags.
4. If text appears on separate lines in the image, keep them on separate lines in HTML - use separate <p> tags or <br> tags for each distinct line.
5. Formatting markers (**, *, etc.) are for styling only - they do NOT indicate lines should be merged.
6. Use appropriate HTML tags: `table, p, span, i, b, u, th, td, tr, br, and div`.
7. You can use ☐ for checkbox.
8. For fill-in-the-blank fields with multiple symbols (_, ▁, …, or .), use the same symbols. The number of symbols MUST match the visual length/width of the blank field as it appears in the image. Count the approximate width of the blank line in the image and use the exact same number of symbols to represent it exactly.
9. Do not fill the blank fields - keep them as they are.
10. Use <table>, <tr>, and <td> tags to represent actual tables. If content is arranged in columns (e.g., left column with labels, right column with options), it MUST be represented as a table structure. Preserve the columnar layout - do not convert tables into paragraphs.
11. Use <ul> and <li> tags to represent lists.
12. Use <ol> and <li> tags to represent ordered lists.
13. Use <h1> through <h6> tags to represent headings. If text is obviously a section header (e.g., titles, section titles, numbered sections like "3-Souligne les bonnes réponses"), it MUST be represented with heading tags (<h1>, <h2>, <h3>, etc.) rather than just bold text or paragraphs. Individual numbered form fields or questions (e.g., "1. La prise de la Bastille : ▁▁▁▁") should use <p> tags, NOT headings. Strong clues that text is a section header: centered text alignment (`text-align: center;`), larger font size, bold formatting, or text that appears to be a major title or section divider.
14. Use <b> tag for bold text, <i> tag for italic text, and <u> tag for underlined text.
15. Use <br> tag to represent line breaks within a paragraph when multiple related lines should be in one paragraph.
16. Use separate <p> tags when each line should be its own paragraph.
17. Preserve the exact content from the image - make sure everything is included in the html representation.
18. Generate the corrected html representation based on the image content.

**Example - Preserving line breaks with formatting:**
If the image shows:
```
Line One
Line Two
```
The HTML MUST be:
```html
<p>Line One</p>
<p>Line Two</p>
```
NOT (WRONG - lines merged):
```html
<p>Line One Line Two</p>

**CRITICAL: Detect and Preserve Table Structures**
- Use <table> when content is clearly arranged in columns (vertical alignment)
- Look for visual alignment: if text appears in distinct vertical columns in the image, use <table>, <tr>, and <td> tags
- Do NOT convert table structures into paragraphs with line breaks - preserve the columnar structure

**Example - Two-column table structure:**
If the image shows a two-column layout:
```
Column 1:          Column 2:
                   Detail about item A.
Item A
                   Additional detail about item A.
```

The HTML MUST be:
```html
<table>
    <tr>
        <td></td><td>Detail about item A.</td>
        <td>Item A</td>
        <td></td><td>Additional detail about item A.</td>
    </tr>
</table>
```
**When to use tables vs paragraphs:**
- Use <table> when content is clearly arranged in columns (vertical alignment)

**Example - Section header vs Form field:**
Section header (use heading tag):
```
Section Title
```
→ `<h3>Section Title</h3>`

Form field/question (use paragraph tag):
```
Form field/question (use paragraph tag):
```
Sample field label: ____________________________
```
→ `<p>Sample field label: ____________________________</p>`

NOT (WRONG - form field should not be a heading):
→ `<h2>Sample field label: ____________________________</h2>`

**Input:**
```html
{block_html}
```
"""

    def inference_blocks(self, document: Document) -> List[BlockData]:
        blocks = super().inference_blocks(document)

        out_blocks = []
        for block_idx, block_data in enumerate(blocks):
            block = block_data["block"]
            page = block_data["page"]

            # DEBUG: Log form block info
            logger.debug(f"[FORM DEBUG] ===== FORM BLOCK {block_idx} =====")
            logger.debug(f"[FORM DEBUG] Form block id={block.id}, page={page.page_id}")
            structure_index = "N/A"
            if page.structure is not None and block.id in page.structure:
                structure_index = page.structure.index(block.id)

            # Format top_k scores if available
            top_k_str = "N/A"
            if block.top_k:
                top_k_formatted = ", ".join(
                    [
                        f"{bt.name}:{score:.3f}"
                        for bt, score in sorted(
                            block.top_k.items(), key=lambda x: x[1], reverse=True
                        )[:5]
                    ]
                )
                top_k_str = f"top_k=[{top_k_formatted}]"

            # Get content preview
            try:
                content_preview = json_to_html(block.render(document))[:150]
                content_text = re.sub(r"<[^>]+>", "", content_preview).strip()[:100]
            except Exception as e:
                content_text = f"<error rendering: {e}>"

            # Also extract plain text content for further analysis/debugging
            text_content = ""
            try:
                # Attempt to get text content from block render (strip HTML tags)
                rendered_html = json_to_html(block.render(document))
                text_content = re.sub(r"<[^>]+>", "", rendered_html).strip()
            except Exception as e:
                text_content = f"<error extracting text: {e}>"

            logger.debug(
                f"[FORM DEBUG] Form block {block.id}: bbox={block.polygon.bbox}, structure_index={structure_index}, {top_k_str}, text_content='{text_content}'"
            )

            # Process Form blocks even if they don't have TableCell children
            # (e.g., when TableProcessor doesn't process Form blocks)
            # BaseTable.assemble_html will handle blocks without TableCell children
            out_blocks.append(block_data)

        logger.debug(f"[FORM DEBUG] Total form blocks to process: {len(out_blocks)}")
        return out_blocks

    def block_prompts(self, document: Document) -> List[PromptData]:
        prompt_data = []
        for prompt_idx, block_data in enumerate(self.inference_blocks(document)):
            block = block_data["block"]
            block_html = json_to_html(block.render(document))

            # DEBUG: Log HTML content to check order
            logger.debug(
                f"[FORM DEBUG] Form block {block.id}: HTML length={len(block_html)}, HTML=\n{block_html}\n"
            )
            # Extract numbered items from HTML to check order
            numbered_items = re.findall(r"(\d+)\.\s*([^<]+)", block_html)
            if numbered_items:
                logger.debug(
                    f"[FORM DEBUG] Form block {block.id}: Found numbered items in HTML: {[(num, text[:30]) for num, text in numbered_items]}"
                )

            prompt = self.form_rewriting_prompt.replace("{block_html}", block_html)
            image = self.extract_image(document, block)
            prompt_data.append(
                {
                    "prompt": prompt,
                    "image": image,
                    "block": block,
                    "schema": FormSchema,
                    "page": block_data["page"],
                }
            )
        return prompt_data

    def rewrite_block(
        self, response: dict, prompt_data: PromptData, document: Document
    ):
        block = prompt_data["block"]
        block_html = json_to_html(block.render(document))

        # DEBUG: Log original HTML order
        original_items = re.findall(r"(\d+)\.\s*([^<]+)", block_html)
        if original_items:
            logger.debug(
                f"[FORM DEBUG] Form block {block.id}: Original HTML numbered items: {[(num, text[:30]) for num, text in original_items]}"
            )

        if not response or "corrected_html" not in response:
            block.update_metadata(llm_error_count=1)
            return

        corrected_html = response["corrected_html"]

        # The original table is okay
        if "no corrections needed" in corrected_html.lower():
            logger.debug(
                f"[FORM DEBUG] Form block {block.id}: LLM returned 'no corrections needed'"
            )
            return
        else:
            logger.debug(
                f"[FORM DEBUG] Form block {block.id}: corrected_html:\n{corrected_html}\n"
            )

        # Potentially a partial response
        if len(corrected_html) < len(block_html) * 0.33:
            block.update_metadata(llm_error_count=1)
            return

        corrected_html = corrected_html.strip().lstrip("```html").rstrip("```").strip()

        # DEBUG: Log corrected HTML order
        corrected_items = re.findall(r"(\d+)\.\s*([^<]+)", corrected_html)
        if corrected_items:
            logger.debug(
                f"[FORM DEBUG] Form block {block.id}: Corrected HTML numbered items: {[(num, text[:30]) for num, text in corrected_items]}"
            )
            if original_items and corrected_items:
                original_nums = [int(num) for num, _ in original_items]
                corrected_nums = [int(num) for num, _ in corrected_items]
                if original_nums != corrected_nums:
                    logger.warning(
                        f"[FORM DEBUG] Form block {block.id}: ORDER CHANGED! Original: {original_nums}, Corrected: {corrected_nums}"
                    )

        block.html = corrected_html


class FormSchema(BaseModel):
    comparison: str
    corrected_html: str

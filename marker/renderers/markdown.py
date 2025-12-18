import re
from collections import defaultdict
from typing import Annotated, Tuple

import regex
import six
from bs4 import NavigableString
from markdownify import MarkdownConverter, re_whitespace
from marker.logger import get_logger
from marker.renderers.html import HTMLRenderer
from marker.schema import BlockTypes
from marker.schema.document import Document
from pydantic import BaseModel

logger = get_logger()


def escape_dollars(text):
    return text.replace("$", r"\$")


def cleanup_text(full_text):
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    full_text = re.sub(r"(\n\s){3,}", "\n\n", full_text)
    return full_text.strip()


def get_formatted_table_text(element):
    text = []
    for content in element.contents:
        if content is None:
            continue

        if isinstance(content, NavigableString):
            stripped = content.strip()
            if stripped:
                text.append(escape_dollars(stripped))
        elif content.name == "br":
            text.append("<br>")
        elif content.name == "math":
            text.append("$" + content.text + "$")
        else:
            content_str = escape_dollars(str(content))
            text.append(content_str)

    full_text = ""
    for i, t in enumerate(text):
        if t == "<br>":
            full_text += t
        elif i > 0 and text[i - 1] != "<br>":
            full_text += " " + t
        else:
            full_text += t
    return full_text


class Markdownify(MarkdownConverter):
    def __init__(
        self,
        paginate_output,
        page_separator,
        inline_math_delimiters,
        block_math_delimiters,
        html_tables_in_markdown,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.paginate_output = paginate_output
        self.page_separator = page_separator
        self.inline_math_delimiters = inline_math_delimiters
        self.block_math_delimiters = block_math_delimiters
        self.html_tables_in_markdown = html_tables_in_markdown

    def convert_div(self, el, text, parent_tags):
        is_page = el.has_attr("class") and el["class"][0] == "page"
        if self.paginate_output and is_page:
            page_id = el["data-page-id"]
            pagination_item = (
                "\n\n" + "{" + str(page_id) + "}" + self.page_separator + "\n\n"
            )
            return pagination_item + text
        else:
            # Add line breaks for regular divs to preserve block-level separation
            return f"{text}\n\n" if text else ""

    def convert_p(self, el, text, parent_tags):
        hyphens = r"-—¬"
        has_continuation = el.has_attr("class") and "has-continuation" in el["class"]
        if has_continuation:
            block_type = BlockTypes[el["block-type"]]
            if block_type in [BlockTypes.TextInlineMath, BlockTypes.Text]:
                if regex.compile(
                    rf".*[\p{{Ll}}|\d][{hyphens}]\s?$", regex.DOTALL
                ).match(
                    text
                ):  # handle hypenation across pages
                    return regex.split(rf"[{hyphens}]\s?$", text)[0]
                return f"{text} "
            if block_type == BlockTypes.ListGroup:
                return f"{text}"
        return f"{text}\n\n" if text else ""  # default convert_p behavior

    def convert_math(self, el, text, parent_tags):
        block = el.has_attr("display") and el["display"] == "block"
        if block:
            return (
                "\n"
                + self.block_math_delimiters[0]
                + text.strip()
                + self.block_math_delimiters[1]
                + "\n"
            )
        else:
            return (
                " "
                + self.inline_math_delimiters[0]
                + text.strip()
                + self.inline_math_delimiters[1]
                + " "
            )

    def convert_table(self, el, text, parent_tags):
        if self.html_tables_in_markdown:
            return "\n\n" + str(el) + "\n\n"

        total_rows = len(el.find_all("tr"))
        colspans = []
        rowspan_cols = defaultdict(int)
        for i, row in enumerate(el.find_all("tr")):
            row_cols = rowspan_cols[i]
            for cell in row.find_all(["td", "th"]):
                colspan = int(cell.get("colspan", 1))
                row_cols += colspan
                for r in range(int(cell.get("rowspan", 1)) - 1):
                    rowspan_cols[
                        i + r
                    ] += colspan  # Add the colspan to the next rows, so they get the correct number of columns
            colspans.append(row_cols)
        total_cols = max(colspans) if colspans else 0

        grid = [[None for _ in range(total_cols)] for _ in range(total_rows)]

        for row_idx, tr in enumerate(el.find_all("tr")):
            col_idx = 0
            for cell in tr.find_all(["td", "th"]):
                # Skip filled positions
                while col_idx < total_cols and grid[row_idx][col_idx] is not None:
                    col_idx += 1

                # Fill in grid
                value = (
                    get_formatted_table_text(cell)
                    .replace("\n", " ")
                    .replace("|", " ")
                    .strip()
                )
                rowspan = int(cell.get("rowspan", 1))
                colspan = int(cell.get("colspan", 1))

                if col_idx >= total_cols:
                    # Skip this cell if we're out of bounds
                    continue

                for r in range(rowspan):
                    for c in range(colspan):
                        try:
                            if r == 0 and c == 0:
                                grid[row_idx][col_idx] = value
                            else:
                                grid[row_idx + r][
                                    col_idx + c
                                ] = ""  # Empty cell due to rowspan/colspan
                        except IndexError:
                            # Sometimes the colspan/rowspan predictions can overflow
                            logger.info(
                                f"Overflow in columns: {col_idx + c} >= {total_cols} or rows: {row_idx + r} >= {total_rows}"
                            )
                            continue

                col_idx += colspan

        markdown_lines = []
        col_widths = [0] * total_cols
        for row in grid:
            for col_idx, cell in enumerate(row):
                if cell is not None:
                    col_widths[col_idx] = max(col_widths[col_idx], len(str(cell)))

        def add_header_line():
            markdown_lines.append(
                "|" + "|".join("-" * (width + 2) for width in col_widths) + "|"
            )

        # Generate markdown rows
        added_header = False
        for i, row in enumerate(grid):
            is_empty_line = all(not cell for cell in row)
            if is_empty_line and not added_header:
                # Skip leading blank lines
                continue

            line = []
            for col_idx, cell in enumerate(row):
                if cell is None:
                    cell = ""
                padding = col_widths[col_idx] - len(str(cell))
                line.append(f" {cell}{' ' * padding} ")
            markdown_lines.append("|" + "|".join(line) + "|")

            if not added_header:
                # Skip empty lines when adding the header row
                add_header_line()
                added_header = True

        # Handle one row tables
        # if total_rows == 1:
        #     add_header_line()

        table_md = "\n".join(markdown_lines)
        return "\n\n" + table_md + "\n\n"

    def convert_ul(self, el, text, parent_tags):
        """
        Convert <ul> tags to markdown.
        If list items already contain numbers (e.g., "2. text") or letters (e.g., "a) text"),
        don't add "-" marker.
        Also handles patterns where numbers/letters appear at the end (e.g., "text 1." -> "1. text").
        """
        numbered_pattern = re.compile(r"^\s*(\d+)(\.|\))\s+(.+)")
        # Pattern to match letter-based numbering at start: "a) text", "b) text", etc.
        letter_numbered_pattern = re.compile(r"^\s*([a-z])(\.|\))\s+(.+)")
        # Pattern to match trailing numbers/letters: "text 1." or "text a) 1)" or "text 1)"
        trailing_number_pattern = re.compile(r"^(.+?)\s+(\d+)\.\s*$")
        trailing_number_paren_pattern = re.compile(r"^(.+?)\s+(\d+)\)\s*$")
        # Pattern for "text a) 1)" - extract the number
        trailing_letter_number_pattern = re.compile(r"^(.+?)\s+[a-z]\)\s+(\d+)\)\s*$")

        # Find only direct child <li> elements (not nested ones)
        li_items = [
            child
            for child in el.children
            if hasattr(child, "name") and child.name == "li"
        ]

        if not li_items:
            return super().convert_ul(el, text, parent_tags)

        lines = []
        for li in li_items:
            # Extract text content from the li element, excluding nested lists
            # Get text only from direct children that aren't ul/ol to avoid duplication
            text_parts = []
            for child in li.children:
                if hasattr(child, "name") and child.name in ["ul", "ol"]:
                    # Skip nested lists - they'll be processed separately by markdownify
                    continue
                elif isinstance(child, NavigableString):
                    text_parts.append(str(child))
                elif hasattr(child, "get_text"):
                    text_parts.append(child.get_text(separator="", strip=False))
            li_text = "".join(text_parts)
            li_text_stripped = li_text.strip()

            # Check for trailing patterns and move number/letter to beginning
            processed = False

            # Pattern: "text a) 1)" -> "1. text"
            match = trailing_letter_number_pattern.match(li_text_stripped)
            if match:
                text_part = match.group(1).strip()
                number = match.group(2)
                lines.append(f"{number}. {text_part}")
                processed = True

            # Pattern: "text 1." -> "1. text"
            if not processed:
                match = trailing_number_pattern.match(li_text_stripped)
                if match:
                    text_part = match.group(1).strip()
                    number = match.group(2)
                    lines.append(f"{number}. {text_part}")
                    processed = True

            # Pattern: "text 1)" -> "1. text"
            if not processed:
                match = trailing_number_paren_pattern.match(li_text_stripped)
                if match:
                    text_part = match.group(1).strip()
                    number = match.group(2)
                    lines.append(f"{number}. {text_part}")
                    processed = True

            if not processed:
                # Check if already numbered (numeric or letter-based)
                if numbered_pattern.match(li_text_stripped):
                    # Already numbered with numbers (e.g., "1. text"), output as-is without "-"
                    lines.append(li_text_stripped)
                elif letter_numbered_pattern.match(li_text_stripped):
                    # Already numbered with letters (e.g., "a) text"), output as-is without "-"
                    lines.append(li_text_stripped)
                else:
                    # Use default behavior with "-"
                    lines.append(f"- {li_text_stripped}")

            # Handle nested lists - process them recursively but don't include their text in parent
            for nested_list in li.find_all(["ul", "ol"], recursive=False):
                # Use markdownify's conversion for nested lists (will recursively call convert_ul)
                # Convert BeautifulSoup element to string first
                nested_markdown = self.convert(str(nested_list))
                if nested_markdown:
                    # Indent nested list items (add 2 spaces to each line)
                    indented_lines = []
                    for line in nested_markdown.strip().split("\n"):
                        if line.strip():
                            indented_lines.append("  " + line)
                        else:
                            indented_lines.append(line)
                    lines.append("\n".join(indented_lines))

        return "\n".join(lines) + "\n\n"

    def convert_a(self, el, text, parent_tags):
        text = self.escape(text)
        # Escape brackets and parentheses in text
        text = re.sub(r"([\[\]()])", r"\\\1", text)
        return super().convert_a(el, text, parent_tags)

    def convert_span(self, el, text, parent_tags):
        if el.get("id"):
            return f'<span id="{el["id"]}">{text}</span>'
        else:
            return text

    def escape(self, text, parent_tags=None):
        text = super().escape(text, parent_tags)
        if self.options["escape_dollars"]:
            text = text.replace("$", r"\$")
        return text

    def process_text(self, el, parent_tags=None):
        text = six.text_type(el) or ""

        # normalize whitespace if we're not inside a preformatted element
        if not el.find_parent("pre"):
            text = re_whitespace.sub(" ", text)

        # escape special characters if we're not inside a preformatted or code element
        if not el.find_parent(["pre", "code", "kbd", "samp", "math"]):
            text = self.escape(text)

        # remove trailing whitespaces if any of the following condition is true:
        # - current text node is the last node in li
        # - current text node is followed by an embedded list
        if el.parent.name == "li" and (
            not el.next_sibling or el.next_sibling.name in ["ul", "ol"]
        ):
            text = text.rstrip()

        return text


class MarkdownOutput(BaseModel):
    markdown: str
    images: dict
    metadata: dict


class MarkdownRenderer(HTMLRenderer):
    page_separator: Annotated[
        str, "The separator to use between pages.", "Default is '-' * 48."
    ] = ("-" * 48)
    inline_math_delimiters: Annotated[
        Tuple[str], "The delimiters to use for inline math."
    ] = ("$", "$")
    block_math_delimiters: Annotated[
        Tuple[str], "The delimiters to use for block math."
    ] = ("$$", "$$")
    html_tables_in_markdown: Annotated[
        bool, "Return tables formatted as HTML, instead of in markdown"
    ] = False

    @property
    def md_cls(self):
        return Markdownify(
            self.paginate_output,
            self.page_separator,
            heading_style="ATX",
            bullets="-",
            escape_misc=False,
            escape_underscores=False,
            escape_asterisks=True,
            escape_dollars=True,
            sub_symbol="<sub>",
            sup_symbol="<sup>",
            inline_math_delimiters=self.inline_math_delimiters,
            block_math_delimiters=self.block_math_delimiters,
            html_tables_in_markdown=self.html_tables_in_markdown,
        )

    def __call__(self, document: Document) -> MarkdownOutput:
        document_output = document.render(self.block_config)
        full_html, images = self.extract_html(document, document_output)

        logger.debug(f"[MarkdownRenderer] full_html: {repr(full_html)}")
        markdown = self.md_cls.convert(full_html)
        logger.debug(f"[MarkdownRenderer] markdown after convert: {repr(markdown)}")
        markdown = cleanup_text(markdown)
        logger.debug(f"[MarkdownRenderer] markdown after cleanup: {repr(markdown)}")

        # Normalize form fill-in-the-blank markers
        markdown = re.sub(
            r"(_|▁|…|\.){4,}", lambda m: "." * len(m.group()) * 2, markdown
        )

        # Normalize sequences of dots to fixed line width (fill-in-the-blank fields)
        FILL_IN_BLANK_WIDTH = 150  # Standard width for fill-in-the-blank fields
        markdown = re.sub(
            r"(\. ?){130,}",
            lambda m: "\n\n" + "." * FILL_IN_BLANK_WIDTH + "\n\n",
            markdown,
        )
        # Ensure we set the correct blanks for pagination markers
        if self.paginate_output:
            if not markdown.startswith("\n\n"):
                markdown = "\n\n" + markdown
            if markdown.endswith(self.page_separator):
                markdown += "\n\n"

        return MarkdownOutput(
            markdown=markdown,
            images=images,
            metadata=self.generate_document_metadata(document, document_output),
        )

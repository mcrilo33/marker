import json
from typing import List, Tuple

from marker.logger import get_logger
from marker.processors.llm import BaseLLMComplexBlockProcessor
from marker.schema import BlockTypes
from marker.schema.blocks import Block
from marker.schema.document import Document
from marker.schema.groups import PageGroup
from pydantic import BaseModel
from tqdm import tqdm

logger = get_logger()


class LLMSectionHeaderProcessor(BaseLLMComplexBlockProcessor):
    page_prompt = """You're a text correction expert specializing in accurately analyzing complex PDF documents. You will be given a list of all of the section headers from a document, along with their page number and approximate dimensions.  The headers will be formatted like below, and will be presented in order.

```json
[
    {
        "bbox": [x1, y1, x2, y2],
        "width": x2 - x1,
        "height": y2 - y1,
        "page": 0,
        "id": "/page/0/SectionHeader/1",
        "html": "<h1>Introduction</h1>",
    }, ...
]
```

Bboxes have been normalized to 0-1000.

Your goal is to make sure that the section headers have the correct levels (h1, h2, h3, h4, h5, or h6).  If a section header does not have the right level, edit the html to fix it.

Guidelines:
- Edit the blocks to ensure that the section headers have the correct levels.
- **Pattern Recognition**: Headers with similar formatting patterns should be at the same level. Look for:
  - **Formatting consistency**: Headers with the same HTML formatting (e.g., all using `<b>` tags, all using `**` bold markdown, all using `<i>` italics) should typically be grouped together.
  - **Symbol patterns**: Headers with similar symbols (→, *, •, -, etc.) should be at the same level.
  - **Content patterns**: Headers with similar content structure (dates, numbering, prefixes) should have consistent levels.
  - **Visual consistency**: Headers that appear visually similar in the document should be grouped hierarchically.
- **Semantic grouping**: Headers that serve the same structural purpose (e.g., all date-based timeline entries, all numbered subsections) should have consistent levels, **regardless of symbol differences**. This is the most important criterion - prioritize semantic purpose over symbol matching.
- **Document structure**: Headers that are clearly subsections of a main topic should be one level deeper than their parent.
- Only edit the h1, h2, h3, h4, h5, and h6 tags.  Do not change any other tags or content in the headers.
- Only output the headers that changed (if nothing changed, output nothing).
- Every header you output needs to have one and only one level tag (h1, h2, h3, h4, h5, or h6).

**Instructions:**
1. Carefully examine the provided section headers and JSON.
2. **Identify patterns**: Look for headers with similar formatting (HTML tags, markdown, symbols), content structure (dates, numbering), and visual appearance.
3. **Group similar headers**: Headers with similar patterns should typically be at the same level.
4. **Check consistency**: Ensure headers that serve the same structural purpose have consistent levels. This takes precedence over symbol differences.
5. Identify any changes you'll need to make, and write a short analysis explaining the patterns you identified.
6. Output "no_corrections", or "corrections_needed", depending on whether you need to make changes.
7. If corrections are needed, output any blocks that need updates.  Only output the block ids and html, like this:
        ```json
        [
            {
                "id": "/page/0/SectionHeader/1",
                "html": "<h2>Introduction</h2>"
            },
            ...
        ]
        ```

**Example 1 - Numbering Pattern:**
Input:
Section Headers
```json
[
    {
        "bbox": [x1, y1, x2, y2],
        "id": "/page/0/SectionHeader/1",
        "page": 0,
        "html": "1 Vector Operations",
    },
    {
        "bbox": [x1, y1, x2, y2],
        "id": "/page/0/SectionHeader/2",
        "page": 0,
        "html": "1.1 Vector Addition",
    },
]
```
Output:
Analysis: The first section header is missing the h1 tag, and the second section header is missing the h2 tag. The numbering pattern (1 vs 1.1) indicates a hierarchical relationship.
```json
[
    {
        "id": "/page/0/SectionHeader/1",
        "html": "<h1>1 Vector Operations</h1>"
    },
    {
        "id": "/page/0/SectionHeader/2",
        "html": "<h2>1.1 Vector Addition</h2>"
    }
]
```

**Example 2 - Formatting Pattern Recognition with Symbol Variations:**
Input:
Section Headers
```json
[
    {
        "id": "/page/0/SectionHeader/1",
        "html": "<h3><b>→ Le 5 mai</b></h3>",
        "page": 0
    },
    {
        "id": "/page/0/SectionHeader/2",
        "html": "<h2><b>→ 20 juin</b></h2>",
        "page": 0
    },
    {
        "id": "/page/0/SectionHeader/3",
        "html": "<h2><b>* Dans la nuit du 4 août</b></h2>",
        "page": 0
    },
    {
        "id": "/page/0/SectionHeader/4",
        "html": "<h2><b>* Le 26 août</b></h2>",
        "page": 0
    }
]
```
Output:
Analysis: All four headers share the same formatting pattern (bold text with `<b>` tags) and are date-based timeline entries. Even though some use → and others use *, they all serve the same structural purpose (chronological timeline entries) and should be at the same level. The first header is incorrectly at h3 while the others are h2. They should all be h2 for consistency. **Key point**: Different symbols (→ vs *) do not necessarily indicate different header levels when the headers serve the same semantic purpose.
```json
[
    {
        "id": "/page/0/SectionHeader/1",
        "html": "<h2><b>→ Le 5 mai</b></h2>"
    }
]
```

**Input:**
Section Headers
```json
{{section_header_json}}
```
"""

    def get_selected_blocks(
        self,
        document: Document,
        page: PageGroup,
    ) -> List[dict]:
        selected_blocks = page.structure_blocks(document)
        json_blocks = [
            self.normalize_block_json(block, document, page, i)
            for i, block in enumerate(selected_blocks)
        ]
        return json_blocks

    def process_rewriting(
        self, document: Document, section_headers: List[Tuple[Block, dict]]
    ):
        section_header_json = [sh[1] for sh in section_headers]
        for item in section_header_json:
            _, _, page_id, block_type, block_id = item["id"].split("/")
            item["page"] = page_id
            item["width"] = item["bbox"][2] - item["bbox"][0]
            item["height"] = item["bbox"][3] - item["bbox"][1]
            del item["block_type"]  # Not needed, since they're all section headers

        prompt = self.page_prompt.replace(
            "{{section_header_json}}", json.dumps(section_header_json)
        )
        response = self.llm_service(
            prompt, None, document.pages[0], SectionHeaderSchema
        )
        logger.debug(f"Got section header reponse from LLM: {response}")

        if not response or "correction_type" not in response:
            logger.warning("LLM did not return a valid response")
            return

        correction_type = response["correction_type"]
        if correction_type == "no_corrections":
            return

        self.load_blocks(response)
        self.handle_rewrites(response["blocks"], document)

    def load_blocks(self, response):
        if isinstance(response["blocks"], str):
            response["blocks"] = json.loads(response["blocks"])

    def rewrite_blocks(self, document: Document):
        # Don't show progress if there are no blocks to process
        section_headers = [
            (block, self.normalize_block_json(block, document, page))
            for page in document.pages
            for block in page.structure_blocks(document)
            if block.block_type == BlockTypes.SectionHeader
        ]
        if len(section_headers) == 0:
            return

        pbar = tqdm(
            total=1,
            desc=f"Running {self.__class__.__name__}",
            disable=self.disable_tqdm,
        )

        self.process_rewriting(document, section_headers)
        pbar.update(1)
        pbar.close()


class BlockSchema(BaseModel):
    id: str
    html: str


class SectionHeaderSchema(BaseModel):
    analysis: str
    correction_type: str
    blocks: List[BlockSchema]

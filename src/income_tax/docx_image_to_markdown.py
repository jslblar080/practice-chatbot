"""
docx_image_to_markdown.py

docx 내 이미지를 Vision LLM(Claude)으로 분석하여
마크다운 테이블/설명으로 변환 후 output.docx로 저장.

Usage:
    pip install python-docx anthropic
    export ANTHROPIC_API_KEY="your-key"

    # CLI
    python docx_image_to_markdown.py input.docx output.docx [--full]

    # Python
    converter = DocxImageToMarkdown(api_key="sk-...")
    converter.convert("input.docx", "output.docx")
    print(converter.summary())
"""

import sys
import os
import base64

from docx import Document as open_docx
from docx.document import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from anthropic import Anthropic
from anthropic.types import (
    Base64ImageSourceParam,
    ImageBlockParam,
    MessageParam,
    TextBlock,
    TextBlockParam,
)


class DocxImageToMarkdown:
    """
    docx 내 이미지를 Vision LLM으로 분석하여 마크다운 텍스트로 교체하는 변환기.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API 키. 미지정 시 ANTHROPIC_API_KEY 환경변수 사용.
    model : str
        사용할 Claude 모델.
    min_image_bytes : int
        이 크기 미만의 이미지는 아이콘/장식으로 간주하여 건너뜀.
    font_name : str
        변환된 마크다운 텍스트에 적용할 고정폭 폰트.
    font_size : int
        폰트 크기 (half-point 단위, 18 = 9pt).
    include_headers_footers : bool
        True이면 헤더/푸터 내 이미지도 처리.
    tag_converted : bool
        True이면 변환된 영역을 [변환된 이미지 #N] 태그로 감쌈.
    system_prompt : str, optional
        Vision 모델에 전달할 시스템 프롬프트. 미지정 시 기본 프롬프트 사용.
    verbose : bool
        True이면 처리 과정을 stdout에 출력.
    """

    DEFAULT_SYSTEM_PROMPT = """\
당신은 이미지 분석 전문가입니다. 이미지를 분석하고 다음 규칙에 따라 응답하세요:

1. 표(table)가 포함된 이미지:
   - 정확한 markdown table 형식으로 변환
   - 헤더 행은 | --- | 구분자 사용
   - 셀 내용은 최대한 원본 그대로 유지
   - 병합된 셀이 있으면 적절히 처리

2. 차트/그래프 이미지:
   - 차트 유형, 축 레이블, 주요 데이터 포인트를 텍스트로 설명
   - 가능하면 핵심 수치를 markdown table로 정리

3. 일반 이미지(사진, 다이어그램 등):
   - 이미지 내용을 간결하게 설명
   - [이미지 설명: ...] 형식으로 작성

순수 마크다운만 반환하세요. 부가 설명이나 코드블록(```)은 사용하지 마세요."""

    _MEDIA_TYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5",
        min_image_bytes: int = 2048,
        font_name: str = "Consolas",
        font_size: int = 18,
        include_headers_footers: bool = False,
        tag_converted: bool = True,
        system_prompt: str | None = None,
        verbose: bool = True,
    ):
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self._model = model
        self._min_image_bytes = min_image_bytes
        self._font_name = font_name
        self._font_size = font_size
        self._include_headers_footers = include_headers_footers
        self._tag_converted = tag_converted
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._verbose = verbose

        # 변환 결과 (convert 호출 시 초기화)
        self.total_images: int = 0
        self.converted: int = 0
        self.skipped: int = 0
        self.failed: int = 0
        self.details: list[dict] = []

    def _reset_stats(self):
        """변환 통계 초기화."""
        self.total_images = 0
        self.converted = 0
        self.skipped = 0
        self.failed = 0
        self.details = []

    def summary(self) -> str:
        """마지막 변환 결과 요약 문자열 반환."""
        return (
            f"총 {self.total_images}개 이미지 | "
            f"변환 {self.converted} | 건너뜀 {self.skipped} | 실패 {self.failed}"
        )

    # ═══════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════

    def convert(self, input_path: str, output_path: str) -> "DocxImageToMarkdown":
        """
        docx를 읽어 이미지를 마크다운으로 변환한 뒤 output_path에 저장.
        체이닝을 위해 self를 반환.

        결과는 self.total_images, self.converted, self.skipped,
        self.failed, self.details, self.summary()로 조회.
        """
        self._reset_stats()
        doc = open_docx(input_path)

        self._log(f"입력: {input_path}")
        self._log(f"출력: {output_path}\n")

        # 본문
        self._process_element(doc.element.body, doc.part, "본문")

        # 헤더 / 푸터
        if self._include_headers_footers:
            self._process_headers_footers(doc)

        doc.save(output_path)
        self._log(f"\n{self.summary()}")
        self._log(f"저장 완료: {output_path}")
        return self

    # ═══════════════════════════════════════════
    # Vision API
    # ═══════════════════════════════════════════

    def _image_to_markdown(self, image_bytes: bytes, media_type: str) -> str:
        """Vision API를 호출하여 이미지를 마크다운으로 변환."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        image_block = ImageBlockParam(
            type="image",
            source=Base64ImageSourceParam(
                type="base64",
                media_type=media_type,  # type: ignore[arg-type]
                data=b64,
            ),
        )
        text_block = TextBlockParam(
            type="text",
            text="이 이미지를 마크다운으로 변환해주세요.",
        )
        message = MessageParam(
            role="user",
            content=[image_block, text_block],
        )

        resp = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=self._system_prompt,
            messages=[message],
        )

        for block in resp.content:
            if isinstance(block, TextBlock):
                return block.text
        raise ValueError("Vision API 응답에 텍스트 블록이 없습니다.")

    # ═══════════════════════════════════════════
    # 이미지 추출
    # ═══════════════════════════════════════════

    def _extract_image(self, drawing: OxmlElement, doc_part: object) -> tuple[bytes, str] | None:  # type: ignore
        """<w:drawing>에서 이미지 바이너리와 media_type을 추출. 실패 시 None."""
        blips = drawing.findall(".//" + qn("a:blip"))
        if not blips:
            return None
        embed_id = blips[0].get(qn("r:embed"))
        if not embed_id:
            return None
        try:
            rel = doc_part.rels[embed_id]  # type: ignore[union-attr]
            ext = os.path.splitext(rel.target_ref)[1].lower()
            media_type = self._MEDIA_TYPES.get(ext, "image/png")
            return rel.target_part.blob, media_type
        except (KeyError, AttributeError):
            return None

    # ═══════════════════════════════════════════
    # XML 조작
    # ═══════════════════════════════════════════

    def _build_paragraphs(self, markdown_text: str) -> list[OxmlElement]:  # type: ignore
        """마크다운 텍스트 → docx XML <w:p> 리스트."""
        paragraphs = []
        for line in markdown_text.split("\n"):
            p = OxmlElement("w:p")

            pPr = OxmlElement("w:pPr")
            sp = OxmlElement("w:spacing")
            sp.set(qn("w:after"), "0")
            sp.set(qn("w:line"), "240")
            sp.set(qn("w:lineRule"), "auto")
            pPr.append(sp)
            p.append(pPr)

            r = OxmlElement("w:r")
            rPr = OxmlElement("w:rPr")
            fonts = OxmlElement("w:rFonts")
            fonts.set(qn("w:ascii"), self._font_name)
            fonts.set(qn("w:hAnsi"), self._font_name)
            rPr.append(fonts)
            sz = OxmlElement("w:sz")
            sz.set(qn("w:val"), str(self._font_size))
            rPr.append(sz)
            r.append(rPr)

            t = OxmlElement("w:t")
            t.set(qn("xml:space"), "preserve")
            t.text = line
            r.append(t)
            p.append(r)
            paragraphs.append(p)

        return paragraphs

    def _replace_drawing(self, para: OxmlElement, drawing: OxmlElement, md_paragraphs: list[OxmlElement]) -> None:  # type: ignore
        """paragraph 내 <w:drawing>을 마크다운 paragraph들로 교체."""
        parent_run = drawing.getparent()
        if parent_run is not None and parent_run.tag == qn("w:r"):
            run_parent = parent_run.getparent()
            if run_parent is not None:
                run_parent.remove(parent_run)

        body = para.getparent()
        if body is not None:
            idx = list(body).index(para)
            for i, md_p in enumerate(md_paragraphs):
                body.insert(idx + 1 + i, md_p)

        runs = para.findall(qn("w:r"))
        has_text = any(
            t.text and t.text.strip() for r in runs for t in r.findall(qn("w:t"))
        )
        if not has_text and not runs and body is not None:
            body.remove(para)

    # ═══════════════════════════════════════════
    # 순회 & 처리
    # ═══════════════════════════════════════════

    def _process_drawing(self, para: OxmlElement, drawing: OxmlElement, doc_part: object, label: str) -> None:  # type: ignore
        """하나의 <w:drawing>을 처리."""
        self.total_images += 1
        idx = self.total_images

        extracted = self._extract_image(drawing, doc_part)

        if extracted is None:
            self._log(f"  [{label}] #{idx}: 추출 실패 → 건너뜀")
            self.skipped += 1
            self.details.append(
                {"index": idx, "status": "skipped", "reason": "extraction_failed"}
            )
            return

        image_bytes, media_type = extracted

        if len(image_bytes) < self._min_image_bytes:
            self._log(f"  [{label}] #{idx}: 크기 미달({len(image_bytes)}B) → 건너뜀")
            self.skipped += 1
            self.details.append(
                {"index": idx, "status": "skipped", "reason": "too_small"}
            )
            return

        kb = len(image_bytes) / 1024
        self._log(f"  [{label}] #{idx}: 변환 중... ({media_type}, {kb:.1f}KB)")

        try:
            md = self._image_to_markdown(image_bytes, media_type)
            self._log(f"           → 완료 ({len(md)}자)")

            if self._tag_converted:
                md = f"[변환된 이미지 #{idx}]\n{md}\n[/변환된 이미지 #{idx}]"

            self._replace_drawing(para, drawing, self._build_paragraphs(md))
            self.converted += 1
            self.details.append({"index": idx, "status": "converted", "chars": len(md)})

        except Exception as e:
            self._log(f"           → 실패: {e}")
            self.failed += 1
            self.details.append({"index": idx, "status": "failed", "error": str(e)})

    def _process_element(self, element: OxmlElement, doc_part: object, label: str) -> None:  # type: ignore
        """XML 요소 내 모든 paragraph의 drawing을 처리."""
        for para in list(element.findall(qn("w:p"))):
            for drawing in para.findall(".//" + qn("w:drawing")):
                self._process_drawing(para, drawing, doc_part, label)

    def _process_headers_footers(self, doc: Document) -> None:
        """모든 섹션의 헤더/푸터를 처리."""
        for section in doc.sections:
            parts = [
                (section.header, "헤더"),
                (section.first_page_header, "첫쪽 헤더"),
                (section.even_page_header, "짝수 헤더"),
                (section.footer, "푸터"),
                (section.first_page_footer, "첫쪽 푸터"),
                (section.even_page_footer, "짝수 푸터"),
            ]
            for part, label in parts:
                if part is None or part.is_linked_to_previous:
                    continue
                self._process_element(part.element, part.part, label)

    # ═══════════════════════════════════════════
    # 유틸
    # ═══════════════════════════════════════════

    def _log(self, msg: str):
        if self._verbose:
            print(msg)


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python docx_image_to_markdown.py <input.docx> [output.docx] [--full]"
        )
        print("  --full  헤더/푸터 내 이미지도 처리")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = (
        sys.argv[2]
        if len(sys.argv) > 2 and not sys.argv[2].startswith("--")
        else "output.docx"
    )
    full_mode = "--full" in sys.argv

    if not os.path.exists(input_file):
        print(f"파일을 찾을 수 없습니다: {input_file}")
        sys.exit(1)

    converter = DocxImageToMarkdown(include_headers_footers=full_mode)
    converter.convert(input_file, output_file)

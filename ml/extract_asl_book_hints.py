import argparse
import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


def load_allowed_words_from_views(views_path: Path):
	text = views_path.read_text(encoding="utf-8", errors="ignore")
	match = re.search(r"ALLOWED_SIGN_WORDS\s*=\s*(\[[\s\S]*?\])", text)
	if not match:
		raise RuntimeError("Could not parse ALLOWED_SIGN_WORDS from views.py")
	words = ast.literal_eval(match.group(1))
	return [str(w).strip() for w in words if str(w).strip()]


def norm_text(text: str) -> str:
	text = (text or "").replace("\u2014", "-").replace("\u2013", "-")
	text = re.sub(r"[ \t]+", " ", text)
	return text


def extract_blocks(page_text: str):
	# OCR/column extraction can be noisy, so keep patterns tolerant and concise.
	pattern = re.compile(
		r"Handshape:\s*(?P<hand>.+?)\s+"
		r"Orientation:\s*(?P<orientation>.+?)\s+"
		r"Location:\s*(?P<location>.+?)\s+"
		r"Movement:\s*(?P<movement>.+?)(?=\s+Handshape:|$)",
		re.IGNORECASE | re.DOTALL,
	)
	blocks = []
	for m in pattern.finditer(page_text):
		hand = norm_text(m.group("hand"))
		orientation = norm_text(m.group("orientation"))
		location = norm_text(m.group("location"))
		movement = norm_text(m.group("movement"))
		hint = (
			f"Handshape {hand}; Orientation {orientation}; "
			f"Location {location}; Movement {movement}"
		)
		blocks.append({"start": m.start(), "hint": hint})
	return blocks


def clean_hint(hint: str) -> str:
	hint = re.sub(r"\s+", " ", hint).strip()
	# Keep prompts compact.
	if len(hint) > 260:
		hint = hint[:257].rstrip() + "..."
	return hint


def main():
	parser = argparse.ArgumentParser(description="Extract book-based ASL sign hints from PDF")
	parser.add_argument("--pdf", required=True, help="Path to ASL handshape dictionary PDF")
	parser.add_argument(
		"--views",
		default=str(Path(__file__).resolve().parents[1] / "A2SL" / "views.py"),
		help="Path to views.py (for ALLOWED_SIGN_WORDS parsing)",
	)
	parser.add_argument(
		"--out",
		default=str(Path(__file__).resolve().parents[1] / "data" / "asl_book_sign_hints.json"),
		help="Output JSON path",
	)
	args = parser.parse_args()

	pdf_path = Path(args.pdf)
	views_path = Path(args.views)
	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	allowed_words = load_allowed_words_from_views(views_path)
	allowed_lower = {w.lower(): w for w in allowed_words}

	reader = PdfReader(str(pdf_path))
	pages = [norm_text(page.extract_text() or "") for page in reader.pages]
	blocks_by_page = [extract_blocks(t) for t in pages]

	results = {}
	for lower_word in allowed_lower:
		word_pattern = re.compile(rf"\b{re.escape(lower_word)}\b", re.IGNORECASE)
		best = None
		for page_idx, page_text in enumerate(pages):
			for wm in word_pattern.finditer(page_text):
				for block in blocks_by_page[page_idx]:
					dist = block["start"] - wm.start()
					# Prefer parameter blocks shortly after the word label.
					if 0 <= dist <= 1400:
						candidate = (dist, page_idx, block["hint"])
						if best is None or candidate < best:
							best = candidate
		if best is not None:
			results[allowed_lower[lower_word]] = clean_hint(best[2])

	payload = {
		"_meta": {
			"source_pdf": str(pdf_path),
			"extracted_at_utc": datetime.now(timezone.utc).isoformat(),
			"total_vocab_words": len(allowed_words),
			"matched_words": len(results),
		},
		"hints": results,
	}
	out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

	print(f"Saved: {out_path}")
	print(f"Matched: {len(results)} / {len(allowed_words)}")


if __name__ == "__main__":
	main()

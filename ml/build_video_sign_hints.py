import argparse
import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path


VIDEO_URL = "https://www.youtube.com/watch?v=ianCxd71xIo"
VIDEO_TITLE = '"100 Basic Signs" (American Sign Language) (www.lifeprint.com)'


def load_allowed_words(views_path: Path):
	text = views_path.read_text(encoding="utf-8", errors="ignore")
	match = re.search(r"ALLOWED_SIGN_WORDS\s*=\s*(\[[\s\S]*?\])", text)
	if not match:
		raise RuntimeError("Could not parse ALLOWED_SIGN_WORDS in views.py")
	words = ast.literal_eval(match.group(1))
	return [str(w).strip() for w in words if str(w).strip()]


def parse_inline_hints(views_text: str):
	# Parse entries like: "WORD: description | "
	pattern = re.compile(r'"([A-Z][A-Z \'\-]+):\s*([^"]+?)\s*\|\s*"')
	raw = {}
	for key, desc in pattern.findall(views_text):
		k = re.sub(r"\s+", " ", key).strip().lower()
		v = re.sub(r"\s+", " ", desc).strip()
		if k and v:
			raw[k] = v

	# Last entry in the long string does not end with "|", capture it too.
	last = re.search(r'"([A-Z][A-Z \'\-]+):\s*([^"\n]+)"\s*\)\s*$', views_text, flags=re.MULTILINE)
	if last:
		k = re.sub(r"\s+", " ", last.group(1)).strip().lower()
		v = re.sub(r"\s+", " ", last.group(2)).strip()
		if k and v:
			raw[k] = v
	return raw


def main():
	parser = argparse.ArgumentParser(description="Build video-referenced ASL hints JSON from views.py")
	parser.add_argument(
		"--views",
		default=str(Path(__file__).resolve().parents[1] / "A2SL" / "views.py"),
		help="Path to views.py",
	)
	parser.add_argument(
		"--out",
		default=str(Path(__file__).resolve().parents[1] / "data" / "asl_video_sign_hints.json"),
		help="Output JSON path",
	)
	args = parser.parse_args()

	views_path = Path(args.views)
	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	text = views_path.read_text(encoding="utf-8", errors="ignore")
	allowed = load_allowed_words(views_path)
	allowed_lookup = {w.lower(): w for w in allowed}
	inline = parse_inline_hints(text)

	hints = {}
	for lower, canonical in allowed_lookup.items():
		val = inline.get(lower)
		if val:
			hints[canonical] = val

	payload = {
		"_meta": {
			"source": "youtube_video",
			"video_url": VIDEO_URL,
			"video_title": VIDEO_TITLE,
			"built_at_utc": datetime.now(timezone.utc).isoformat(),
			"total_vocab_words": len(allowed),
			"matched_words": len(hints),
		},
		"hints": hints,
	}
	out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
	print(f"Saved: {out_path}")
	print(f"Matched: {len(hints)} / {len(allowed)}")


if __name__ == "__main__":
	main()

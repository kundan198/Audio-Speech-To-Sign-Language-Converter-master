from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login,logout
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from django.contrib.staticfiles import finders
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
import base64
import json
import os
import re
import uuid
from pathlib import Path
from functools import lru_cache
import urllib.request

ALLOWED_SIGN_WORDS = [
	"After", "Again", "Against", "Age", "All", "Alone", "Also", "And", "Ask", "At", "Be",
	"Beautiful", "Before", "Best", "Better", "Busy", "But", "Bye", "Can", "Cannot",
	"Change", "College", "Come", "Computer", "Day", "Distance", "Do", "Do Not",
	"Does Not", "Eat", "Engineer", "Fight", "Finish", "From", "Glitter", "Go", "God",
	"Gold", "Good", "Great", "Hand", "Hands", "Happy", "Hello", "Help", "Her", "Here",
	"His", "Home", "Homepage", "How", "Invent", "It", "Keep", "Language", "Laugh",
	"Learn", "ME", "More", "My", "Name", "Next", "Not", "Now", "Of", "On", "Our", "Out",
	"Pretty", "Right", "Sad", "Safe", "See", "Self", "Sign", "Sing", "So", "Sound",
	"Stay", "Study", "Talk", "Television", "Thank", "Thank You", "That", "They",
	"This", "Those", "Time", "To", "Type", "Us", "Walk", "Wash", "Way", "We", "Welcome",
	"What", "When", "Where", "Which", "Who", "Whole", "Whose", "Why", "Will", "With",
	"Without", "Words", "Work", "World", "Wrong", "You", "Your", "Yourself",
]

@lru_cache(maxsize=1)
def _load_video_sign_hints():
	"""Load ASL hints extracted from the configured reference video.

	Expected JSON format:
	1) {"WORD": "hint", ...}
	2) {"hints": {"WORD": "hint", ...}, "_meta": {...}}
	"""
	path = Path(__file__).resolve().parents[1] / "data" / "asl_video_sign_hints.json"
	if not path.exists():
		return {}
	try:
		raw = json.loads(path.read_text(encoding="utf-8"))
	except Exception:
		return {}

	if isinstance(raw, dict) and isinstance(raw.get("hints"), dict):
		raw = raw["hints"]
	if not isinstance(raw, dict):
		return {}

	out = {}
	for k, v in raw.items():
		if not isinstance(v, str):
			continue
		key = str(k).strip().lower()
		val = v.strip()
		if key and val:
			out[key] = val
	return out


def _video_sign_hints_for_vocab(vocab):
	"""Render video-based hints in the same inline format expected by the prompt."""
	hints = _load_video_sign_hints()
	if not hints:
		return ""
	parts = []
	for word in vocab:
		hint = hints.get(word.strip().lower())
		if hint:
			parts.append(f"{word.upper()}: {hint}")
	return " | ".join(parts)


def home_view(request):
	return render(request,'home.html')


def about_view(request):
	return render(request,'about.html')


def contact_view(request):
	return render(request,'contact.html')


def _sign_vocab():
	assets_dir = Path(__file__).resolve().parents[1] / "assets"
	return sorted(path.stem for path in assets_dir.glob("*.mp4"))


def _extract_json_list(text):
	text = (text or "").strip()
	if not text:
		return []
	try:
		data = json.loads(text)
	except json.JSONDecodeError:
		match = re.search(r"\[.*\]", text, re.DOTALL)
		if not match:
			return []
		try:
			data = json.loads(match.group())
		except json.JSONDecodeError:
			return []
	return [str(item).strip() for item in data if str(item).strip()] if isinstance(data, list) else []


def _clip_ready_words(words):
	"""Filter sign tokens to only those with an animation clip in assets/.
	Strict mode: words not in ALLOWED_SIGN_WORDS are dropped entirely -
	no letter-spelling fallback - so the output never contains stray letters."""
	allowed_set = {w.lower(): w for w in ALLOWED_SIGN_WORDS}
	return [allowed_set[w.lower()] for w in words if w.lower() in allowed_set]



# ---------------------------------------------------------------------------
# Ollama helper - fast local LLM, no API key needed
# Set OLLAMA_URL (default http://localhost:11434) and
# OLLAMA_MODEL (default llama3.2) in your .env to configure.
# ---------------------------------------------------------------------------

def _ollama_generate(prompt: str, *, max_tokens: int = 80, timeout: float = 6.0) -> str:
	"""Send a prompt to a local Ollama instance and return the response text.

	Returns empty string on any error so callers can fall back gracefully.
	Speed knobs:
	  - num_predict caps output tokens (fewer = faster)
	  - temperature 0 = greedy decode (no sampling overhead)
	  - num_ctx 512 keeps the KV-cache tiny for short prompts
	"""
	url = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/generate"
	model = os.environ.get("OLLAMA_MODEL", "llama3.2")
	body = json.dumps({
		"model": model,
		"prompt": prompt,
		"stream": False,
		"options": {
			"temperature": 0,
			"num_predict": max_tokens,
			"num_ctx": 512,
		},
	}).encode()
	req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
	try:
		with urllib.request.urlopen(req, timeout=timeout) as resp:
			data = json.loads(resp.read())
			return (data.get("response") or "").strip()
	except Exception:
		return ""

@login_required(login_url="login")
@require_http_methods(["POST"])
def translate_signs_view(request):
	"""JSON endpoint: text -> sign word list (no page reload)."""
	try:
		payload = json.loads(request.body.decode("utf-8"))
	except (ValueError, UnicodeDecodeError) as exc:
		return JsonResponse({"ok": False, "error": str(exc)}, status=400)

	text = (payload.get("text") or "").strip()
	if not text:
		return JsonResponse({"ok": False, "error": "No text provided."}, status=400)

	try:
		words = _convert_sentence_with_llm(text) or _convert_sentence_locally(text)
	except Exception:
		words = []
	# Final safety net: strip anything that isn't a known sign word
	words = _clip_ready_words(words)
	return JsonResponse({"ok": True, "words": words, "text": text})


def _convert_sentence_with_llm(sentence):
	"""Convert an English sentence to ASL sign tokens using Ollama.
	Output is strictly filtered to ALLOWED_SIGN_WORDS - no letter spelling."""
	allowed_list = "\n".join(ALLOWED_SIGN_WORDS)
	prompt = (
		"You are a Sign Language simplifier.\n"
		"Convert the sentence into a meaningful sequence using ONLY the allowed words below.\n\n"
		"Allowed words:\n" + allowed_list + "\n\n"
		"Rules:\n"
		"- Output ONLY a JSON list.\n"
		"- Do not use ANY word outside the allowed list.\n"
		"- Keep the meaning as close as possible.\n"
		"- Use ME for I or me.\n"
		"- Use Cannot for can\'t.\n"
		"- Use Do Not for don\'t.\n"
		"- Use Thank You for thanks.\n"
		"- Remove articles, prepositions, and filler words.\n\n"
		"Sentence: " + sentence + "\n\nJSON list:"
	)

	raw = _ollama_generate(prompt, max_tokens=120, timeout=float(os.environ.get("OLLAMA_TEXT_TIMEOUT", "8.0")))
	if raw:
		allowed = {word.lower(): word for word in ALLOWED_SIGN_WORDS}
		signs = [allowed[w.lower()] for w in _extract_json_list(raw) if w.lower() in allowed]
		if signs:
			return _clip_ready_words(signs)
	return []


# Alias so any existing callers still work
_convert_sentence_with_gemini = _convert_sentence_with_llm
def _convert_sentence_locally(text):
	# Lowercase the text so stop-word removal is case-insensitive.
	# Previously `text.lower()` was called but its return value was discarded
	# (strings are immutable), so stop words like "The", "Is", "Was" were never
	# removed and leaked into the sign output.
	text = text.lower()
	words = word_tokenize(text)

	tagged = nltk.pos_tag(words)
	tense = {}
	tense["future"] = len([word for word in tagged if word[1] == "MD"])
	tense["present"] = len([word for word in tagged if word[1] in ["VBP", "VBZ","VBG"]])
	tense["past"] = len([word for word in tagged if word[1] in ["VBD", "VBN"]])
	tense["present_continuous"] = len([word for word in tagged if word[1] in ["VBG"]])

	#stopwords that will be removed
	stop_words = set(["mightn't", 're', 'wasn', 'wouldn', 'be', 'has', 'that', 'does', 'shouldn', 'do', "you've",'off', 'for', "didn't", 'm', 'ain', 'haven', "weren't", 'are', "she's", "wasn't", 'its', "haven't", "wouldn't", 'don', 'weren', 's', "you'd", "don't", 'doesn', "hadn't", 'is', 'was', "that'll", "should've", 'a', 'then', 'the', 'mustn', 'i', 'nor', 'as', "it's", "needn't", 'd', 'am', 'have',  'hasn', 'o', "aren't", "you'll", "couldn't", "you're", "mustn't", 'didn', "doesn't", 'll', 'an', 'hadn', 'whom', 'y', "hasn't", 'itself', 'couldn', 'needn', "shan't", 'isn', 'been', 'such', 'shan', "shouldn't", 'aren', 'being', 'were', 'did', 'ma', 't', 'having', 'mightn', 've', "isn't", "won't"])

	#removing stopwords and applying lemmatizing nlp process to words
	lr = WordNetLemmatizer()
	filtered_text = []
	for w,p in zip(words,tagged):
		if w not in stop_words:
			if p[1]=='VBG' or p[1]=='VBD' or p[1]=='VBZ' or p[1]=='VBN' or p[1]=='NN':
				filtered_text.append(lr.lemmatize(w,pos='v'))
			elif p[1]=='JJ' or p[1]=='JJR' or p[1]=='JJS'or p[1]=='RBR' or p[1]=='RBS':
				filtered_text.append(lr.lemmatize(w,pos='a'))

			else:
				filtered_text.append(lr.lemmatize(w))

	#adding the specific word to specify tense
	words = filtered_text
	temp=[]
	for w in words:
		if w=='I':
			temp.append('ME')
		else:
			temp.append(w)
	words = temp
	# Only prepend a tense marker when at least one tense indicator was found.
	# Previously `max(tense, key=tense.get)` was called unconditionally: when all
	# counts are 0 it picks an arbitrary key (dict iteration order), which could
	# incorrectly prepend "Before", "Will", or "Now" to a neutral sentence like
	# "I eat food".
	max_count = max(tense.values())
	probable_tense = max(tense, key=tense.get) if max_count > 0 else None

	if probable_tense == "past" and tense["past"] >= 1:
		words = ["Before"] + words
	elif probable_tense == "future" and tense["future"] >= 1:
		if "Will" not in words:
			words = ["Will"] + words
	elif probable_tense == "present" and tense["present_continuous"] >= 1:
		words = ["Now"] + words

	# Strict filter: only keep words that exist in ALLOWED_SIGN_WORDS.
	# Words with no animation are dropped entirely  --  no letter-spelling fallback.
	allowed_set = {w.lower(): w for w in ALLOWED_SIGN_WORDS}
	filtered_text = []
	for w in words:
		canonical = allowed_set.get(w.lower())
		if canonical:
			filtered_text.append(canonical)
		# else: silently drop  --  never spell out individual letters
	return filtered_text

@login_required(login_url="login")
def animation_view(request):
	if request.method == 'POST':
		text = request.POST.get('sen')
		try:
			words = _convert_sentence_with_llm(text) or _convert_sentence_locally(text)
		except Exception:
			words = []
		# Final safety net: strip anything that isn't a known sign word
		words = _clip_ready_words(words)
		return render(request,'animation.html',{'words':words,'text':text})
	else:
		return render(request,'animation.html')




def signup_view(request):
	if request.method == 'POST':
		form = UserCreationForm(request.POST)
		if form.is_valid():
			user = form.save()
			login(request,user)
			request.session['login_splash_nonce'] = uuid.uuid4().hex
			# log the user in
			return redirect('animation')
	else:
		form = UserCreationForm()
	return render(request,'signup.html',{'form':form})



def login_view(request):
	if request.method == 'POST':
		form = AuthenticationForm(data=request.POST)
		if form.is_valid():
			#log in user
			user = form.get_user()
			login(request,user)
			request.session['login_splash_nonce'] = uuid.uuid4().hex
			if 'next' in request.POST:
				return redirect(request.POST.get('next'))
			else:
				return redirect('animation')
	else:
		form = AuthenticationForm()
	return render(request,'login.html',{'form':form})


def logout_view(request):
	request.session.pop('login_splash_nonce', None)
	logout(request)
	return redirect("home")


def _fast_point_classification(landmarks_seq):
	"""
	Apply fast geometric rules using MediaPipe landmarks to detect basic signs.
	landmarks_seq: list of frames, each containing a list of hands, each with 21 points (x,y,z).
	Returns a string label if matched, else None.
	"""
	if not landmarks_seq or not isinstance(landmarks_seq, list):
		return None
		
	# Flatten out empty frames or missing hands
	valid_frames = []
	for frame in landmarks_seq:
		# If it's a single frame of landmarks from sign_to_text.html, it might just be the points list directly
		if isinstance(frame, list) and len(frame) >= 21 and 'x' in frame[0]:
			valid_frames.append(frame)
		# If it's from conversation.html, it's a list of hands
		elif isinstance(frame, list) and len(frame) > 0 and isinstance(frame[0], list) and len(frame[0]) >= 21:
			valid_frames.append(frame[0])
			
	if not valid_frames:
		return None
		
	start_hand = valid_frames[0]
	end_hand = valid_frames[-1]
	
	def dist(p1, p2):
		return ((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)**0.5
		
	def is_open_hand(hand):
		# Fingertips (8,12,16,20) above MCPs (5,9,13,17) - note y goes down
		return (hand[8]['y'] < hand[5]['y'] and 
				hand[12]['y'] < hand[9]['y'] and 
				hand[16]['y'] < hand[13]['y'] and 
				hand[20]['y'] < hand[17]['y'])
				
	def thumb_extended(hand):
		return dist(hand[4], hand[5]) > 0.08
		
	open_start = is_open_hand(start_hand)
	open_end = is_open_hand(end_hand)
	
	y_shift = end_hand[0]['y'] - start_hand[0]['y']
	x_shift = abs(end_hand[0]['x'] - start_hand[0]['x'])
	motion_dist = dist(start_hand[0], end_hand[0])
	avg_y = sum(f[0]['y'] for f in valid_frames) / len(valid_frames)
	
	# Only dynamic signs if we have enough frames
	if len(valid_frames) >= 3:
		if open_start and open_end and y_shift > 0.08 and x_shift < 0.1:
			return "Thank You"
		if open_start and open_end and x_shift > 0.05 and y_shift < 0.05:
			return "Hello"
		if open_start and open_end and y_shift < -0.1:
			return "Morning"
			
	# Static signs (Mom, Dad)
	if open_start and thumb_extended(start_hand) and motion_dist < 0.05:
		if avg_y < 0.35:
			return "Dad"
		elif 0.35 <= avg_y < 0.6:
			return "Mom"
			
	return None


# -----------------------------------------------------------------------------
# Sign -> Text (reverse direction): webcam frames -> Gemini Vision -> ASL text
# -----------------------------------------------------------------------------

@login_required(login_url="login")
def sign_to_text_view(request):
	"""Render the sign-to-text page. Reverse direction of the existing tool:
	the user signs into the webcam and we transcribe to English text + speak it.
	"""
	return render(request, 'sign_to_text.html')


@login_required(login_url="login")
def train_signs_view(request):
	return render(request, 'train_signs.html', {"vocab": _sign_vocab()})


@login_required(login_url="login")
@require_http_methods(["GET"])
def training_stats_view(request):
	data_dir = Path(__file__).resolve().parents[1] / "data" / "sign_samples"
	counts = {}
	if data_dir.exists():
		for label_dir in data_dir.iterdir():
			if label_dir.is_dir():
				counts[label_dir.name] = len(list(label_dir.glob("*.json")))
	return JsonResponse({"ok": True, "counts": counts})


@login_required(login_url="login")
@require_http_methods(["POST"])
def save_training_sample_view(request):
	try:
		payload = json.loads(request.body.decode('utf-8'))
	except (ValueError, UnicodeDecodeError) as exc:
		return JsonResponse({"ok": False, "error": f"Invalid JSON: {exc}"}, status=400)

	label = (payload.get("label") or "").strip()
	landmarks = payload.get("landmarks") or []
	vocab = set(_sign_vocab())
	if label not in vocab:
		return JsonResponse({"ok": False, "error": "Unknown label."}, status=400)
	if len(landmarks) < 8:
		return JsonResponse({"ok": False, "error": "Record a longer motion sample."}, status=400)

	safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_")
	data_dir = Path(__file__).resolve().parents[1] / "data" / "sign_samples" / safe_label
	data_dir.mkdir(parents=True, exist_ok=True)
	sample = {
		"label": label,
		"landmarks": landmarks[:96],
		"user": request.user.username,
	}
	out = data_dir / f"{uuid.uuid4().hex}.json"
	out.write_text(json.dumps(sample))
	return JsonResponse({"ok": True, "label": label, "count": len(list(data_dir.glob('*.json')))})


@login_required(login_url="login")
@require_http_methods(["POST"])
def recognize_sign_view(request):
	"""Receive base64-encoded webcam frames (and optional MediaPipe landmark
	data) from the browser, send them to Gemini Vision for ASL recognition,
	and return the recognized text as JSON.

	Expected JSON body:
	  {
	    "frames": ["data:image/jpeg;base64,...", ...],   # 1-5 frames
	    "landmarks": [...],                                # optional, list of
	                                                       # MediaPipe hand landmark
	                                                       # arrays per frame
	    "context": "previous transcript so far"            # optional, for continuity
	  }

	Response:
	  { "ok": true, "text": "hello", "raw": "..." }
	  { "ok": false, "error": "..." }
	"""
	try:
		payload = json.loads(request.body.decode('utf-8'))
	except (ValueError, UnicodeDecodeError) as exc:
		return JsonResponse({"ok": False, "error": f"Invalid JSON: {exc}"}, status=400)

	frames = payload.get('frames') or []
	landmarks = payload.get('landmarks') or []
	context = (payload.get('context') or '').strip()

	if not frames:
		return JsonResponse({"ok": False, "error": "No frames provided."}, status=400)

	# 1. Fast geometric rule-based classification
	fast_class = _fast_point_classification(landmarks)
	if fast_class:
		return JsonResponse({
			"ok": True,
			"text": fast_class,
			"raw": fast_class,
			"engine": "fast-heuristic",
			"predictions": [],
		})

	wlasl_result = _recognize_with_wlasl(frames, landmarks)
	if wlasl_result and wlasl_result.get("text") and wlasl_result.get("text") != "[unclear]":
		return JsonResponse({
			"ok": True,
			"text": wlasl_result["text"],
			"raw": wlasl_result["text"],
			"engine": "wlasl-i3d",
			"predictions": wlasl_result.get("predictions", []),
		})

	# 3. Local keypoint GRU (fast, no API key, works from landmarks alone)
	keypoint_result = _recognize_with_keypoint(landmarks)
	if keypoint_result and keypoint_result.get("text") and keypoint_result.get("text") != "[unclear]":
		return JsonResponse({
			"ok": True,
			"text": keypoint_result["text"],
			"raw": keypoint_result["text"],
			"engine": "keypoint-gru",
			"predictions": keypoint_result.get("predictions", []),
		})

	api_key = os.environ.get('GEMINI_API_KEY', '').strip()
	if not api_key:
		# Return best available result rather than failing hard
		for result, engine in [(keypoint_result, "keypoint-gru"), (wlasl_result, "wlasl-i3d")]:
			if result:
				return JsonResponse({
					"ok": True,
					"text": result.get("text", "[unclear]"),
					"raw": result.get("text", "[unclear]"),
					"engine": engine,
					"predictions": result.get("predictions", []),
				})
		return JsonResponse({
			"ok": False,
			"error": "No recognizer available (GEMINI_API_KEY not set, keypoint/WLASL returned nothing).",
		}, status=500)

	# Lazy import so the rest of the app still works even if the SDK is missing.
	try:
		import google.generativeai as genai
	except ImportError:
		return JsonResponse({
			"ok": False,
			"error": "google-generativeai is not installed. Run: pip install google-generativeai",
		}, status=500)

	try:
		genai.configure(api_key=api_key)
		model = genai.GenerativeModel(os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash'))

		# Decode each base64 data URL into an inline image part.
		image_parts = []
		for frame in frames[:5]:  # cap to keep request fast
			if not isinstance(frame, str):
				continue
			# Strip "data:image/jpeg;base64," prefix if present.
			b64 = frame.split(',', 1)[-1]
			try:
				img_bytes = base64.b64decode(b64)
			except Exception:
				continue
			image_parts.append({
				"mime_type": "image/jpeg",
				"data": img_bytes,
			})

		if not image_parts:
			return JsonResponse({"ok": False, "error": "No decodable frames."}, status=400)

		landmark_hint = ''
		if landmarks:
			# Trim landmark payload so we don't blow up the prompt.
			try:
				short = json.dumps(landmarks)[:4000]
				landmark_hint = (
					"\n\nMediaPipe hand-landmark data (normalized x,y,z per joint, "
					"21 joints per hand) is provided as additional structural "
					"signal:\n" + short
				)
			except (TypeError, ValueError):
				landmark_hint = ''

		context_hint = f"\n\nPrior transcript so far: \"{context}\"" if context else ''

		prompt = (
			"You are an expert American Sign Language (ASL) interpreter. "
			"The following images are sequential frames of a single sign or "
			"short sequence of signs from a webcam. Identify the ASL sign(s) "
			"being performed and respond with the most likely ENGLISH "
			"translation.\n\n"
			"Rules:\n"
			"1. Respond with ONLY the English word(s) or short phrase that "
			"the signer is communicating. No explanations, no quotes, no "
			"alternative options.\n"
			"2. If you see ASL fingerspelling (one letter per frame), "
			"concatenate the letters into a word.\n"
			"3. If the gesture is unclear or no hands are visible, respond "
			"with exactly: [unclear]\n"
			"4. Keep the response under 10 words."
			+ context_hint
			+ landmark_hint
		)

		response = model.generate_content([prompt, *image_parts])
		raw = (response.text or '').strip()

		# Sanitize: remove surrounding quotes, trailing punctuation, and any
		# explanatory prefix the model occasionally adds.
		cleaned = raw.strip().strip('"').strip("'")
		# Take only the first line if the model was chatty.
		cleaned = cleaned.splitlines()[0] if cleaned else cleaned
		# Strip leading "Translation:" / "Sign:" labels if present.
		cleaned = re.sub(r'^(translation|sign|answer|asl)\s*[:\-]\s*', '', cleaned, flags=re.IGNORECASE)
		cleaned = cleaned.strip()

		return JsonResponse({"ok": True, "text": cleaned, "raw": raw})

	except Exception as exc:  # pragma: no cover - surface real errors to frontend
		return JsonResponse({"ok": False, "error": f"Gemini request failed: {exc}"}, status=500)


# -----------------------------------------------------------------------------
# Live Conversation (flagship): renders the two-panel page
# -----------------------------------------------------------------------------

@login_required(login_url="login")
def conversation_view(request):
	"""Render the side-by-side two-way conversation page."""
	return render(request, 'conversation.html')


@login_required(login_url="login")
@require_http_methods(["POST"])
def simplify_text_view(request):
	"""Use Ollama to simplify text for a given mode (e.g. healthcare jargon
	-> plain language). Returns { ok, text }.
	"""
	try:
		payload = json.loads(request.body.decode('utf-8'))
	except (ValueError, UnicodeDecodeError) as exc:
		return JsonResponse({"ok": False, "error": f"Invalid JSON: {exc}"}, status=400)

	text = (payload.get('text') or '').strip()
	mode = (payload.get('mode') or 'standard').strip().lower()
	if not text:
		return JsonResponse({"ok": False, "error": "No text provided."}, status=400)

	mode_prompts = {
		"healthcare": (
			"You are translating a doctor's words for a Deaf patient. "
			"Rewrite the following sentence in plain, friendly English a "
			"patient would understand. Replace every medical term with "
			"common words. Keep it short. Output ONLY the rewritten "
			"sentence, no preamble, no quotes."
		),
		"education": (
			"Rewrite the following sentence for a student in plain, "
			"clear English. Break complex ideas into simple ones. Keep "
			"the meaning. Output ONLY the rewritten sentence."
		),
	}
	system = mode_prompts.get(mode)
	if not system:
		# Standard mode: no rewriting.
		return JsonResponse({"ok": True, "text": text})

	prompt = system + "\n\nSentence: " + text

	out = _ollama_generate(prompt, max_tokens=80, timeout=float(os.environ.get("OLLAMA_TEXT_TIMEOUT", "8.0")))
	if out:
		out = out.splitlines()[0].strip().strip('"').strip("'")
		return JsonResponse({"ok": True, "text": out or text})

	return JsonResponse({"ok": True, "text": text, "note": "Ollama unavailable; returning original text."})


# -----------------------------------------------------------------------------
# Live single-frame Gemini Flash Vision recognition (fast path for live mode)
# -----------------------------------------------------------------------------

@login_required(login_url="login")
@require_http_methods(["POST"])
def live_recognize_view(request):
	"""Single JPEG frame ->' Gemini Flash Vision ->' ASL word.
	Optimised for the live sign-to-speech pipeline: one frame, tight prompt.

	Body:  { "frame": "data:image/jpeg;base64,...", "context": "words so far" }
	Response: { "ok": true, "text": "Hello" }
	"""
	try:
		payload = json.loads(request.body.decode('utf-8'))
	except (ValueError, UnicodeDecodeError) as exc:
		return JsonResponse({"ok": False, "error": str(exc)}, status=400)

	frame     = (payload.get('frame') or '').strip()
	context   = (payload.get('context') or '').strip()
	landmarks = payload.get('landmarks') or []  # optional: MediaPipe hand landmarks

	if not frame and not landmarks:
		return JsonResponse({"ok": False, "error": "No frame."}, status=400)

	# Fast path: local keypoint model (no API key, no image decode needed)
	if landmarks:
		kr = _recognize_with_keypoint(landmarks)
		if kr and kr.get("text") and kr.get("text") != "[unclear]":
			return JsonResponse({
				"ok": True, "text": kr["text"],
				"engine": "keypoint-gru",
				"predictions": kr.get("predictions", []),
			})

	if not frame:
		return JsonResponse({"ok": True, "text": "[unclear]", "note": "No frame and keypoint was unclear."})

	api_key = os.environ.get('GEMINI_API_KEY', '').strip()
	if not api_key:
		return JsonResponse({"ok": False, "error": "GEMINI_API_KEY not set and no landmarks provided."}, status=500)

	try:
		import google.generativeai as genai
	except ImportError:
		return JsonResponse({"ok": False, "error": "google-generativeai not installed."}, status=500)

	try:
		b64 = frame.split(',', 1)[-1]
		img_bytes = base64.b64decode(b64)
	except Exception as exc:
		return JsonResponse({"ok": False, "error": f"Bad frame: {exc}"}, status=400)

	ctx_hint = f'\nSentence context (words signed so far): "{context}"  --  use this to infer the next likely word.' if context else ''

	vocab_str = ', '.join(ALLOWED_SIGN_WORDS)
	SIGN_HINTS = (
		"AFTER: dominant flat hand slides forward over non-dominant flat hand | "
		"AGAIN: non-dominant hand flat palm up, dominant curved hand arcs and taps into palm | "
		"ALL: right open hand sweeps in wide arc around left hand | "
		"ALONE: index finger extended upright, moves in small repeated circles | "
		"AND: open hand with spread fingers, draws right while fingers close together | "
		"ASK: both hands together as if praying, pull toward body | "
		"BE: B-handshape touches lips then moves forward | "
		"BEAUTIFUL: open hand circles face, then fingers close into flat-O (or 'and' hand) | "
		"BEFORE: both flat hands face down, dominant hand draws back toward body | "
		"BETTER: B-hand at chin/mouth, moves upward and closes to A-handshape | "
		"BEST: B-hand at mouth moves up, transforms to A-hand (superlative of good) | "
		"BUSY: both B-hands with wrists touching, both hands rock back and forth | "
		"BYE: open flat hand waves side to side near shoulder (like waving goodbye) | "
		"CAN: both S-hands (fists) palms down, move downward simultaneously | "
		"CANNOT: right index finger strikes down across left index finger | "
		"CHANGE: both A-hands cross at wrists, then twist/swap positions | "
		"COLLEGE: non-dominant flat hand palm up, dominant flat hand circles above then lands | "
		"COME: both index fingers curved, pull/beckon toward body | "
		"COMPUTER: C-hand at temple or dominant C-hand moves up non-dominant arm | "
		"DAY: dominant arm bent at elbow horizontal, index points up and arm lowers like sunset | "
		"DISTANCE: D-hands or thumb+index point and move outward showing far away | "
		"DO: both C-hands palms down, move side to side in unison | "
		"DO NOT: D-hand + NOT, or index finger shakes side to side firmly | "
		"DOES NOT: similar to DO NOT, shake of head with sign | "
		"EAT: fingertips pinched (flat-O) tap toward mouth repeatedly | "
		"ENGINEER: E-hand or fingerspell; sometimes E slides across forehead | "
		"FIGHT: both S-hands (fists) cross and strike toward each other | "
		"FINISH: both open hands palms facing self, flick outward (fingers spread) suddenly | "
		"FROM: F-hand or index finger draws back toward body from starting point | "
		"GO: both index fingers point forward, move forward away from body | "
		"GOD: G-hand or B-hand arcs from forehead upward and out | "
		"GOLD: G-hand at ear/earlobe, then Y-hand shakes forward | "
		"GOOD: flat hand at chin/mouth, moves forward and down onto non-dominant palm | "
		"GREAT: both open hands move apart indicating large/great size | "
		"HAND: dominant flat hand slides across back of non-dominant flat hand | "
		"HANDS: both flat hands, dominant slides across non-dominant, then switch | "
		"HAPPY: flat hand brushes upward on chest, repeated motion | "
		"HELLO: open hand near temple/forehead, moves outward (salute-like wave) | "
		"HELP: dominant A-fist sits on non-dominant open palm, both hands rise together | "
		"HER: flat hand pushes sideways (toward the person referred to) or HER possessive | "
		"HERE: both flat hands palms up, move in horizontal circles in front of body | "
		"HIS: flat hand pushes forward toward male reference | "
		"HOME: fingers pinched touch cheek, then move to touch temple (eating then sleeping) | "
		"HOW: both curved/bent hands placed back-to-back, roll forward and upward | "
		"INVENT: index finger at forehead flicks upward (idea popping out) | "
		"IT: index finger points to an object or neutral space | "
		"KEEP: both K-hands, dominant K taps on top of non-dominant K | "
		"LANGUAGE: both L-hands near each other, move outward to sides | "
		"LAUGH: L-hands at corners of mouth, pull back and upward repeatedly | "
		"LEARN: dominant open hand on non-dominant palm, closes and moves up to forehead | "
		"ME: index finger points toward self/chest | "
		"MORE: both flat-O hands (pinched fingers), tap fingertips together repeatedly | "
		"MY: flat hand placed firmly on chest | "
		"NAME: both H-hands (index+middle extended), dominant H taps non-dominant H twice | "
		"NEXT: open hand behind other open hand, flips over and forward | "
		"NOT: thumb under chin, moves forward OR A-hand at chin moves forward | "
		"NOW: both Y-hands (or bent hands) drop straight down abruptly | "
		"OF: (usually fingerspelled or part of phrase) | "
		"ON: dominant flat hand places on top of non-dominant flat hand | "
		"OUR: curved hand at right shoulder sweeps to left shoulder | "
		"OUT: dominant hand inside non-dominant C-hand, pulls out and upward | "
		"PLEASE: flat hand placed on chest, moves in clockwise circle | "
		"RIGHT: R-hand or index+middle finger pointing right; or correct = R slides across | "
		"SAD: both open hands at face, fingers pointing up, hands drop slowly downward | "
		"SAFE: both S-hands cross at wrists in front of chest, then uncross outward | "
		"SEE: V-hand (index+middle finger) pointing from under eyes outward toward object | "
		"SELF: A-hand or index bounces forward toward reference | "
		"SIGN: both index fingers pointing up, circle alternately (like signing/language) | "
		"SING: dominant open hand waves back and forth over non-dominant arm (like conducting) | "
		"SO: (conjunction) flat hands held side by side; or S-O fingerspelled | "
		"SOUND: index finger at ear, circles forward | "
		"STAY: Y-hand pushes firmly downward (stay put) | "
		"STOP: dominant flat hand chops down onto non-dominant flat palm abruptly | "
		"STUDY: dominant open hand wiggles fingers over non-dominant flat palm | "
		"TALK: index finger at mouth alternately taps forward (speaking) | "
		"TELEVISION: T-V fingerspelled or TV sign with T and V handshapes | "
		"THANK YOU: flat hand (fingers together) at chin/lips moves forward toward other person | "
		"THANK: same as THANK YOU  --  flat hand from chin out | "
		"THAT: Y-hand drops down onto non-dominant flat palm | "
		"THEY: index finger sweeps in arc pointing to the group referenced | "
		"THIS: Y-hand or index finger taps on non-dominant palm | "
		"THOSE: index finger points and sweeps toward group of things | "
		"TIME: index finger taps wrist (where a watch would be) | "
		"TO: dominant index finger moves to touch tip of non-dominant index finger | "
		"TYPE: both hands in typing position, fingers move as if typing on keyboard | "
		"US: U-hand touches one shoulder then swings to touch other shoulder | "
		"WALK: both flat hands alternately move forward (mimicking walking feet) | "
		"WASH: dominant A-hand rubs in circular motion over non-dominant A-hand | "
		"WAY: both W-hands (or open hands) move forward in parallel tracks | "
		"WE: index finger touches own chest then swings to touch the other person(s) | "
		"WELCOME: dominant flat hand sweeps inward toward body (welcoming gesture) | "
		"WHAT: index finger waggles across the fingers of the other open hand | "
		"WHEN: index fingers circle each other then one lands on the other | "
		"WHERE: index finger waggles back and forth side to side | "
		"WHICH: both A-hands alternate up and down (choosing between options) | "
		"WHO: L-hand near lips, index finger makes small circle near chin | "
		"WHOLE: dominant flat hand circles over non-dominant flat hand then lands | "
		"WHOSE: W-hand near chin, transitions to S then forward (WHO + possessive) | "
		"WHY: W-hand at forehead, pulls down and changes to Y-handshape | "
		"WILL: flat hand at cheek/side of face, moves forward into space | "
		"WITH: both A-hands (fists) come together side by side | "
		"WITHOUT: both A-hands come together then separate, opening to open hands | "
		"WORDS: index finger and thumb pinch together, tap on non-dominant index finger | "
		"WORK: both S-hands (fists), dominant wrist taps on non-dominant wrist repeatedly | "
		"WORLD: both W-hands circle each other, dominant lands on top of non-dominant | "
		"WRONG: Y-hand knocks against chin | "
		"YOU: index finger points directly toward the other person | "
		"YOUR: flat hand pushes forward toward the other person (possession) | "
		"YOURSELF: A-hand or index bounces forward twice toward the other person"
	)
	video_hint_str = _video_sign_hints_for_vocab(ALLOWED_SIGN_WORDS)
	if video_hint_str:
		SIGN_HINTS = f"{video_hint_str} | {SIGN_HINTS}"
	prompt = (
		"You are a certified Deaf interpreter and ASL (American Sign Language) expert trained on lifeprint.com references.\n"
		"Analyze this single video frame and identify the ASL sign being performed.\n"
		f"{ctx_hint}\n\n"
		"=== STEP 1  --  Analyze these 4 ASL parameters in the image ===\n"
		"1. HANDSHAPE: fingers shape (fist/S, open/B, index/1, V/2, H, L, C, O, F, Y, W, etc.)\n"
		"2. LOCATION: forehead, temple, cheek, chin, lips, chest, shoulder, neutral space in front\n"
		"3. PALM ORIENTATION: toward signer, away, up, down, sideways left/right\n"
		"4. MOVEMENT: tap, circle, arc, slide, wave, chop, roll, wiggle  --  any motion blur visible\n\n"
		"=== STEP 2  --  Match to vocabulary (ONLY valid answers) ===\n"
		f"{vocab_str}\n\n"
		"=== STEP 3  --  Use these detailed sign descriptions to identify ===\n"
		f"{SIGN_HINTS}\n\n"
		"=== STEP 4  --  Output ===\n"
		"Respond with ONLY the single matching word or phrase from the vocabulary list.\n"
		"- NEVER output individual letters unless the sign is clearly ASL fingerspelling AND no vocabulary word fits.\n"
		"- Do NOT add punctuation, explanation, or alternatives.\n"
		"- If no hand is visible or truly unrecognizable: unclear"
	)

	# Try models in order until one works
	VISION_MODELS = [
		'gemini-2.5-flash',
		'gemini-2.5-flash-preview-04-17',
		'gemini-2.0-flash-lite',
		'gemini-1.5-flash-latest',
		'gemini-pro-vision',
	]
	override = os.environ.get('GEMINI_MODEL', '').strip()
	if override:
		VISION_MODELS = [override] + VISION_MODELS

	last_err = 'No model worked'
	for model_name in VISION_MODELS:
		try:
			genai.configure(api_key=api_key)
			model = genai.GenerativeModel(model_name)
			response = model.generate_content([
				prompt,
				{"mime_type": "image/jpeg", "data": img_bytes},
			])
			raw  = (response.text or '').strip().strip('"').strip("'")
			word = raw.splitlines()[0].strip() if raw else ''
			word = re.sub(r'[^A-Za-z\-\' ]', '', word).strip().split()[0] if word.split() else ''
			if not word or word.lower() in ('unclear', 'unknown', 'none', 'no', ''):
				return JsonResponse({"ok": True, "text": "[unclear]", "prompt": prompt, "raw": raw, "model": model_name})
			return JsonResponse({"ok": True, "text": word.title(), "prompt": prompt, "raw": raw, "model": model_name})
		except Exception as exc:
			last_err = f'{model_name}: {exc}'
			continue
	return JsonResponse({"ok": False, "error": last_err, "prompt": prompt}, status=500)


# -----------------------------------------------------------------------------
# Batch recognize: all sign frames in ONE Gemini call ->' all words at once
# -----------------------------------------------------------------------------

@login_required(login_url="login")
@require_http_methods(["POST"])
def batch_recognize_view(request):
	"""Receive multiple JPEG frames (one per captured sign) and return all words
	in a single Gemini Flash Vision call.

	Body:  { "frames": ["data:image/jpeg;base64,...", ...] }
	Response: { "ok": true, "words": ["Hello", "My", "Name"] }
	"""
	try:
		payload = json.loads(request.body.decode('utf-8'))
	except (ValueError, UnicodeDecodeError) as exc:
		return JsonResponse({"ok": False, "error": str(exc)}, status=400)

	frames = payload.get('frames') or []
	if not frames:
		return JsonResponse({"ok": False, "error": "No frames."}, status=400)

	api_key = os.environ.get('GEMINI_API_KEY', '').strip()
	if not api_key:
		return JsonResponse({"ok": False, "error": "GEMINI_API_KEY not set."}, status=500)

	try:
		import google.generativeai as genai
	except ImportError:
		return JsonResponse({"ok": False, "error": "google-generativeai not installed."}, status=500)

	# Decode each frame
	image_parts = []
	for i, frame in enumerate(frames[:10]):   # cap at 10 signs per sentence
		try:
			b64 = frame.split(',', 1)[-1]
			image_parts.append({"mime_type": "image/jpeg", "data": base64.b64decode(b64)})
		except Exception:
			pass   # skip bad frames

	if not image_parts:
		return JsonResponse({"ok": False, "error": "No valid frames decoded."}, status=400)

	n = len(image_parts)
	vocab_str = ', '.join(ALLOWED_SIGN_WORDS)
	video_hint_str = _video_sign_hints_for_vocab(ALLOWED_SIGN_WORDS)
	video_hint_block = f"{video_hint_str}\n\n" if video_hint_str else ""
	prompt = (
		f"You are a certified Deaf interpreter and ASL (American Sign Language) expert trained on lifeprint.com references.\n"
		f"I am showing you {n} video frame(s) in chronological order. Each frame shows a person holding a DIFFERENT ASL sign.\n\n"
		"=== For EACH frame, analyze these 4 ASL parameters ===\n"
		"1. HANDSHAPE: fist/S, open/B, index/1, V/2, H, L, C, O, F, Y, W  --  what is the exact shape?\n"
		"2. LOCATION: forehead, temple, cheek, chin, lips, chest, shoulder, neutral space in front\n"
		"3. PALM ORIENTATION: toward signer, away, up, down, left, right\n"
		"4. MOVEMENT: tap, circle, arc, slide, wave, chop, roll, wiggle, drop  --  any motion blur?\n\n"
		"=== Valid vocabulary (ONLY answers from this list) ===\n"
		f"{vocab_str}\n\n"
		"=== Detailed sign reference (lifeprint.com style) ===\n"
		f"{video_hint_block}"
		"AFTER: dominant flat hand slides forward over non-dominant flat hand | "
		"AGAIN: non-dominant palm up flat, dominant curved hand arcs and taps into palm | "
		"ALL: right open hand sweeps in wide arc around left hand | "
		"AND: open spread hand draws right while fingers close together | "
		"ASK: both hands together prayer-like, pull toward body | "
		"BEAUTIFUL: open hand circles face, fingers close into flat-O | "
		"BEFORE: both flat hands face down, dominant draws back toward body | "
		"BETTER: B-hand at chin moves upward closing to A-handshape | "
		"BUSY: both B-hands wrists touching, rock back and forth | "
		"BYE: open flat hand waves side to side (goodbye wave) | "
		"CAN: both S-fists palms down, move downward simultaneously | "
		"CANNOT: right index strikes down across left index finger | "
		"CHANGE: both A-hands cross at wrists then twist/swap | "
		"COME: both curved index fingers beckon/pull toward body | "
		"DAY: dominant arm horizontal, index up, arm lowers like sunset | "
		"DO: both C-hands palms down, move side to side in unison | "
		"EAT: fingertips pinched (flat-O) tap toward mouth repeatedly | "
		"FIGHT: both S-fists cross and strike toward each other | "
		"FINISH: both open hands palms toward self, flick suddenly outward | "
		"GO: both index fingers point forward, move away from body | "
		"GOOD: flat hand at chin, moves forward and down onto non-dominant palm | "
		"HAPPY: flat hand brushes upward on chest repeatedly | "
		"HELLO: open hand near forehead/temple, moves outward (salute wave) | "
		"HELP: dominant A-fist on non-dominant open palm, both hands rise | "
		"HOME: pinched fingers touch cheek then temple (eat+sleep) | "
		"HOW: both curved hands back-to-back, roll forward and upward | "
		"INVENT: index at forehead flicks upward (idea emerging) | "
		"KEEP: both K-hands, dominant K taps non-dominant K | "
		"LAUGH: L-hands at corners of mouth, pull back and up | "
		"LEARN: open hand on non-dominant palm, closes moving to forehead | "
		"ME: index finger points to own chest/self | "
		"MORE: both flat-O hands tap fingertips together repeatedly | "
		"MY: flat hand placed firmly on chest | "
		"NAME: both H-hands (index+middle), dominant H taps non-dominant H twice | "
		"NOT: thumb under chin moves forward OR A-hand at chin moves forward | "
		"NOW: both Y-hands or bent hands drop straight down abruptly | "
		"ON: dominant flat hand places on top of non-dominant flat hand | "
		"PLEASE: flat hand on chest moves in clockwise circle | "
		"RIGHT: correct = R-hand slides across; direction = point right | "
		"SAD: both open hands at face drop slowly downward | "
		"SAFE: both S-hands cross at chest, then uncross outward | "
		"SEE: V-hand from under eyes points outward toward object | "
		"SIGN: both index fingers pointing up, circle alternately | "
		"SING: dominant hand waves back and forth over non-dominant arm | "
		"SORRY: A-hand or S-hand circles on chest | "
		"STAY: Y-hand pushes firmly downward | "
		"STOP: dominant flat hand chops abruptly onto non-dominant flat palm | "
		"STUDY: dominant open hand wiggles fingers over non-dominant flat palm | "
		"TALK: index finger at mouth alternately taps forward | "
		"THANK YOU: flat hand at chin/lips moves forward toward other person | "
		"THAT: Y-hand drops onto non-dominant flat palm | "
		"THEY: index finger sweeps in arc toward the group | "
		"TIME: index finger taps wrist where watch would be | "
		"TO: dominant index moves to touch tip of non-dominant index | "
		"WALK: both flat hands alternately move forward like walking feet | "
		"WASH: dominant A-hand rubs in circle over non-dominant A-hand | "
		"WELCOME: dominant flat hand sweeps inward toward body | "
		"WHAT: index finger waggles across fingers of other open hand | "
		"WHEN: index fingers circle each other then one lands on the other | "
		"WHERE: index finger waggles back and forth side to side | "
		"WHICH: both A-hands alternate up and down (choosing) | "
		"WHO: L-hand near lips, index makes small circle near chin | "
		"WHY: W-hand at forehead pulls down changing to Y-handshape | "
		"WILL: flat hand at cheek/face, moves forward into space | "
		"WITH: both A-fists come together side by side | "
		"WITHOUT: both A-fists come together then separate opening to open hands | "
		"WORK: both S-fists, dominant wrist taps non-dominant wrist repeatedly | "
		"WORLD: both W-hands circle each other, dominant lands on top | "
		"WRONG: Y-hand knocks against chin | "
		"YOU: index finger points directly toward the other person | "
		"YOUR: flat hand pushes forward toward the other person | "
		"YOURSELF: A-hand or index bounces forward twice toward other person\n\n"
		"=== Output format ===\n"
		f"Output ONLY {n} word(s)/phrase(s) from the vocabulary list, separated by commas.\n"
		"Example for 3 frames: Hello, My, Name\n"
		"- NEVER use individual letters unless the frame clearly shows fingerspelling AND no vocab word fits.\n"
		"- No numbering, no explanations, no extra text  --  just comma-separated words.\n"
		"- If a frame has no visible hand or is truly unrecognizable, write: unclear"
	)

	VISION_MODELS = [
		'gemini-2.5-flash',
		'gemini-2.5-flash-preview-04-17',
		'gemini-2.0-flash-lite',
		'gemini-1.5-flash-latest',
		'gemini-pro-vision',
	]
	override = os.environ.get('GEMINI_MODEL', '').strip()
	if override:
		VISION_MODELS = [override] + VISION_MODELS

	genai.configure(api_key=api_key)
	response = None
	used_model = None
	last_err = 'No model worked'
	for model_name in VISION_MODELS:
		try:
			model = genai.GenerativeModel(model_name)
			content = [prompt] + image_parts
			response = model.generate_content(content)
			used_model = model_name
			break
		except Exception as exc:
			last_err = f'{model_name}: {exc}'
			continue

	if response is None:
		return JsonResponse({"ok": False, "error": last_err}, status=500)

	try:
		raw = (response.text or '').strip()

		# Parse comma-separated reply into a clean word list
		raw_words = [w.strip().strip('"').strip("'") for w in raw.split(',')]
		words = []
		for w in raw_words:
			w = re.sub(r'[^A-Za-z\-\' ]', '', w).strip()
			if w and w.lower() not in ('unclear', 'unknown', 'none', ''):
				words.append(w.title())

		return JsonResponse({"ok": True, "words": words, "raw": raw, "prompt": prompt})
	except Exception as exc:
		return JsonResponse({"ok": False, "error": str(exc)}, status=500)


# -----------------------------------------------------------------------------
# Ultra-fast landmark-only hand classification (no image, text API only)
# -----------------------------------------------------------------------------

@login_required(login_url="login")
@require_http_methods(["POST"])
def classify_hand_view(request):
	"""Classify ASL sign from hand geometry features  --  no image, text-only Gemini call.

	Body:  { "features": "Thumb:ext, Index:up, Middle:up, Ring:down, Pinky:down | ...", "context": "..." }
	Response: { "ok": true, "text": "Hello" }
	"""
	try:
		payload = json.loads(request.body.decode('utf-8'))
	except (ValueError, UnicodeDecodeError) as exc:
		return JsonResponse({"ok": False, "error": str(exc)}, status=400)

	features = (payload.get('features') or '').strip()
	context  = (payload.get('context')  or '').strip()
	landmarks = payload.get('landmarks') or []

	if not features and not landmarks:
		return JsonResponse({"ok": False, "error": "No features."}, status=400)
		
	# 1. Fast geometric rule-based classification
	fast_class = _fast_point_classification([landmarks]) if landmarks else None
	if fast_class:
		return JsonResponse({"ok": True, "text": fast_class})

	# 2. Local keypoint GRU (primary - no API key needed)
	if landmarks:
		kr = _recognize_with_keypoint(landmarks)
		if kr:
			return JsonResponse({
				"ok": True,
				"text": kr["text"],
				"engine": kr.get("model", "keypoint-gru"),
				"predictions": kr.get("predictions", []),
			})

	# 3. Gemini text fallback (landmarks missing or model absent)
	api_key = os.environ.get('GEMINI_API_KEY', '').strip()
	if not api_key:
		return JsonResponse({"ok": True, "text": "[unclear]", "note": "No recognizer available."})

	try:
		import google.generativeai as genai
	except ImportError:
		return JsonResponse({"ok": False, "error": "google-generativeai not installed."}, status=500)

	ctx_hint = f' (words signed so far: "{context}")' if context else ''
	prompt = (
		f"You are a real-time ASL sign classifier{ctx_hint}.\n"
		f"MediaPipe hand data: {features}\n\n"
		"What single ASL word or letter does this hand shape represent?\n"
		"Rules: reply with ONE word or ONE letter only. No explanation. No punctuation. "
		"If genuinely unclear reply: unclear"
	)
	try:
		genai.configure(api_key=api_key)
		model = genai.GenerativeModel(os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash'))
		response = model.generate_content(prompt)
		raw  = (response.text or '').strip()
		word = raw.split()[0] if raw.split() else ''
		word = re.sub(r'[^A-Za-z\-]', '', word).strip()
		if not word or word.lower() in ('unclear', 'unknown', 'none'):
			return JsonResponse({"ok": True, "text": "[unclear]"})
		return JsonResponse({"ok": True, "text": word.title()})
	except Exception as exc:
		return JsonResponse({"ok": False, "error": str(exc)}, status=500)


# -----------------------------------------------------------------------------
# Sign words -> natural English sentence (Ollama)
# -----------------------------------------------------------------------------

def _rule_based_sentence(words):
	"""Construct a basic natural English sentence from ASL sign tokens without LLM."""
	if not words:
		return ""
	w = [x.lower() for x in words]

	# Single-word quick map
	if len(words) == 1:
		single = {
			"hello": "Hello there!", "hi": "Hi!", "bye": "Goodbye!",
			"good": "That's great!", "great": "That's great!", "best": "That's the best!",
			"help": "Can you help me?", "thank you": "Thank you!", "thank": "Thank you!",
			"happy": "I'm so happy!", "sad": "I'm feeling sad.", "sorry": "I'm sorry.",
			"yes": "Yes!", "no": "No.", "please": "Please help me.",
			"learn": "I want to learn.", "study": "I need to study.",
			"go": "Let's go!", "come": "Please come here.", "eat": "Let's eat!",
			"work": "I'm working.", "home": "I'm going home.", "name": "What is your name?",
			"me": "It's me!", "you": "How are you?", "beautiful": "That's beautiful!",
			"wrong": "Something is wrong.", "safe": "You are safe.", "stay": "Please stay.",
			"stop": "Please stop.", "now": "Do it now!", "time": "What's the time?",
		}
		return single.get(w[0], words[0] + ".")

	# Greeting prefix
	if w[0] == "hello":
		rest = _rule_based_sentence(words[1:])
		return "Hello! " + rest

	# ME / My subject
	if w[0] in ("me", "my"):
		if len(words) == 2 and w[1] == "name":
			return "What is my name?"
		return "My " + " ".join(words[1:]).lower() + "."

	# Question starters
	question_words = {"what", "when", "where", "who", "why", "how", "which", "whose"}
	if w[0] in question_words:
		return " ".join(words) + "?"

	# Thank / Thank You
	if w[0] in ("thank", "thank you"):
		return "Thank you so much!"

	# Help pattern
	if w[0] == "help":
		return "Please help me " + " ".join(words[1:]).lower() + "." if len(words) > 1 else "Can you help me?"

	# Default: capitalise first word, join rest, end with period
	return words[0].capitalize() + " " + " ".join(words[1:]).lower() + "."


def _ollama_sentence_is_valid(sentence, words):
	"""Return False if Ollama just echoed the input words back unchanged."""
	norm_out = re.sub(r'[^a-z\s]', '', sentence.lower()).strip()
	norm_in  = re.sub(r'[^a-z\s]', '', ' '.join(words).lower()).strip()
	# Reject if output is identical to or a trivial prefix/suffix of the input
	return norm_out != norm_in and norm_in not in norm_out


@login_required(login_url="login")
@require_http_methods(["POST"])
def formulate_sentence_view(request):
	"""Sign words -> natural English sentence via Ollama, with rule-based fallback."""
	try:
		payload = json.loads(request.body.decode('utf-8'))
	except (ValueError, UnicodeDecodeError) as exc:
		return JsonResponse({"ok": False, "error": f"Invalid JSON: {exc}"}, status=400)

	words = [str(w).strip() for w in (payload.get('words') or []) if str(w).strip()]
	if not words:
		return JsonResponse({"ok": False, "error": "No words provided."}, status=400)

	prompt = (
		"Task: turn these ASL sign tokens into one fluent English sentence.\n"
		"Signs: " + ", ".join(words) + "\n"
		"Rules: output ONLY the sentence — no labels, no quotes, no extra text.\n"
		"Make it natural: add pronouns, verbs, articles as needed.\n\n"
		"Examples:\n"
		"Signs: Hello => Hello there!\n"
		"Signs: Good => That's great!\n"
		"Signs: Help => Can you help me?\n"
		"Signs: Hello, Good => Hello, hope you're doing well!\n"
		"Signs: ME, Name => What is my name?\n"
		"Signs: You, Happy => Are you happy?\n"
		"Signs: Help, ME, Learn => Please help me learn.\n"
		"Signs: Thank You => Thank you so much!\n"
		"Signs: Walk, Home => I am walking home.\n"
		"Signs: Study, College => I study at college.\n\n"
		"Signs: " + ", ".join(words) + " =>"
	)

	ollama_out = _ollama_generate(
		prompt, max_tokens=60,
		timeout=float(os.environ.get("OLLAMA_TEXT_TIMEOUT", "8.0")),
	)

	if ollama_out:
		sentence = ollama_out.splitlines()[0].strip().strip('"').strip("'")
		sentence = re.sub(r'^[=->\s]+', '', sentence).strip()
		if sentence and _ollama_sentence_is_valid(sentence, words):
			return JsonResponse({"ok": True, "sentence": sentence, "provider": "ollama"})

	# Ollama failed or echoed the input — use rule-based construction
	sentence = _rule_based_sentence(words)
	return JsonResponse({"ok": True, "sentence": sentence, "provider": "rules"})


# -----------------------------------------------------------------------------
# ElevenLabs voice synthesis (server-side proxy so the API key stays private)
# -----------------------------------------------------------------------------

ELEVEN_BASE = "https://api.elevenlabs.io/v1"
WLASL_DEFAULT_URL = "http://127.0.0.1:8766/predict"




def _recognize_with_keypoint(landmarks, min_confidence=None, min_margin=None):
    """Run the local GRU keypoint classifier directly in-process.

    Loads the model once and caches it.  Returns the same dict shape as
    _recognize_with_wlasl so callers can treat both interchangeably:
      {"ok": True, "text": "Hello", "predictions": [...], "model": "keypoint-gru"}
    Returns None when the model file is missing or landmarks are empty.
    """
    if not landmarks:
        return None
    import sys
    from pathlib import Path as _Path
    ROOT = _Path(__file__).resolve().parents[1]
    model_path  = ROOT / "models" / "keypoint_sign" / "keypoint_model.pt"
    labels_path = ROOT / "models" / "keypoint_sign" / "labels.json"
    if not model_path.exists() or not labels_path.exists():
        return None

    # Lazy import heavy deps so startup stays fast
    try:
        import torch
        import numpy as np
    except ImportError:
        return None

    # Add ml/ to path so we can import keypoint_features
    ml_path = str(ROOT / "ml" / "wlasl_i3d")
    if ml_path not in sys.path:
        sys.path.insert(0, ml_path)
    try:
        from keypoint_features import landmarks_to_array, FEATURE_SIZE
    except ImportError:
        return None

    # --- model cache (module-level singleton) ---
    cache = _recognize_with_keypoint.__dict__
    mtime = model_path.stat().st_mtime
    if cache.get("mtime") != mtime or cache.get("model") is None:
        import json as _json
        labels = _json.loads(labels_path.read_text())
        # Inline the GRU architecture (same as train_keypoint_model.py)
        import torch.nn as nn
        class _GRU(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.gru = nn.GRU(FEATURE_SIZE, 192, 2, batch_first=True,
                                  bidirectional=True, dropout=0.25)
                self.head = nn.Sequential(
                    nn.LayerNorm(384), nn.Dropout(0.25), nn.Linear(384, n))
            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(out[:, -1, :])

        ckpt  = torch.load(model_path, map_location="cpu")
        model = _GRU(len(labels))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        cache["model"]  = model
        cache["labels"] = labels
        cache["mtime"]  = mtime

    model  = cache["model"]
    labels = cache["labels"]

    conf_thresh   = float(min_confidence or os.environ.get("KEYPOINT_MIN_CONFIDENCE", "0.75"))
    margin_thresh = float(min_margin      or os.environ.get("KEYPOINT_MIN_MARGIN",     "0.15"))

    import torch
    import numpy as np
    arr    = landmarks_to_array(landmarks)
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor)[0], dim=0)
    top_vals, top_idx = torch.topk(probs, k=min(5, len(labels)))
    predictions = [
        {"label": labels[int(i)], "confidence": float(v), "class_id": int(i)}
        for v, i in zip(top_vals, top_idx)
    ]
    best      = predictions[0]
    runner_up = predictions[1]["confidence"] if len(predictions) > 1 else 0.0
    confident = best["confidence"] >= conf_thresh and (best["confidence"] - runner_up) >= margin_thresh
    return {
        "ok":           True,
        "text":         best["label"] if confident else "[unclear]",
        "predictions":  predictions,
        "model":        "keypoint-gru",
    }

def _recognize_with_wlasl(frames, landmarks=None):
	"""Try the local WLASL I3D recognizer.

	The PyTorch model runs in a separate Python 3.12 process so Django can keep
	using its existing environment. Return None when the local model is disabled
	or unavailable, letting the Gemini fallback handle the request.
	"""
	if os.environ.get('SIGN_RECOGNITION_ENGINE', 'wlasl').lower() in ('gemini', 'off', 'disabled'):
		return None
	url = os.environ.get('WLASL_SERVER_URL', WLASL_DEFAULT_URL).strip()
	if not url:
		return None
	try:
		import requests
		response = requests.post(
			url,
			json={
				"frames": frames[:16],
				"landmarks": (landmarks or [])[:64],
				"top_k": 5,
				"min_confidence": float(os.environ.get('WLASL_MIN_CONFIDENCE', '0.55')),
				"min_margin": float(os.environ.get('WLASL_MIN_MARGIN', '0.15')),
				"target_frames": int(os.environ.get('WLASL_TARGET_FRAMES', '48')),
			},
			timeout=float(os.environ.get('WLASL_TIMEOUT', '0.8')),
		)
		if response.status_code != 200:
			return None
		data = response.json()
		if not data.get("ok"):
			return None
		return data
	except Exception:
		return None



@login_required(login_url="login")
def elevenlabs_voices_view(request):
	api_key = os.environ.get('ELEVENLABS_API_KEY', '').strip()
	if not api_key:
		return JsonResponse({"ok": False, "error": "ELEVENLABS_API_KEY not set."})
	try:
		import requests
		r = requests.get(f"{ELEVEN_BASE}/voices", headers={"xi-api-key": api_key}, timeout=10)
		if r.status_code != 200:
			return JsonResponse({"ok": False, "error": f"ElevenLabs {r.status_code}"})
		voices = [{"voice_id": v["voice_id"], "name": v["name"], "category": v.get("category", "")}
				  for v in r.json().get("voices", [])]
		return JsonResponse({"ok": True, "voices": voices})
	except Exception as exc:
		return JsonResponse({"ok": False, "error": str(exc)})


@login_required(login_url="login")
@require_http_methods(["POST"])
def elevenlabs_tts_view(request):
	try:
		payload = json.loads(request.body.decode('utf-8'))
	except (ValueError, UnicodeDecodeError) as exc:
		return JsonResponse({"ok": False, "error": str(exc)}, status=400)

	text     = (payload.get('text') or '').strip()
	voice_id = (payload.get('voice_id') or 'EXAVITQu4vr4xnSDxMaL').strip()
	api_key  = os.environ.get('ELEVENLABS_API_KEY', '').strip()

	if not text:
		return JsonResponse({"ok": False, "error": "No text."}, status=400)
	if not api_key:
		return JsonResponse({"ok": False, "error": "ELEVENLABS_API_KEY not set."}, status=500)

	try:
		import requests
		r = requests.post(
			f"{ELEVEN_BASE}/text-to-speech/{voice_id}",
			headers={"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
			json={"text": text, "model_id": "eleven_monolingual_v1",
				  "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
			timeout=30,
		)
		if r.status_code != 200:
			return JsonResponse({"ok": False, "error": f"ElevenLabs {r.status_code}"}, status=502)
		return HttpResponse(r.content, content_type="audio/mpeg")
	except Exception as exc:
		return JsonResponse({"ok": False, "error": str(exc)}, status=500)

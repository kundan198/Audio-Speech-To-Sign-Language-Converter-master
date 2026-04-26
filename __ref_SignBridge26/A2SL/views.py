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
	"""Keep Gemini tokens aligned with the exact clip filenames in assets/."""
	vocab = set(_sign_vocab())
	normalized = {word.lower(): word for word in vocab}
	out = []
	for word in words:
		asset = normalized.get(word.lower())
		if asset:
			out.append(asset)
		else:
			out.extend(c.upper() for c in word if c.isalnum() and c.upper() in vocab)
	return out


@login_required(login_url="login")
@require_http_methods(["POST"])
def translate_signs_view(request):
	"""JSON endpoint: text → sign word list (no page reload)."""
	try:
		payload = json.loads(request.body.decode('utf-8'))
	except (ValueError, UnicodeDecodeError) as exc:
		return JsonResponse({"ok": False, "error": str(exc)}, status=400)
	text = (payload.get('text') or '').strip()
	if not text:
		return JsonResponse({"ok": False, "error": "No text provided."}, status=400)
	words = _convert_sentence_with_gemini(text) or _convert_sentence_locally(text)
	return JsonResponse({"ok": True, "words": words, "text": text})


def _convert_sentence_with_gemini(sentence):
	api_key = os.environ.get("GEMINI_API_KEY", "").strip()
	if not api_key:
		return []

	try:
		import google.generativeai as genai
	except ImportError:
		return []

	prompt = f"""
You are a Sign Language simplifier.

Convert the sentence into a sign sequence using the allowed words where possible.

Allowed words:
{ALLOWED_SIGN_WORDS}

Rules:
- Output ONLY a valid JSON list of strings. No markdown, no explanation.
- Use exact allowed words where they fit the meaning.
- For any word NOT in the allowed list, spell it out as individual capital letters (e.g. "height" → "H","E","I","G","H","T").
- Use "ME" for I/me.
- Use "Cannot" for can't/cannot.
- Use "Do Not" for don't/do not.
- Use "Thank You" for thanks/thank you.
- Drop filler words (is, the, a, an, are, was, were) unless they carry meaning.
- Prefer short ASL-style order (topic first, question word last).

Sentence:
{sentence}
"""
	try:
		genai.configure(api_key=api_key)
		model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
		response = model.generate_content(prompt)
	except Exception:
		return []

	signs = [w.strip() for w in _extract_json_list(response.text) if w.strip()]
	return _clip_ready_words(signs)


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

	filtered_text = []
	for w in words:
		path = w + ".mp4"
		f = finders.find(path)
		#splitting the word if its animation is not present in database
		if not f:
			for c in w:
				filtered_text.append(c)
		#otherwise animation of word
		else:
			filtered_text.append(w)
	return filtered_text

@login_required(login_url="login")
def animation_view(request):
	if request.method == 'POST':
		text = request.POST.get('sen')
		words = _convert_sentence_with_gemini(text) or _convert_sentence_locally(text)


		return render(request,'animation.html',{'words':words,'text':text})
	else:
		return render(request,'animation.html')




def signup_view(request):
	if request.method == 'POST':
		form = UserCreationForm(request.POST)
		if form.is_valid():
			user = form.save()
			login(request,user)
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
			if 'next' in request.POST:
				return redirect(request.POST.get('next'))
			else:
				return redirect('animation')
	else:
		form = AuthenticationForm()
	return render(request,'login.html',{'form':form})


def logout_view(request):
	logout(request)
	return redirect("home")


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

	wlasl_result = _recognize_with_wlasl(frames, landmarks)
	if wlasl_result and wlasl_result.get("text") and wlasl_result.get("text") != "[unclear]":
		return JsonResponse({
			"ok": True,
			"text": wlasl_result["text"],
			"raw": wlasl_result["text"],
			"engine": "wlasl-i3d",
			"predictions": wlasl_result.get("predictions", []),
		})

	api_key = os.environ.get('GEMINI_API_KEY', '').strip()
	if not api_key:
		if wlasl_result:
			return JsonResponse({
				"ok": True,
				"text": wlasl_result.get("text", "[unclear]"),
				"raw": wlasl_result.get("text", "[unclear]"),
				"engine": "wlasl-i3d",
				"predictions": wlasl_result.get("predictions", []),
			})
		return JsonResponse({
			"ok": False,
			"error": "GEMINI_API_KEY environment variable is not set on the server.",
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
	"""Use Gemini to simplify text for a given mode (e.g. healthcare jargon
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

	api_key = os.environ.get('GEMINI_API_KEY', '').strip()
	if not api_key:
		# Soft fail: just return the original text so the UI keeps flowing.
		return JsonResponse({"ok": True, "text": text, "note": "GEMINI_API_KEY not set; passthrough."})

	try:
		import google.generativeai as genai
	except ImportError:
		return JsonResponse({"ok": True, "text": text, "note": "SDK missing; passthrough."})

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

	try:
		genai.configure(api_key=api_key)
		model = genai.GenerativeModel(os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash'))
		prompt = system + "\n\nSentence: " + text
		response = model.generate_content(prompt)
		out = (response.text or '').strip().strip('"').strip("'")
		out = out.splitlines()[0] if out else text
		return JsonResponse({"ok": True, "text": out})
	except Exception as exc:
		# Soft-fail to passthrough so the conversation keeps moving.
		return JsonResponse({"ok": True, "text": text, "note": f"Gemini error: {exc}"})


# -----------------------------------------------------------------------------
# Ultra-fast landmark-only hand classification (no image, text API only)
# -----------------------------------------------------------------------------

@login_required(login_url="login")
@require_http_methods(["POST"])
def classify_hand_view(request):
	"""Classify ASL sign from hand geometry features — no image, text-only Gemini call.

	Body:  { "features": "Thumb:ext, Index:up, Middle:up, Ring:down, Pinky:down | ...", "context": "..." }
	Response: { "ok": true, "text": "Hello" }
	"""
	try:
		payload = json.loads(request.body.decode('utf-8'))
	except (ValueError, UnicodeDecodeError) as exc:
		return JsonResponse({"ok": False, "error": str(exc)}, status=400)

	features = (payload.get('features') or '').strip()
	context  = (payload.get('context')  or '').strip()

	if not features:
		return JsonResponse({"ok": False, "error": "No features."}, status=400)

	api_key = os.environ.get('GEMINI_API_KEY', '').strip()
	if not api_key:
		return JsonResponse({"ok": False, "error": "GEMINI_API_KEY not set."}, status=500)

	try:
		import google.generativeai as genai
	except ImportError:
		return JsonResponse({"ok": False, "error": "google-generativeai not installed."}, status=500)

	ctx_hint = f' Context (prior words): "{context}".' if context else ''
	prompt = (
		f"ASL sign classifier.{ctx_hint}\n"
		f"Hand geometry: {features}\n"
		"(Format: finger-states | spread S/T | palm-fwd/back | tilt | fingertip-coords. "
		"Multiple frames separated by → show motion.)\n"
		"Identify the ASL letter or word. Reply with ONE word or letter only. "
		"No explanation. If unclear: unclear"
	)

	try:
		genai.configure(api_key=api_key)
		model = genai.GenerativeModel('gemini-2.0-flash-lite')
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
# Sign words -> natural English sentence (Gemini formulation)
# -----------------------------------------------------------------------------

@login_required(login_url="login")
@require_http_methods(["POST"])
def formulate_sentence_view(request):
	"""Take a list of raw sign-translated words and return a natural English sentence via Gemini.

	Body:  { "words": ["Hello", "My", "Name", "John"] }
	Response: { "ok": true, "sentence": "Hello, my name is John." }
	"""
	try:
		payload = json.loads(request.body.decode('utf-8'))
	except (ValueError, UnicodeDecodeError) as exc:
		return JsonResponse({"ok": False, "error": f"Invalid JSON: {exc}"}, status=400)

	words = [str(w).strip() for w in (payload.get('words') or []) if str(w).strip()]
	if not words:
		return JsonResponse({"ok": False, "error": "No words provided."}, status=400)

	raw_text = ' '.join(words)

	api_key = os.environ.get('GEMINI_API_KEY', '').strip()
	if not api_key:
		return JsonResponse({"ok": True, "sentence": raw_text, "note": "GEMINI_API_KEY not set; passthrough."})

	try:
		import google.generativeai as genai
	except ImportError:
		return JsonResponse({"ok": True, "sentence": raw_text, "note": "SDK missing; passthrough."})

	prompt = (
		"You are a sign language interpreter assistant. "
		"The user signed the following words in sequence (ASL/sign-language order):\n\n"
		f"Words: {words}\n\n"
		"Convert these words into one natural, grammatically correct English sentence. "
		"Fill in missing articles, prepositions, and verb forms to make it fluent. "
		"Keep the meaning. Output ONLY the finished sentence — no explanation, no quotes, no preamble."
	)

	try:
		genai.configure(api_key=api_key)
		model = genai.GenerativeModel(os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash'))
		response = model.generate_content(prompt)
		sentence = (response.text or '').strip().strip('"').strip("'")
		sentence = sentence.splitlines()[0].strip() if sentence else raw_text
		return JsonResponse({"ok": True, "sentence": sentence})
	except Exception as exc:
		return JsonResponse({"ok": True, "sentence": raw_text, "note": f"Gemini error: {exc}"})


# -----------------------------------------------------------------------------
# ElevenLabs voice synthesis (server-side proxy so the API key stays private)
# -----------------------------------------------------------------------------

ELEVEN_BASE = "https://api.elevenlabs.io/v1"
WLASL_DEFAULT_URL = "http://127.0.0.1:8766/predict"


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
			timeout=float(os.environ.get('WLASL_TIMEOUT', '5')),
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
	"""Return the list of voices available on the configured ElevenLabs account.

	Response: { ok: true, voices: [{voice_id, name, category}, ...] }
	"""
	api_key = os.environ.get('ELEVENLABS_API_KEY', '').strip()
	if not api_key:
		return JsonResponse({"ok": False, "error": "ELEVENLABS_API_KEY not set."}, status=500)
	try:
		import requests
	except ImportError:
		return JsonResponse({"ok": False, "error": "requests not installed."}, status=500)
	try:
		r = requests.get(
			f"{ELEVEN_BASE}/voices",
			headers={"xi-api-key": api_key},
			timeout=15,
		)
		if r.status_code != 200:
			return JsonResponse({"ok": False, "error": f"ElevenLabs {r.status_code}: {r.text[:200]}"}, status=502)
		data = r.json()
		voices = [
			{
				"voice_id": v.get("voice_id"),
				"name": v.get("name"),
				"category": v.get("category"),
				"labels": v.get("labels") or {},
			}
			for v in data.get("voices", [])
		]
		return JsonResponse({"ok": True, "voices": voices})
	except Exception as exc:
		return JsonResponse({"ok": False, "error": f"ElevenLabs request failed: {exc}"}, status=500)


@login_required(login_url="login")
@require_http_methods(["POST"])
def elevenlabs_tts_view(request):
	"""Synthesize speech with ElevenLabs and stream the MP3 back to the browser.

	Body:  { text: str, voice_id?: str, model?: str, stability?: float, similarity?: float, style?: float }
	Returns: audio/mpeg bytes (or JSON error).
	"""
	api_key = os.environ.get('ELEVENLABS_API_KEY', '').strip()
	if not api_key:
		return JsonResponse({"ok": False, "error": "ELEVENLABS_API_KEY not set."}, status=500)

	try:
		payload = json.loads(request.body.decode('utf-8'))
	except (ValueError, UnicodeDecodeError) as exc:
		return JsonResponse({"ok": False, "error": f"Invalid JSON: {exc}"}, status=400)

	text = (payload.get('text') or '').strip()
	if not text:
		return JsonResponse({"ok": False, "error": "No text."}, status=400)
	# Cap to keep latency low on stage.
	if len(text) > 800:
		text = text[:800]

	voice_id = (payload.get('voice_id') or os.environ.get('ELEVENLABS_DEFAULT_VOICE')
				or "21m00Tcm4TlvDq8ikWAM").strip()
	model = (payload.get('model') or "eleven_turbo_v2_5").strip()

	try:
		stability = float(payload.get('stability', 0.45))
		similarity = float(payload.get('similarity', 0.75))
		style = float(payload.get('style', 0.0))
	except (TypeError, ValueError):
		stability, similarity, style = 0.45, 0.75, 0.0

	try:
		import requests
	except ImportError:
		return JsonResponse({"ok": False, "error": "requests not installed."}, status=500)

	try:
		r = requests.post(
			f"{ELEVEN_BASE}/text-to-speech/{voice_id}",
			headers={
				"xi-api-key": api_key,
				"Content-Type": "application/json",
				"Accept": "audio/mpeg",
			},
			json={
				"text": text,
				"model_id": model,
				"voice_settings": {
					"stability": stability,
					"similarity_boost": similarity,
					"style": style,
					"use_speaker_boost": True,
				},
			},
			timeout=30,
		)
		if r.status_code != 200:
			return JsonResponse(
				{"ok": False, "error": f"ElevenLabs {r.status_code}: {r.text[:300]}"},
				status=502,
			)
		resp = HttpResponse(r.content, content_type="audio/mpeg")
		resp["Cache-Control"] = "no-store"
		return resp
	except Exception as exc:
		return JsonResponse({"ok": False, "error": f"TTS failed: {exc}"}, status=500)

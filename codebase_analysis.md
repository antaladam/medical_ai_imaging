# Codebase Analysis: Three Starter Projects

Analysis of the three tutorial repositories pulled into this folder from [Shubhamsaboo/awesome-llm-apps](https://github.com/Shubhamsaboo/awesome-llm-apps), read against the MSc thesis brief in `main.txt` ("A Multimodal, Explainable AI Decision-Support Prototype for Medical Image and Clinical-Text Analysis"). Every behavioral claim below is cited to the file (and line, where useful) that was actually read — nothing here is taken from README marketing copy without verifying it in code.

## Table of Contents

1. [Project 1 — AI Medical Imaging Agent](#project-1--ai-medical-imaging-agent)
2. [Project 2 — Vision RAG (Cohere Embed-4)](#project-2--vision-rag-cohere-embed-4)
3. [Project 3 — Multimodal Agentic RAG](#project-3--multimodal-agentic-rag)
4. [Cross-Project Comparison](#cross-project-comparison)
5. [Gaps This Thesis Needs to Fill](#gaps-this-thesis-needs-to-fill)

---

## Project 1 — AI Medical Imaging Agent

Path: `starter_ai_agents/ai_medical_imaging_agent/ai_medical_imaging.py`

### Pipeline / architecture

Single-file (158-line) Streamlit app, single-shot, no retrieval, no persistence. Full control flow, in order:

1. `st.session_state.GOOGLE_API_KEY` initialized to `None` on first run (`ai_medical_imaging.py:10-11`). Sidebar shows a password-type text input until a key is entered; on entry the key is stored in `session_state` and `st.rerun()` is called immediately (`:25-28`) — this forces the whole script to re-execute top-to-bottom (Streamlit's execution model: the entire file is a function body re-run on every interaction), which is why the `Agent` construction below is gated behind a session-state check rather than instantiated once. A "🔄 Reset API Key" button clears the key and reruns again (`:31-33`).
2. An `agno.agent.Agent` is constructed **on every script rerun** (not cached/memoized — no `@st.cache_resource`) with `model=Gemini(id="gemini-2.5-pro", api_key=...)` and `tools=[DuckDuckGoTools()]`, `markdown=True` (`ai_medical_imaging.py:45-52`), but only if a key is present (`if st.session_state.GOOGLE_API_KEY else None`) — so a new `Agent`/model client object is built on literally every button click or rerun, not reused across requests. Note: the README claims "Gemini 2.0 Flash" (`README.md:3,58`) but the code actually instantiates `gemini-2.5-pro` — a **documentation/code mismatch**, confirmed by reading the source directly.
3. The 5-section diagnostic prompt (`:58-93`) is a plain Python string built unconditionally on every rerun, regardless of whether a file was uploaded — cheap, but confirms there is no prompt templating/versioning system, just one hardcoded literal.
4. UI layout: three `st.container()`s (`upload_container`, `image_container`, `analysis_container`, `:99-102`) that are populated conditionally rather than a linear script — `image_container` and `analysis_container` only render content once `uploaded_file is not None`.
5. `st.file_uploader(..., type=["jpg","jpeg","png","dicom"])` (`:105-109`). **Verified correction to the prior pass**: `PILImage.open(uploaded_file)` is called at line 115, *directly inside the `if uploaded_file is not None:` block and outside any `try/except`* — the `try/except Exception as e: st.error(...)` only wraps the later analyze-button block (`:137-155`). This means a real `.dicom` file (or any file PIL can't decode) crashes with an **unhandled exception and Streamlit's default error screen**, not a graceful `st.error` message — the earlier draft's claim that this "would raise inside the try/except" was imprecise and is corrected here.
6. Image is displayed centered via a 3-column layout (`st.columns([1,2,1])`, image in the middle column, `:112-126`), resized to 500px width preserving aspect ratio (`:116-120`) purely for display — this resized copy, not the original upload, is what's actually analyzed.
7. On "🔍 Analyze Image" click: inside `st.spinner`, the resized image is saved to a **hardcoded relative path `temp_resized_image.png`** in the process's working directory (`:138-139`) — no per-session UUID, no temp-directory isolation. Concurrent users on a shared server instance (or concurrent Streamlit sessions) overwrite the same file, a race condition and privacy hazard confirmed by reading the exact save call.
8. Wrapped in `agno.media.Image(filepath=temp_path)` and passed to `medical_agent.run(query, images=[agno_image])` (`:142-145`) — note `medical_agent` was already constructed as a session-scoped Python variable from step 2 (not re-fetched here), and `.run()` is synchronous/blocking (no streaming to the UI; the whole 5-section response arrives in one shot after the spinner).
9. Response rendered directly as markdown via `st.markdown(response.content)`, no post-processing, no parsing of the 5 sections into separate UI elements — it is one contiguous markdown blob (`:146-149`).
10. The `except Exception as e: st.error(f"Analysis error: {e}")` (`:154-155`) catches failures only from the save/agent-run/render step — API-key errors, network errors, or agent-tool failures during `.run()` all funnel into this one generic message with no differentiation of failure type.

### Model(s) used

- **Gemini 2.5 Pro** (per code, not per README) via `agno.models.google.Gemini`, invoked through the `agno` agent framework's `.run()` method.
- **DuckDuckGo search** as an agent tool (`agno.tools.duckduckgo.DuckDuckGoTools`) — the model can call this itself during generation (agentic tool-use, not classic RAG retrieval).

### Prompt (quoted from source, `ai_medical_imaging.py:58-93`)

A single large system/user prompt instructs the model to act as a "highly skilled medical imaging expert" and structure output into five fixed sections: **Image Type & Region**, **Key Findings**, **Diagnostic Assessment** (explicitly asks for "confidence level" and "differential diagnoses in order of likelihood"), **Patient-Friendly Explanation**, and **Research Context** (explicitly instructs the model to invoke the DuckDuckGo tool for literature/treatment-protocol links). This is the only prompt in the whole app — there is no system prompt separate from this user-turn instruction block.

### Inputs / outputs

- Input: one image file (no clinical text/context field exists anywhere in the UI).
- Output: single free-text markdown blob following the 5-section structure above. No structured/JSON output, no machine-parseable fields, no per-claim citations back to the image (the "confidence level" and web references are free text generated by the LLM, not computed or retrieved objects).

### Retrieval / grounding mechanism

None for the image itself. The only "retrieval" is the DuckDuckGo tool call the agent may issue during generation for literature references — this is un-grounded in the sense that there's no vetted corpus; it's live web search, and the returned links are not verified against the model's diagnostic claims (no traceability from a specific "Key Finding" to a specific search result).

### Explainability elements

The "Diagnostic Assessment" section asks the model to self-report a confidence level and ranked differentials, but this is **an unconstrained free-text claim from the LLM**, not a calibrated probability, not backed by any retrieval, and not validated against ground truth. There's no attention map, no saliency visualization, no logit/token-probability signal exposed — it's a single opaque generation call.

### Limitations / gaps in code

- **Corrected**: no error handling for actual DICOM files — the crash happens *before* the app's only `try/except`, at `PILImage.open(uploaded_file)` (`:115`), immediately on upload, not during analysis. This surfaces as Streamlit's raw traceback screen, not the app's own `Analysis error:` message.
- No de-identification / PHI handling anywhere — raw uploaded image is written to disk unencrypted at a shared filename (`temp_resized_image.png`, process-working-directory-relative, no per-session isolation).
- No calibration: "confidence level" is prose, not a number derived from model internals or an ensemble.
- No evaluation harness, no logging of predictions vs. outcomes, no versioning of prompts.
- API key is entered client-side into `st.session_state` in plaintext and held in memory for the session — acceptable for a demo, not for clinical data.
- `Agent` object (and its model client) is reconstructed from scratch on every Streamlit rerun (no `st.cache_resource`) — functionally harmless here (stateless single call) but means there is no connection/session reuse, and would need addressing if the thesis extension adds any stateful component (e.g., conversation memory, batched evaluation runs).
- Single point-in-time inference; no comparison mode (there is no "image-only vs image+text" toggle — this project *is* the image-only arm by construction, since it has no clinical-text input field at all).
- No timeout/retry around the `.run()` call — a slow or hung Gemini/DuckDuckGo response blocks the Streamlit session indefinitely under the spinner.

### Dependencies (`requirements.txt`)

`streamlit` (UI), `agno` (agent orchestration/tool-calling framework), `Pillow` (image I/O/resize), `duckduckgo-search` (web search tool backing the agent), `google-generativeai` (Gemini SDK).

---

## Project 2 — Vision RAG (Cohere Embed-4)

Path: `rag_tutorials/vision_rag/vision_rag.py`

### Pipeline / architecture

554-line Streamlit app implementing genuine two-stage RAG over **images as the retrieval unit** (not text extracted from images).

1. Sidebar collects two API keys (Cohere + Google) into plain (non-session-persisted beyond the widget's own `key=`) locals `cohere_api_key`/`google_api_key` (`vision_rag.py:19-30`); clients (`cohere.ClientV2`, `genai.Client`) are (re)constructed **on every script rerun** once both keys are present (`:43-54`), each wrapped in its own try/except that surfaces init failures via `st.sidebar.error` — no caching of the clients themselves (unlike the embedding calls, see below).
2. **Ingestion**: user either loads 6 hardcoded sample finance-chart images downloaded from Substack-hosted URLs (`tesla.png`, `netflix.png`, `nike.png`, `google.png`, `accenture.png`, `tecent.png` — exact dict at `vision_rag.py:224-231`) or uploads their own PNG/JPG/PDF (`vision_rag.py:411-480`).
   - Images: opened via `PIL.Image.open`, resized in-place if over `max_pixels = 1568*1568` (`resize_image`, `:83-92`, preserves aspect ratio via `.thumbnail`), converted to a base64 data-URI string (`base64_from_image`, `:95-107`), embedded whole via Cohere `embed-v4.0` with `input_type="search_document"` (`compute_image_embedding`, `:128-145`).
   - **`compute_image_embedding` and `download_and_embed_sample_images` are both wrapped in `@st.cache_data(ttl=3600, show_spinner=False)`** (`:127`, `:220`) — a caching layer not identified in the prior pass. The Cohere client parameter is named `_cohere_client` (leading underscore) specifically so Streamlit's cache-key hasher skips it (Streamlit's documented convention for unhashable objects); the cache key is effectively the base64 string itself, so re-embedding the identical image content within the same hour is free, but this also means a stale/wrong cached embedding would silently persist for up to an hour if the underlying model or account changed.
   - PDFs: opened with PyMuPDF (`fitz`) from the uploaded byte stream, each **page rendered as a raster PNG at 150 dpi** via `page.get_pixmap(dpi=150)` → `Image.frombytes("RGB", ...)`, saved to `pdf_pages/<pdf_stem>/page_N.png`, then embedded the same way as a standalone image — no text extraction/OCR occurs anywhere (`process_pdf_file`, `:150-217`). This is a deliberate "no-OCR" design choice, stated in the README (`README.md:13`) and confirmed structurally in code. Per-page embedding failures are tolerated: a `None` placeholder is appended and filtered out afterward (`:192-213`), so a partially-failed PDF still contributes its successfully embedded pages rather than aborting entirely.
   - Directory layout actually created on disk: `img/` (samples), `pdf_pages/<pdf-filename-without-extension>/page_<N>.png` (PDF pages), `uploaded_img/<original-filename>` (direct image uploads) — all via `os.makedirs(..., exist_ok=True)`, never cleaned up, so repeated runs accumulate files indefinitely across sessions (confirmed: no `shutil.rmtree`/cleanup call anywhere in the file).
   - Re-run/re-upload dedup: there are actually two separate, differently-scoped checks. **Corrected citation**: the sample-loader's session-level dedup is `current_paths = set(st.session_state.image_paths); new_paths = [p for p in sample_img_paths if p not in current_paths]` at the call site (`:385-386`), not at `:246` — line `246` (`if img_path not in img_paths:`) is a *different*, function-internal dedup inside `download_and_embed_sample_images` against its own local list, not against `st.session_state.image_paths`. The uploader's check against `st.session_state.image_paths` is correctly at `:431`. Net effect is the same either way: re-clicking "Load Sample Images" or re-selecting the same file within a session is a no-op (`st.info("...already loaded/processed...")`, `:399`, `:480`) — but this is a **path-string check, not a content hash**, so uploading a renamed duplicate file creates a second, redundant embedding.
   - Embeddings and file paths accumulate in `st.session_state.image_paths` / `st.session_state.doc_embeddings` (initialized at `:38-41`, appended via `np.vstack` at `:396`, `:471-475`) — in-memory (session-scoped) only; the *embeddings* are lost on session end, though the *image/page files on disk* are not.
3. **Retrieval**: user's text question is embedded with the *same* `embed-v4.0` model but `input_type="search_query"` (`:316-321`); cosine similarity (`np.dot` — valid as true cosine similarity because Cohere's `embed-v4.0` returns unit-normalized vectors, though the code never explicitly re-normalizes or asserts this) is computed against every stored image embedding; `np.argmax` picks the single top-1 image (`search()`, `:305-346`). **Top-1 only — no top-k, no re-ranking, no multi-image fusion**, and no similarity threshold: `search()` always returns *some* image even if the best score is low (`:338-339`).
4. **Generation**: retrieved image (re-opened fresh from disk via `PIL.Image.open(img_path)`, `:358`) + original question text sent together as a two-part multimodal `contents` list to Gemini (`answer()`, `:349-375`) — README says `gemini-2.5-flash-preview-04-17` (`README.md:66`), code calls `model="gemini-2.5-flash"` (`:366`) — another README/code drift, though less severe (likely just an alias/version-family match rather than a wrong model class, unlike Project 1's drift).
5. Answer displayed as raw text under the retrieved image thumbnail with a caption naming its source file/PDF page (constructed by string-splitting the path on `os.sep` for PDF pages to recover `<pdf_name>.pdf, page_N`, `:530-544`).
6. The "Run Vision RAG" button itself is `disabled=` by a single boolean expression requiring both API keys, a non-empty question, at least one loaded image, and a non-empty embeddings array all simultaneously (`:512-513`) — a client-side precondition gate, not a server-side validation layer.

### Prompt (quoted, `vision_rag.py:359-363`)

```
Answer the question based on the following image. Be as elaborate as possible giving extra relevant information.
Don't use markdown formatting in the response.
Please provide enough context for your answer.

Question: {question}
```

Sent as a two-part content list `[prompt_text, PIL_image]` to `gemini_client.models.generate_content`.

### Inputs / outputs

- Input: text question + a pool of images/PDF pages (no clinical framing — this is a generic finance-chart demo repurposed as a RAG *pattern* reference, not medical).
- Output: (a) the single retrieved image shown to the user with a source caption (`vision_rag.py:530-539` — this **is** a traceability/citation mechanism, unlike Project 1), and (b) free-text answer string with no further structure, no confidence score.

### Retrieval / grounding mechanism

This is the clearest grounding example of the three: the answer is explicitly conditioned on one retrieved image, and that image is shown to the user alongside the answer with its filename/page number as provenance (`vision_rag.py:531-538`). However: (1) retrieval is top-1 hard-cutoff — if the correct evidence is spread across two images/pages, it's simply not retrieved; (2) there is no check that the generated answer's claims are actually supported by the retrieved image content (no answer-grounding verification step, no citation of *which part* of the image supports which claim); (3) similarity score itself is computed but never surfaced to the user as a confidence/reliability signal (`cos_sim_scores` at `vision_rag.py:335` is used only for `argmax`, never displayed).

### Explainability elements

Provenance via "Source: filename/page" caption is the one concrete explainability primitive present. No confidence score shown, no reasoning trace, no alternative-candidate display (user never sees the 2nd/3rd most similar image nor its score, so cannot judge retrieval quality).

### Limitations / gaps in code

- Top-1 retrieval only; no top-k fusion, no re-ranking, no threshold below which the system should say "no relevant evidence found" (`vision_rag.py:337-339` always returns *something*, even if the best cosine similarity is low).
- No calibration between retrieval similarity and answer reliability — and note the similarity score is computed (`cos_sim_scores`, `:335`) but genuinely never displayed anywhere in the UI, confirmed by grepping the file for every `st.` call after line 335; the value only feeds `np.argmax`.
- No de-identification handling (not medical, but would need to be added).
- PDFs/images are written to local disk folders (`pdf_pages/<name>/page_N.png`, `uploaded_img/<filename>`, `img/<sample>.png`) with no cleanup — files persist across sessions indefinitely; confirmed no deletion call anywhere in the module.
- Error handling exists per-function (try/except around embedding/search/answer calls) but failures degrade to `st.error`/`st.warning` text, not exceptions the wider system could act on, and there's no retry/backoff.
- The 1-hour `@st.cache_data` TTL on embeddings (`:127`, `:220`) is invisible to the user — there's no cache-invalidation control in the UI, so a changed Cohere account/model behind the same API key would silently keep returning cached vectors for up to an hour.
- Re-upload dedup is by file path string only, not content hash — same image under two filenames double-embeds and pollutes the retrieval pool with a duplicate.
- Sample dataset is finance infographics, not medical images — the domain has to be re-targeted entirely for this thesis; the *mechanism* (image-embedding retrieval + VLM answer) is the reusable part.

### Dependencies (`requirements.txt`)

`cohere` (Embed-4 multimodal embedding SDK), `google-generativeai` (Gemini SDK, note: code actually imports the newer `google.genai` client, not `google.generativeai` — another minor requirements/code drift), `PyMuPDF` (`fitz`, PDF→image rendering), `Pillow`, `numpy` (cosine similarity), `requests`/`tqdm` (sample-image download), `streamlit`.

---

## Project 3 — Multimodal Agentic RAG

Path: `rag_tutorials/multimodal_agentic_rag/` — full-stack app: FastAPI backend (`backend/server.py` 251 lines, `backend/rag_store.py` 508 lines, `backend/agentic_rag_agent/agent.py` 44 lines + empty `__init__.py`, `backend/app_state.py` 4 lines) + React/Vite/Three.js frontend (`frontend/src/App.tsx` 782 lines, plus `main.tsx`, `styles.css`, `vite.config.ts`, `package.json`). Every backend and frontend source file was read in full for this analysis. This is by far the most architecturally complete of the three and closest in spirit to what the thesis needs to extend.

### Pipeline / architecture

#### Backend routes in detail (`server.py`, 251 lines total)

- `GET /health` (`:139-146`) — returns `{status, adk, setup_error, ...RAG_STORE.space_tool()}`. `status` is `"ok"` only if `ADK_AVAILABLE` (import of `google.adk` succeeded *and* `GOOGLE_API_KEY` env var is set, `:24,32-33`) and `SETUP_ERROR` is empty; otherwise `"setup_required"`. `ADK_AVAILABLE`/`SETUP_ERROR` are computed once at **module import time**, so a key added to the environment after the process starts is never picked up without a restart.
- `GET /space` (`:149-151`) — returns `RAG_STORE.snapshot()` directly: all sources, all PCA-projected points, the (up to 80-entry) event log, provider/dimensions/model metadata.
- `POST /sources/text` (`:154-161`) — Pydantic `TextSourceRequest{title, text, modality:"text"}`; calls `RAG_STORE.add_text_source`; any exception is converted to `HTTPException(400, str(exc))`.
- `POST /sources/url` (`:164-178`) — Pydantic `UrlSourceRequest{url: HttpUrl, title?}` (Pydantic's `HttpUrl` type itself rejects malformed URLs before the handler body runs); calls `_validate_fetch_url` (SSRF guard, see below) in a threadpool, then an async `httpx.AsyncClient` GET with `timeout=20`, `follow_redirects=True` — note **redirects are followed without re-validating the final destination against `_validate_fetch_url`**, so a public URL that 302-redirects to a private/internal address would bypass the SSRF check (the guard only inspects the originally submitted hostname). HTML is stripped via `_extract_text_from_html` (BeautifulSoup, strips `script/style/nav/footer/header` tags, `:72-76`) and the result is hard-truncated to the first 12,000 characters (`text[:12000]`, `:174`) before being stored as a text source (so it goes through the same `_chunk_text`/170-word chunking as manually typed text).
- `POST /sources/file` (`:181-201`) — `UploadFile` + `Form` fields `title`/`notes`; reads the full upload into memory (`await file.read()`) and rejects anything over **120 MB** (`:188-189`) before calling `RAG_STORE.add_file_source`.
- `DELETE /sources/{source_id}` (`:204-209`) — **not covered in the prior pass**: calls `RAG_STORE.remove_source`, returns `404` if the id doesn't exist, otherwise `{deleted, space}`. Deletion is real (removes the source and all its chunks from the in-memory lists, see `rag_store.py:372-381`), not a soft-delete/tombstone.
- `POST /ask` (`:212-243`) — see Generation below.
- CORS: `CORSMiddleware` with `allow_origins` from `ALLOWED_ORIGINS` env var, defaulting to `["http://localhost:5177", "http://127.0.0.1:5177"]` (matching the Vite dev server port, confirmed in `frontend/vite.config.ts:7`), `allow_credentials=False`, `allow_methods=["*"]`, `allow_headers=["*"]` (`:40-51`).
- No authentication dependency on any route, confirmed by reading every `@app.*` decorator — none references a security scheme.

#### `app_state.py` (4 lines, full file)

A single module-level singleton: `RAG_STORE = MultimodalRagStore()`, constructed once at import time with the default `dimensions=768`. This is the entire "global state" pattern — no dependency-injection container, no per-request store, so `server.py` and `agentic_rag_agent/agent.py` both import and share the exact same Python object (confirmed both files do `from app_state import RAG_STORE`).

**Thread-safety note** (previously missing from this analysis): `MultimodalRagStore.__init__` creates `self._lock = threading.RLock()` (`rag_store.py:126`), and every mutating or list-scanning method (`add_text_source`, `add_file_source`, `remove_source`, `search`, `snapshot`, `space_tool`) wraps its body in `with self._lock:`. Since FastAPI route handlers offload these calls via `run_in_threadpool` (`server.py`, every route), concurrent requests genuinely can execute them on different worker threads simultaneously against the one shared `RAG_STORE` singleton — the `RLock` is a real, load-bearing safeguard against concurrent list mutation (e.g. two uploads racing on `self.sources.append(...)`), not a vestigial import. It does not, however, make an individual multi-step operation atomic with respect to a *read* that happens outside the lock (e.g. the `retrieval_payload`/`snapshot` values returned from `/ask` are correct as of the moment they were computed, but nothing prevents a concurrent `DELETE /sources/{id}` from invalidating a citation the client is about to render).

1. **Ingestion**:
   - Text: chunked at up to 170 words per chunk with a 35-word overlap step (`_chunk_text`, `rag_store.py:73-86` — precisely: if the source is ≤170 words it's a single chunk; otherwise it slides a window with `step = CHUNK_WORDS - CHUNK_OVERLAP = 135` words, and any trailing fragment under 25 words is dropped rather than kept as a tiny chunk, `:84`), each chunk embedded separately via Gemini Embedding 2 (`_embed_text`, `:138-147`) with task-prefixed content (literally `"task: retrieval document: {chunk text}"`, `:139` combined with the call at `:314`).
   - URL: see `POST /sources/url` above. **SSRF protection is implemented**: `_validate_fetch_url` (`:79-98`) blocks localhost/private/loopback/link-local/reserved IPs via `socket.getaddrinfo` + `ipaddress` checks on the *submitted* hostname, overridable only via `ALLOW_PRIVATE_URLS` env var — a concrete, verifiable safety control absent from the other two projects, with the caveat above about unvalidated redirects.
   - Files (image/audio/video/PDF): embedded **natively multimodally**, and unlike text sources, **a file source is always exactly one chunk** regardless of file size or type (`add_file_source`, `chunks=1` at `:344`) — confirmed by reading the dataclass construction, this applies uniformly to a one-page image and a multi-hundred-page PDF alike, since Gemini Embedding 2 embeds the whole file as one vector. This is a structural difference from Project 2, which instead splits a PDF into one embedded raster image *per page*. Small files (≤18 MB, `INLINE_MEDIA_LIMIT_BYTES`, `:20`) that are not audio/video go inline as `types.Part.from_bytes` directly to Gemini Embedding 2 (`_embed_file`, `:189-211`); files over 18 MB, or any audio/video regardless of size, go through the Gemini File API (upload → poll for `ACTIVE`/`SUCCEEDED` state every 2s up to a 90s cap → embed by file URI → delete the uploaded file in a `finally` block, logging but not raising on cleanup failure) (`_embed_uploaded_file`, `:149-187`). There is also a **fallback path** not noted previously: if an inline embed attempt raises and the mime type is video/audio/pdf, `_embed_file` retries via the File API instead of failing outright (`:208-211`) — a small but real resilience feature.
   - Each file source's embedding is **blended**: raw media embedding × 0.68 + a text-embedded title/notes annotation × 0.32, then renormalized (`_blend_vectors`, `:89-96`, called at `:338`). The 0.32 weight is a bare literal default-argument value with no comment or derivation shown — a good ablation candidate for the thesis.
2. **Retrieval** (`MultimodalRagStore.search`, `:383-425`): query embedded with `"task: question answering | query"` prefix; cosine similarity (`_cosine`, pure-Python dot/norm, `:59-65`) computed chunk-by-chunk against every stored chunk; **best chunk per source** is kept (not per chunk) and the top `top_k` (default 6, `AskRequest.top_k` accepts 1–12 at the Pydantic/API level, `server.py:69` — **but see the frontend walkthrough below: the shipped UI never actually exposes this**) *sources* are returned as matches, each carrying its similarity score and the actual matched chunk text as "evidence" (`retrieval_payload`, `:452-465`). Every `search()` call also recomputes the full 3D PCA projection over the current source set plus the new query point (`:387-388`) — see the PCA walkthrough below for why this matters.
3. **Generation**: `/ask` (`server.py:212-243`) runs retrieval **once** via `RAG_STORE.search`, converts it to a `retrieval_payload` dict, and passes that *exact same object* into the ADK agent flow as a closed-over tool result (`retrieve_relevant_context` closure defined inside `_run_adk_agent`, `server.py:116-118` — this closure ignores its own `query`/`top_k` parameters and just returns the pre-computed `retrieval` dict). **Correction/addition to the prior pass**: `agentic_rag_agent/agent.py` also defines a *second*, independent implementation of the same-named tool at module level (`retrieve_relevant_context`, `agent.py:7-9`), which calls `RAG_STORE.retrieval_tool(query=query, top_k=top_k)` for real, using whatever arguments the LLM's tool call actually supplies — this is the function bound to `root_agent = build_agent()` at the bottom of `agent.py:43` (i.e., what you'd get running the agent standalone via the ADK CLI/`adk web`). `server.py` never uses this default — it always calls `build_agent(retrieve_relevant_context)` with *its own* closure that overrides the tool (`server.py:120`). So the deployed FastAPI backend deliberately trades away genuine agent-driven retrieval for a determinism guarantee: whatever the agent's tool call returns is *guaranteed* to be byte-identical to what the citations panel shows, because it's the same Python object, not a second LLM-triggered lookup that could drift. This is a real, deliberate design tradeoff, not simply an oversight — worth stating precisely in the thesis rather than just calling it a limitation.
   - The Google ADK `Agent` (`agentic_rag_agent/agent.py:17-40`) runs model `gemini-3-flash-preview`, temperature 0.25, max 900 output tokens, tools `[inspect_embedding_space, retrieve_relevant_context]`.
   - Session handling: `server.py:121-122` creates a **brand-new ADK session** (`session_service.create_session(...)`) on every single `/ask` call, via a process-wide `InMemorySessionService()` instantiated once at module load (`server.py:53`). There is no session-id reuse across requests anywhere in `server.py` or `App.tsx` (the frontend never stores or sends a session id). **Confirmed: each `/ask` is fully stateless conversationally** — the agent never sees prior questions or its own prior answers; only the freshly retrieved evidence and the current question are in its context (`server.py:123-126`).
4. **Response**: JSON containing `answer` (free text, post-processed by the ADK event loop which keeps only the *last* non-empty text event, `_event_text`/`_run_adk_agent`, `server.py:101-136`), `matches` (structured: citation id, source title, modality, similarity score, evidence snippet), `query_point` (3D-projected query embedding), a synthetic `trace` array describing 3 pipeline stages for UI display (`server.py:220-236` — a cosmetic 3-line summary; **distinct from** `RAG_STORE.events`, the real timestamped event log capped at 80 entries and covering `source_added`/`source_removed`/`query_embedded`/`file_cleanup_failed`, which is only exposed via `GET /space`/`GET /health`, not via `/ask`'s `trace` field), and a full embedding-space `snapshot`.

#### Frontend component walkthrough (`frontend/src/App.tsx`, 782 lines, React 19 + Vite 7 + Three.js 0.181, no router/state library — `package.json`)

- Single-file, single-component app (`export default function App()`, `:430`) plus three local helper components: `AnswerContent` (answer-text formatter), `SourceRow` (sidebar list item), `VectorSpace` (the Three.js canvas). No routing; one page.
- `API` base URL comes from `import.meta.env.VITE_API_URL`, falling back to `http://localhost:8897` (`:24`) — a plain `fetch`-based client, no generated SDK, no request retry/backoff, no auth header ever attached to any request (consistent with the backend having no auth).
- State (`useState`, `:431-451`): `space` (the whole `SpaceSnapshot` from the backend), the "add source" form fields (`tab`, `title`, `text`, `url`, `notes`, `selectedFile`), the Q&A fields (`question`, `answer`, `matches`, `trace`, `queryPoint`, `selectedPoint`), and several busy/status/error flags per action (`isAddingSource`, `isAsking`, `removingSourceId`, `sourceStatus/Error`, `qaStatus/Error`).
- **Data freshness model**: `refreshSpace()` (`:459-462`) does a `GET /space` fetch **only once, on mount** (`useEffect(..., [])`, `:464-466`). After that, `space` is updated exclusively as a side effect of whatever the *last local action* returned (`setSpace(data.space)` inside `addSource`, `removeSource`, and `askQuestion`) — there is no polling and no websocket, so two browser tabs (or a second user) editing the same backend `RAG_STORE` will silently diverge until a manual action triggers a fresh fetch. The only other way to force a refresh is the small refresh icon button next to "Add source" (`:603-605`), which just re-runs `refreshSpace()`.
- **Add-source flow**: three tabs — Text / URL / File (`:608-654`) — sharing one `title` field. Defaults are pre-filled with a self-referential demo: sample `text` describing Gemini Embedding 2 itself (`sampleText`, `:87`), a default `url` pointing at Google's own "Building with Gemini Embedding 2" blog post (`:435`), and a default `question` asking how Gemini Embedding 2 helps agentic RAG (`:437`) — i.e., out of the box the app is a demo *about its own embedding model*, not about any external domain (confirms: this project ships with zero medical framing, same as Project 2's finance charts). File tab accepts `.txt,.md,.pdf,image/*,audio/*,video/*` at the `<input type="file">` level (`:637`) — a client-side filter only; the backend's `add_file_source` will attempt to embed whatever mime type actually arrives. `addSource()` (`:468-508`) branches on `tab` to hit `/sources/text`, `/sources/url`, or `/sources/file` (the last via `FormData`, not JSON) and on success replaces the entire `space` state with the response's fresh snapshot.
- **Remove-source flow**: `removeSource()` (`:510-530`) gates on a blocking native `window.confirm()` dialog, then `DELETE /sources/{id}`; on success it also locally filters `matches` to drop any citation pointing at the deleted source. **Correction after re-verifying against source**: `selectedPoint` clearing *is* conditional on it referencing the deleted source (`if (selectedPoint?.source_id === source.id) setSelectedPoint(null)`, `:522`), but `queryPoint` clearing is **not** — `if (queryPoint) setQueryPoint(null)` (`:523`) wipes any existing query point on *every* source removal, regardless of whether that query point referenced the removed source. So the state reconciliation here is asymmetric: precise for citations/selectedPoint, blunt for queryPoint.
- **Ask flow**: `askQuestion()` (`:532-557`) POSTs `{question, top_k: 6}` to `/ask` — **`top_k` is a hardcoded literal `6` in the fetch body**; despite the backend's `AskRequest.top_k` accepting any value 1–12, no UI control (slider, input, dropdown) exists anywhere in `App.tsx` to change it. This corrects the prior pass's "UI-configurable 1–12" claim: that range is only reachable by calling the API directly (e.g. via curl/Postman), not through the shipped frontend. On success it replaces `answer`, `matches`, `trace`, `queryPoint`, and the entire `space` snapshot in one go.
- **Answer rendering** (`AnswerContent`/`cleanAnswerText`, `:109-165`): before display, the raw answer string is passed through a regex sanitizer that strips citation-id-shaped bracket tokens (`/\[[a-f0-9]{8,12}-\d+\]/gi`), markdown bold markers, and asterisk-bullet prefixes (`cleanAnswerText`, `:109-117`) — a defensive client-side backstop in case the LLM doesn't perfectly follow agent-instruction point 4 ("do not include raw citation ids... the UI shows citations separately"). The cleaned text is then heuristically split into paragraphs/lists/headings by blank-line and prefix pattern-matching (no markdown parser library used) purely for visual structure — this is presentation logic only, it does not re-derive or validate any claim against evidence.
- **Citations panel** (`:749-777`): renders each `match` with an icon, title, a rounded percentage similarity score via `scorePct()` (`:89-91`, clamps to [0,1] then multiplies by 100), a horizontal score bar, and the raw evidence text snippet. Hovering a citation row calls `setSelectedPoint` to highlight the corresponding point in the 3D view (`:764-765`) — this is the one piece of genuine cross-linking between the citation list and the embedding-space visualization.
- **`VectorSpace` (3D view, `:187-398`)**: a fully manual Three.js scene built/torn down in a single `useEffect` keyed on `[points, queryPoint, matches, onSelect]` (`:208-395` — **corrected range**: `:204-206` is a separate, smaller `useEffect` keyed only on `[selectedId]` that just syncs a ref; the scene-building effect itself starts at `:208`) — i.e. **the entire WebGL scene is destroyed and rebuilt from scratch on every new answer or source change**, not incrementally updated. It renders: a fogged scene with a ground `GridHelper`, three colored axis lines, a starfield backdrop of 150 random points, one small sphere mesh per point in `space.points` (plus the transient `queryPoint` when present) positioned at `projection.{x,y,z} * 1.35`, and glow-sprite halos on the query point and any point matched by the current citations (`matchedIds`, `:266,274-291`). Each mesh gets a small continuous orbit/bob animation (`requestAnimationFrame` loop, `:342-374`) purely for visual liveliness — not encoding any data. Hover uses a `THREE.Raycaster` against the mesh list to detect the pointed-at point and calls `onSelect` (parent's `setSelectedPoint`), driving the "hover-card" info box (`:692-699`). Camera drifts slightly toward the pointer position (`:344-346`) but there is no drag-to-orbit/zoom control implemented — the "3D navigation" is limited to this pointer-parallax effect.
- **Important cross-file interaction finding** (backend + frontend read together): because `rag_store.py`'s `_pca_projection` (see below) is recomputed **from the full current point set** on every `search()` call, and `askQuestion()` unconditionally does `setSpace(data.space)` with the response's freshly recomputed projection, **every existing source's x/y/z position can shift after each new question is asked**, not just the transient query point's. The 3D "map" is therefore not a stable, comparable coordinate frame across turns — a researcher or clinician comparing "where did evidence X sit two questions ago" cannot do so reliably from this UI, since the axes themselves are redefined each time. This is a concrete interpretability caveat the thesis should be aware of if it reuses this visualization.

### Model(s) used

- **Gemini Embedding 2** (`gemini-embedding-2`, `rag_store.py:16`) for all embeddings — text, image, PDF, audio, video — with configurable output dimensionality (default 768, `rag_store.py:17`).
- **`gemini-3-flash-preview`** as the ADK answer-generation model (`agent.py:20`).

### Prompt / agent instruction (quoted, `agent.py:22-34`)

```
You are the Google ADK coordinator for a multimodal agentic RAG workspace.

For every user question:
1. Use inspect_embedding_space to understand the current workspace.
2. Use retrieve_relevant_context with the user's question before answering.
3. Ground the answer in the retrieved evidence. Do not invent facts that are not supported by the workspace.
4. Do not include raw citation ids, source ids, bracket citations, Markdown bold markers, or asterisk bullets in the answer. The UI shows citations separately.
5. Start with a clear direct answer in 2-3 sentences.
6. If helpful, add a short "Key points:" section with simple hyphen bullets.
7. Explain briefly when the vector evidence is weak or sparse.
8. Keep the answer useful and direct.
```

Point 3 and 7 are the closest thing to an explicit "explainability/calibration" instruction across all three projects — but note it is still a **prompted behavior**, not a measured/enforced one; nothing in `server.py` validates that the model actually flagged weak evidence when similarity scores were in fact low. The frontend's `cleanAnswerText` (`App.tsx:109-117`) exists specifically because point 4 of this instruction is only a prompt, not a hard constraint — the client-side regex strip of citation-id patterns and markdown bold/bullet markers is a defensive backstop for cases where the model doesn't fully comply.

#### PCA implementation walkthrough (`_pca_projection`, `rag_store.py:213-266`)

A from-scratch, non-sklearn power-iteration PCA, run fresh on every `snapshot()`/`search()` call:
1. Centers all input vectors (one per source, plus the query vector when present) by subtracting the per-dimension mean (`:222-223`).
2. For each of 3 target components: seeds a deterministic pseudo-random unit vector from a fixed `sin`/`cos` formula keyed only on dimension index and component index (`:227-231` — same seed every call, independent of the actual data), orthogonalizes it against previously found components (`_orthogonalize`, `:110-115`), then runs 24 iterations of power-method refinement (`candidate = normalize(orthogonalize(Σ score·row))`, `:234-243`) to converge it toward the dominant remaining eigenvector of the centered data.
3. Projects every point onto the 3 resulting components, then uniformly rescales all coordinates so the maximum point radius maps to a fixed constant `2.65` (`:254-258`) — this is purely a display-space normalization, not a statistically meaningful unit.
4. **No eigenvalues or variance-explained ratio is computed or exposed anywhere** — confirmed by reading the full method; there is no way for a user or the thesis's evaluation code to know what fraction of the 768-dimensional embedding's structure the 3 displayed axes actually capture.
5. Because step 1's centering and step 2's power iteration both depend on *which points are currently in the set* (all current sources, plus whichever query is active), and this whole function reruns from scratch on every `search()` call (`rag_store.py:387-388`), **the resulting 3D coordinate frame is not stable across queries** — see the frontend walkthrough above for the resulting UX consequence.

### Inputs / outputs

- Input: text, URL, PDF, image, audio, or video, added incrementally to a persistent (session-lifetime) index; then natural-language questions against the whole accumulated corpus.
- Output: structured JSON — free-text answer, a list of typed citation objects (id, source, modality, similarity score, evidence text), and a process trace. This is the only project with a **machine-readable, structured citation list** rather than prose-embedded or caption-only provenance.

### Retrieval / grounding mechanism

Most rigorous of the three: per-source best-chunk retrieval with explicit similarity scores exposed end-to-end to both the LLM (as tool input) and the UI (as citation metadata) (`retrieval_payload`, `rag_store.py:452-465`). The same retrieval object is guaranteed to be what both the citations panel and the answer are based on (README explicitly states this design goal, `README.md:29`, code-verified: `retrieval_payload` is computed once in `/ask` and threaded through unchanged to the ADK tool closure, `server.py:217-219`, and the *default* module-level tool in `agent.py:7-9` that would do genuine independent retrieval is never actually wired into the running server — see the Generation walkthrough above). Weakness: still top-k by cosine similarity only, no cross-encoder re-ranking, no hybrid lexical+vector search, and the "agent" doesn't actually decide *whether* to retrieve or *how much* — retrieval is unconditionally forced before every answer (`server.py:217`, always called regardless of the ADK instruction's step ordering). Text sources can contribute multiple candidate chunks per search (170-word windows); file sources (image/audio/video/PDF) structurally contribute exactly one chunk each (`rag_store.py:344`), so "best chunk per source" only has real selection power for text/URL sources — for a file source, there is only ever one candidate to pick.

### Explainability elements

- Structured citations with similarity scores (quantitative, inspectable, and actually rendered to the user as a percentage + bar in the citations panel, `App.tsx:769-771`) — strongest of the three projects.
- Agent explicitly instructed to flag weak/sparse evidence in the answer text (prompted, unverified).
- 3D PCA embedding-space visualization lets a user/researcher inspect *why* retrieval did or didn't work (nearest-neighbor structure visible), which is a genuine interpretability tool absent from Projects 1–2 — though PCA-into-3D from a 768-dim space is a lossy, custom (non-sklearn) projection with no reported explained-variance metric, and (newly verified) **the coordinate frame itself is unstable across queries** since it's recomputed from scratch each time over a data-dependent point set (see PCA walkthrough above) — this weakens its value as a *comparison* tool across multiple questions, even though it's a real aid for inspecting any single answer.
- A real, timestamped event log (`RAG_STORE.events`, capped at 80 entries, covering source add/remove and query-embedded events, exposed at `GET /space`/`GET /health`) exists and is more substantive than the cosmetic 3-line `trace` array `/ask` returns — but it still has no correctness/outcome field, so it's an audit trail of *actions*, not an evaluation log of *accuracy*.
- Still no calibration curve, no confidence interval, no agreement/failure-case metrics — this remains a single-pass generation with heuristic grounding aids, not a measured-uncertainty system.

### Limitations / gaps in code

- **In-memory only** — explicitly stated in README (`README.md:118`) and structurally true (`MultimodalRagStore.__init__`, `rag_store.py:118-127`, plain Python lists, no DB). Restart loses everything.
- No authentication on any endpoint (`server.py` has no auth dependency anywhere, and `App.tsx`'s `fetch` calls never attach any auth header, confirmed both sides) — acceptable for a local demo, disqualifying for real clinical data without additions.
- README's own "Notes" section explicitly says: "For production, replace the in-memory store with durable storage and add authentication, background ingestion, evals, observability, and a managed vector database" (`README.md:122`) — i.e., the authors themselves flag the exact gap category (evals/observability) this thesis must fill.
- No calibration or failure-case evaluation harness exists at all — the `/ask` `trace` field (`server.py:220-236`) is a cosmetic 3-line UI description; the more substantive `RAG_STORE.events` log (see Explainability above) is an action audit trail, not an accuracy/outcome log.
- Retrieval always top-k by cosine similarity per-source-best-chunk; no mechanism to detect and report "no relevant source found" other than the LLM's own prompted hedging.
- Blended media+annotation embedding (`_blend_vectors`, weight 0.32) is a fixed heuristic constant with no justification or tuning shown in code — a good candidate for ablation in the thesis (image-only embedding vs. blended embedding retrieval quality).
- **New**: `/sources/url`'s SSRF guard validates the submitted hostname but `httpx.AsyncClient(follow_redirects=True)` does not re-validate the post-redirect destination (`server.py:169-171`) — a narrow but real gap in an otherwise-careful control, worth noting if the thesis reuses URL ingestion for pulling in reference literature.
- **New**: ADK sessions are created fresh on every `/ask` call with no id ever persisted or reused by the frontend (`server.py:121-122`, `App.tsx` never stores a session id) — confirms there is no multi-turn conversational memory; each question is answered independently even if a clinician is following up on the previous answer.
- **New**: the shipped frontend hardcodes `top_k: 6` and exposes no control for it (`App.tsx:542`), even though the backend API supports 1–12 — the "top-k tunability" only exists at the API layer, not for an actual user of the app as built.
- **New**: the 3D embedding-space view's coordinate frame is recomputed from scratch (data-dependent, not just re-seeded) on every query, so source positions are not stable/comparable across questions (see PCA walkthrough) — a caveat for any thesis use of this visualization as a "before/after" grounding-quality comparison tool.
- Not medical-specific: general multimodal RAG demo (finance/generic assets in the architecture diagram; the shipped default content is literally about Gemini Embedding 2 itself), would need a medical-image + clinical-note corpus substituted in.

### Dependencies

**Backend** (`backend/requirements.txt`): `fastapi`/`uvicorn` (API server), `google-genai` (Gemini Embedding 2 + File API + generation client), `google-adk` (agent orchestration/tool-calling runner), `python-multipart` (file upload parsing), `beautifulsoup4` (URL text extraction), `httpx` (async HTTP fetch for URL ingestion).

**Frontend** (`frontend/package.json`, full dependency list): `react`/`react-dom` 19.2.x, `three` 0.181.x (WebGL/3D scene for the embedding-space view), `lucide-react` 0.561.x (icon set used throughout the UI), `vite` 7.2.x + `@vitejs/plugin-react` (dev server/bundler, dev server pinned to port 5177 in `vite.config.ts:7`, matching the backend's default CORS allowlist), TypeScript 5.9.x + `@types/*` as dev dependencies. No state-management library, no router, no CSS framework (plain `styles.css`), no HTTP client library beyond the native `fetch` API — a deliberately minimal-dependency single-page app.

---

## Cross-Project Comparison

| Aspect | Project 1: Medical Imaging Agent | Project 2: Vision RAG | Project 3: Multimodal Agentic RAG |
|---|---|---|---|
| Architecture style | Single-shot agentic call (agno + tool-use) | Two-stage RAG (embed → retrieve top-1 → generate) | Full-stack agentic RAG (FastAPI + ADK agent + React) |
| Modalities in | Image only | Image, PDF (rendered as images) | Text, URL, PDF, image, audio, video |
| Embedding model | None (no retrieval) | Cohere `embed-v4.0` | Gemini Embedding 2 |
| Generation model | Gemini 2.5 Pro (code) / "2.0 Flash" (README, stale) | Gemini 2.5 Flash | `gemini-3-flash-preview` via Google ADK |
| Retrieval breadth | N/A | Top-1 only, no re-ranking | API supports top-k 1–12, best-chunk-per-source; **shipped UI hardcodes top_k=6** with no control exposed |
| Grounding/citation | None (free web search only) | Retrieved image shown with filename/page caption | Structured citation objects w/ similarity scores, decoupled from answer text, rendered as % + bar in UI |
| Explainability signal | Self-reported free-text "confidence level" | Cosine similarity computed but not surfaced | Similarity scores surfaced + 3D embedding-space view (coordinate frame unstable across queries) + prompted hedging on weak evidence + real (but outcome-free) event log |
| Multi-turn memory | N/A (single shot) | N/A (single shot) | None — new ADK session created per `/ask` call, no session id persisted anywhere |
| Calibration / eval harness | None | None | None (explicitly flagged as future work by original authors) |
| Safety/privacy controls | None (shared temp file, no PHI handling) | None (persistent local file dump) | SSRF guard on URL ingestion (redirects not re-validated); still no auth, no de-identification |
| Comparison-arm fit | **Image-only arm** (no text input exists) | Image+text-query arm, but query is a *question*, not clinical context | Best substrate for **image+clinical-text arm** — can ingest a scan *and* structured clinical notes into one grounded index |

---

## Gaps This Thesis Needs to Fill

Synthesized across all three codebases, relative to the MSc-level contribution requirements in `main.txt` (measure image interpretation, retrieval grounding, explanation quality, calibration, failure cases; document data-protection/clinical-safety boundaries; compare image-only vs. image+text workflows):

1. **No calibration exists anywhere.** All three "confidence" signals are either absent (Project 1's free-text confidence is unconstrained LLM prose) or present-but-unused (Projects 2 and 3 compute cosine similarity but never validate it against ground truth or expose calibration curves/Brier scores). The thesis needs to add an actual calibration measurement layer — e.g., correlate retrieval similarity / self-reported confidence against clinician-labeled correctness.
2. **No failure-case or evaluation harness.** None of the three log predictions, retrieved evidence, or outcomes anywhere for later analysis. Project 3's `trace` object is cosmetic. A logging/eval pipeline (predictions + retrieved evidence + ground truth + human review) has to be built from scratch.
3. **No image-only vs. image+text comparison arm exists in any single project.** Project 1 is structurally image-only (no text input field). Project 3 is the only one capable of ingesting both an image and clinical text into the same grounded index, but nobody has wired up a controlled A/B comparison mode. The thesis needs to build this comparison explicitly, likely by extending Project 3's store/retrieval to run twice per query (image-only context vs. image+retrieved-text context) and diffing the two answers/citations.
4. **No de-identification or clinical-data-protection handling in any project.** All three either don't handle PHI at all (Projects 1–2 write raw files to shared/local disk with no encryption or access control) or explicitly punt on it (Project 3's README says production would need auth/durable storage added). This must be designed and documented as its own thesis section, not inherited from any starter repo.
5. **Grounding verifiability is uneven and, even in the best case (Project 3), unverified.** Project 3 is the only one with structured, similarity-scored citations decoupled from the answer, but nothing checks that the generated answer's claims are actually entailed by the cited evidence (no NLI/entailment check, no citation-accuracy metric). This is a concrete, implementable contribution: an automatic "is this claim supported by its cited evidence" verification step, feeding the "verifiability" half of the research question.
6. **Retrieval quality is shallow everywhere.** Top-1 (Project 2) or top-k cosine-only with a fixed, untuned blend weight (Project 3, `_blend_vectors` weight = 0.32 hardcoded) — no re-ranking, no hybrid search, no ablation of retrieval design choices. The thesis's "retrieval grounding" measurement axis should include an ablation over these choices (top-k value, blend weight, with/without re-ranking) rather than assuming the starter defaults are correct.
7. **Two README/code inconsistencies were found** (Project 1's stated vs. actual Gemini model version; Project 2's stated vs. actual model string and SDK), which is a reminder to verify all thesis-reported model versions against the actual invocation code, not the documentation, for reproducibility.
8. **No multi-turn conversational memory in Project 3.** A new ADK session is created on every `/ask` call (`server.py:121-122`) and no session id is ever persisted by the frontend. If the thesis's clinician-facing prototype needs iterative follow-up questions ("what about the second finding?"), this has to be built — the starter has no scaffolding for it at all.
9. **The one interpretability aid Project 3 does have (the 3D embedding-space view) is not stable across turns.** Because its PCA is recomputed from the full current point set on every query, source positions shift between questions (see the PCA walkthrough in Project 3's section). If the thesis wants to use an embedding-space visualization to let a clinician compare *how* two different questions retrieved evidence, this instability needs to be fixed (e.g., by fitting the projection once and reusing fixed axes, rather than refitting per query) — otherwise the visualization actively misleads rather than aids comparison.

# Chat UI — Parallax Frontend

## Context

Parallax lacks a graphical interface. Users currently interact via CLI or MCP. This PRD covers building a React+Bun single-page chat application that lets users generate image, video, and audio media through a conversational UI, with inline playback and download — without touching distribution or CLI integration.

## Goals

- Deliver a focused chat interface for media generation.
- Support all media types the server exposes: image, video, audio.
- Stream generation progress via SSE in the chat timeline.
- Allow inline playback of generated media without leaving the page.
- Enable file upload for workflows that require an input image (i2v, is2v).

## User Stories

### US-001: Select media type and configure parameters
**As a** user, **I want** to select image, video, or audio as the target media type and configure the relevant parameters (model, pipeline type, width, height, duration) **so that** I can tailor the generation to my needs before submitting.

**Acceptance Criteria:**
- [ ] A media type selector (image / video / audio) is visible and selectable before composing a prompt.
- [ ] Selecting "image" shows: model selector, width, height fields.
- [ ] Selecting "video" shows: model selector, pipeline type selector (t2v, i2v, is2v, flf2v, ia2v), width, height, duration fields.
- [ ] Selecting "audio" shows: model selector, duration field.
- [ ] All parameter fields have sensible defaults pre-filled (e.g. width=768, height=512, duration=5).
- [ ] Changing media type resets pipeline-type selector to the first valid option for that type.

### US-002: Upload input image for i2v and is2v pipelines
**As a** user, **I want** to upload a reference image when selecting an i2v, is2v, or ia2v pipeline **so that** I can use it as the conditioning input for the generation.

**Acceptance Criteria:**
- [ ] An image upload control appears only when pipeline type is i2v, is2v, or ia2v.
- [ ] The control accepts `.jpg`, `.jpeg`, `.png`, `.webp` files.
- [ ] After selecting a file, a thumbnail preview is shown inside the chat input area.
- [ ] The uploaded file is sent as `multipart/form-data` alongside the generation request.
- [ ] If no file is selected and pipeline type requires one, submitting shows an inline validation message and does not send the request.

### US-003: Submit prompt and stream generation progress
**As a** user, **I want** to type a prompt and submit the generation request, seeing live progress updates in the chat **so that** I know the job is running and how far along it is.

**Acceptance Criteria:**
- [ ] A text input at the bottom of the chat accepts the prompt (Enter to submit, Shift+Enter for newline).
- [ ] On submit, the prompt appears as a user bubble in the chat timeline.
- [ ] An assistant bubble appears immediately below, showing a progress indicator.
- [ ] The client connects to `GET /jobs/{id}/stream` (SSE) and renders each progress event as a percentage or status string inside the assistant bubble.
- [ ] If the SSE connection drops before completion, the bubble shows "Connection lost — check job status."
- [ ] Submitting while a job is in-flight is disabled (button grayed out, input disabled).

### US-004: Display generated media inline and allow download
**As a** user, **I want** the generated media to appear inline in the chat once the job completes **so that** I can review it immediately and download it if satisfied.

**Acceptance Criteria:**
- [ ] On job completion, the assistant bubble replaces the progress indicator with the generated media rendered inline:
  - Image: `<img>` tag with the result URL.
  - Video: `<video controls>` tag with the result URL; autoplay is off.
  - Audio: `<audio controls>` tag with the result URL.
- [ ] A "Download" button appears below the media element; clicking it triggers a browser download with the correct filename and extension.
- [ ] If the job fails, the assistant bubble shows the error message returned by the server.
- [ ] The chat history persists in-memory for the duration of the browser session (no localStorage required).

---

## Functional Requirements

- FR-1: The app is a single HTML page (SPA) built with React 18+ and Bun as the bundler/dev server.
- FR-2: All server communication uses the existing `server/:5000` REST API and SSE endpoints — no new server routes are introduced in this PRD.
- FR-3: Media type, pipeline type, model, width, height, and duration are sent as JSON body fields on `POST /create/{media_type}`.
- FR-4: The uploaded image (when present) is sent via `multipart/form-data`; if absent, `application/json` is used.
- FR-5: The app runs entirely in the browser — no Node/Bun process is required at runtime after the build step.
- FR-6: The built output is a self-contained `dist/` directory (one `index.html` + hashed JS/CSS assets) with no server-side rendering.
- FR-7: The server base URL defaults to `http://localhost:5000` and is configurable via a `VITE_API_URL`-equivalent build-time env var (`PARALLAX_API_URL`).

## Non-Goals

- Serving the frontend via FastAPI or any other server (covered by PRD 002).
- CLI install command (covered by PRD 003).
- Authentication or multi-user support.
- Persistent chat history across browser sessions.
- Mobile-responsive layout (desktop-first only).
- Dark/light theme toggle.

## Open Questions

- Which models and pipeline types should be hardcoded in the selector vs. fetched from a server endpoint? (Recommend: fetch from a new `GET /models` endpoint — but that endpoint is out of scope for this PRD; hardcode for now.)
- Should `ia2v` (audio+image to video) also show an audio upload field, or is the `--audio` path handled differently?

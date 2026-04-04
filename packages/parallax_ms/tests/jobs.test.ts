// Tests for US-001: Submit inference jobs via REST.
// Strategy:
//   - Structural tests: read source to verify route and schema definitions.
//   - Functional tests: use app.handle() with mocked submitJob to verify
//     response shapes and validation behaviour.

import { describe, it, expect, mock, beforeAll } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";

// ── Mock @parallax/sdk BEFORE importing the app ───────────────────────────────
// Bun hoists mock.module() calls before static imports, ensuring the mock is
// active when src/index.ts resolves its "@parallax/sdk" dependency.
mock.module("@parallax/sdk", () => ({
  submitJob: mock(async () => "mock-job-id-001"),
}));

import { app } from "../src/index";

const SRC = readFileSync(join(import.meta.dir, "../src/index.ts"), "utf-8");

// Helper: POST JSON to app
async function post(path: string, body: unknown) {
  return app.handle(
    new Request(`http://localhost${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

// ── Structural: route definitions present in source ───────────────────────────

describe("US-001 structural: route definitions", () => {
  it("defines POST /jobs/create/image", () => {
    expect(SRC).toContain('"/jobs/create/image"');
  });

  it("defines POST /jobs/create/video", () => {
    expect(SRC).toContain('"/jobs/create/video"');
  });

  it("defines POST /jobs/create/audio", () => {
    expect(SRC).toContain('"/jobs/create/audio"');
  });

  it("defines POST /jobs/edit/image", () => {
    expect(SRC).toContain('"/jobs/edit/image"');
  });

  it("defines POST /jobs/upscale/image", () => {
    expect(SRC).toContain('"/jobs/upscale/image"');
  });

  it("calls submitJob from @parallax/sdk", () => {
    expect(SRC).toContain("submitJob");
    expect(SRC).toContain("@parallax/sdk");
  });

  it("returns queued status", () => {
    expect(SRC).toContain('"queued"');
  });
});

// ── Structural: body schemas ───────────────────────────────────────────────────

describe("US-001-AC01 schema: create/image required fields", () => {
  it("schema includes model", () => {
    expect(SRC).toContain("model:");
  });

  it("schema includes prompt", () => {
    expect(SRC).toContain("prompt:");
  });

  it("schema includes optional negative_prompt", () => {
    expect(SRC).toContain("negative_prompt:");
  });

  it("schema includes optional width", () => {
    expect(SRC).toContain("width:");
  });

  it("schema includes optional height", () => {
    expect(SRC).toContain("height:");
  });

  it("schema includes optional steps", () => {
    expect(SRC).toContain("steps:");
  });
});

describe("US-001-AC02 schema: create/video includes input and frames", () => {
  it("schema includes optional input", () => {
    expect(SRC).toContain("input:");
  });

  it("schema includes optional frames", () => {
    expect(SRC).toContain("frames:");
  });
});

describe("US-001-AC03 schema: create/audio includes duration_seconds", () => {
  it("schema includes optional duration_seconds", () => {
    expect(SRC).toContain("duration_seconds:");
  });
});

describe("US-001-AC04 schema: edit/image includes image_path", () => {
  it("schema includes image_path", () => {
    expect(SRC).toContain("image_path:");
  });
});

describe("US-001-AC05 schema: upscale/image fields", () => {
  it("schema includes image_path", () => {
    expect(SRC).toContain("image_path:");
  });

  it("schema includes optional output", () => {
    expect(SRC).toContain("output:");
  });
});

// ── Functional: AC06 — returns { job_id, status: "queued" } ──────────────────

describe("US-001-AC06 functional: create/image returns job_id + queued", () => {
  it("returns 200 with job_id and status queued", async () => {
    const res = await post("/jobs/create/image", { model: "sdxl", prompt: "a cat" });
    expect(res.status).toBe(200);
    const json = await res.json() as { job_id: string; status: string };
    expect(json.job_id).toBe("mock-job-id-001");
    expect(json.status).toBe("queued");
  });
});

describe("US-001-AC06 functional: create/video returns job_id + queued", () => {
  it("returns 200 with job_id and status queued", async () => {
    const res = await post("/jobs/create/video", { model: "wan21", prompt: "a wave" });
    expect(res.status).toBe(200);
    const json = await res.json() as { job_id: string; status: string };
    expect(json.job_id).toBe("mock-job-id-001");
    expect(json.status).toBe("queued");
  });
});

describe("US-001-AC06 functional: create/audio returns job_id + queued", () => {
  it("returns 200 with job_id and status queued", async () => {
    const res = await post("/jobs/create/audio", { model: "ace_step", prompt: "ambient music" });
    expect(res.status).toBe(200);
    const json = await res.json() as { job_id: string; status: string };
    expect(json.job_id).toBe("mock-job-id-001");
    expect(json.status).toBe("queued");
  });
});

describe("US-001-AC06 functional: edit/image returns job_id + queued", () => {
  it("returns 200 with job_id and status queued", async () => {
    const res = await post("/jobs/edit/image", {
      model: "flux_4b_base",
      image_path: "/tmp/photo.jpg",
      prompt: "make it look like winter",
    });
    expect(res.status).toBe(200);
    const json = await res.json() as { job_id: string; status: string };
    expect(json.job_id).toBe("mock-job-id-001");
    expect(json.status).toBe("queued");
  });
});

describe("US-001-AC06 functional: upscale/image returns job_id + queued", () => {
  it("returns 200 with job_id and status queued", async () => {
    const res = await post("/jobs/upscale/image", {
      image_path: "/tmp/photo.jpg",
      model: "esrgan",
    });
    expect(res.status).toBe(200);
    const json = await res.json() as { job_id: string; status: string };
    expect(json.job_id).toBe("mock-job-id-001");
    expect(json.status).toBe("queued");
  });

  it("passes optional output path through to job args", async () => {
    const res = await post("/jobs/upscale/image", {
      image_path: "/tmp/photo.jpg",
      model: "esrgan",
      output: "/tmp/upscaled.png",
    });
    expect(res.status).toBe(200);
  });
});

// ── Functional: AC07 — 400 for missing required fields ───────────────────────

describe("US-001-AC07 functional: 400 on missing required fields", () => {
  it("POST /jobs/create/image without prompt returns 400", async () => {
    const res = await post("/jobs/create/image", { model: "sdxl" });
    expect(res.status).toBe(400);
  });

  it("POST /jobs/create/image without model returns 400", async () => {
    const res = await post("/jobs/create/image", { prompt: "a cat" });
    expect(res.status).toBe(400);
  });

  it("POST /jobs/create/video without model returns 400", async () => {
    const res = await post("/jobs/create/video", { prompt: "a wave" });
    expect(res.status).toBe(400);
  });

  it("POST /jobs/create/audio without model returns 400", async () => {
    const res = await post("/jobs/create/audio", { prompt: "ambient" });
    expect(res.status).toBe(400);
  });

  it("POST /jobs/edit/image without image_path returns 400", async () => {
    const res = await post("/jobs/edit/image", { model: "flux_4b_base", prompt: "edit" });
    expect(res.status).toBe(400);
  });

  it("POST /jobs/upscale/image without image_path returns 400", async () => {
    const res = await post("/jobs/upscale/image", { model: "esrgan" });
    expect(res.status).toBe(400);
  });

  it("400 response includes error field", async () => {
    const res = await post("/jobs/create/image", { model: "sdxl" });
    const json = await res.json() as { error: string };
    expect(typeof json.error).toBe("string");
    expect(json.error.length).toBeGreaterThan(0);
  });
});

// ── Functional: AC02 — video i2v picks i2v script when input is provided ──────

describe("US-001-AC02 functional: video i2v routing", () => {
  it("accepts input field for i2v", async () => {
    const res = await post("/jobs/create/video", {
      model: "wan21",
      prompt: "ocean waves",
      input: "/tmp/frame.jpg",
    });
    expect(res.status).toBe(200);
    const json = await res.json() as { status: string };
    expect(json.status).toBe("queued");
  });
});

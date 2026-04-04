// Tests for US-002: `edit image` action wires up flux and qwen models.
// Strategy: call buildEditImageArgs directly and assert the resulting args array
// contains the correct flags and values per model.

import { describe, it, expect } from "bun:test";
import { buildEditImageArgs, type EditImageOpts } from "../../src/models/image";
import { getScript } from "../../src/models/registry";

// Base opts shared across tests — override per-test as needed.
function baseOpts(model: string, overrides: Partial<EditImageOpts> = {}): EditImageOpts {
  return {
    model,
    prompt:  "test prompt",
    input:   "photo.jpg",
    width:   "1024",
    height:  "1024",
    steps:   "20",
    cfg:     "7",
    output:  "output.png",
    ...overrides,
  };
}

// US-002-AC01: edit.ts imports verified structurally (import resolution checked by typecheck).

// US-002-AC04: buildEditImageArgs maps CLI options to Python script args per model.

describe("US-002-AC04: flux_4b_base args", () => {
  const args = buildEditImageArgs(baseOpts("flux_4b_base"), "/models");

  it("includes --models-dir", () => {
    expect(args).toContain("--models-dir");
    expect(args[args.indexOf("--models-dir") + 1]).toBe("/models");
  });

  it("includes --prompt", () => {
    expect(args).toContain("--prompt");
    expect(args[args.indexOf("--prompt") + 1]).toBe("test prompt");
  });

  it("maps --input to --image", () => {
    expect(args).toContain("--image");
    expect(args[args.indexOf("--image") + 1]).toBe("photo.jpg");
    expect(args).not.toContain("--input");
  });

  it("includes --width and --height", () => {
    expect(args).toContain("--width");
    expect(args[args.indexOf("--width") + 1]).toBe("1024");
    expect(args).toContain("--height");
    expect(args[args.indexOf("--height") + 1]).toBe("1024");
  });

  it("includes --steps", () => {
    expect(args).toContain("--steps");
    expect(args[args.indexOf("--steps") + 1]).toBe("20");
  });

  it("does not include --cfg", () => {
    expect(args).not.toContain("--cfg");
  });

  it("includes --output", () => {
    expect(args).toContain("--output");
    expect(args[args.indexOf("--output") + 1]).toBe("output.png");
  });

  it("does not include --subject-image when not provided", () => {
    expect(args).not.toContain("--subject-image");
  });
});

describe("US-002-AC04: flux_9b_kv args", () => {
  const args = buildEditImageArgs(
    baseOpts("flux_9b_kv", { subjectImage: "subject.jpg" }),
    "/models"
  );

  it("includes --subject-image when provided", () => {
    expect(args).toContain("--subject-image");
    expect(args[args.indexOf("--subject-image") + 1]).toBe("subject.jpg");
  });

  it("maps --input to --image", () => {
    expect(args).toContain("--image");
    expect(args[args.indexOf("--image") + 1]).toBe("photo.jpg");
  });

  it("does not include --cfg", () => {
    expect(args).not.toContain("--cfg");
  });
});

describe("US-002-AC04: flux_9b_kv without subject-image", () => {
  const args = buildEditImageArgs(baseOpts("flux_9b_kv"), "/models");

  it("does not include --subject-image when not provided", () => {
    expect(args).not.toContain("--subject-image");
  });
});

describe("US-002-AC04: qwen args", () => {
  const args = buildEditImageArgs(baseOpts("qwen"), "/models");

  it("includes --models-dir", () => {
    expect(args).toContain("--models-dir");
  });

  it("maps --input to --image", () => {
    expect(args).toContain("--image");
    expect(args[args.indexOf("--image") + 1]).toBe("photo.jpg");
  });

  it("includes --prompt", () => {
    expect(args).toContain("--prompt");
  });

  it("includes --steps", () => {
    expect(args).toContain("--steps");
  });

  it("includes --cfg", () => {
    expect(args).toContain("--cfg");
    expect(args[args.indexOf("--cfg") + 1]).toBe("7");
  });

  it("maps --output output.png to --output-prefix output (strips .png)", () => {
    expect(args).toContain("--output-prefix");
    expect(args[args.indexOf("--output-prefix") + 1]).toBe("output");
    expect(args).not.toContain("--output");
  });

  it("does not include --width or --height", () => {
    expect(args).not.toContain("--width");
    expect(args).not.toContain("--height");
  });

  it("does not include --no-lora by default", () => {
    expect(args).not.toContain("--no-lora");
  });
});

describe("US-002-AC04: qwen --output-prefix strips .png suffix", () => {
  it("strips trailing .png from output", () => {
    const args = buildEditImageArgs(baseOpts("qwen", { output: "my_result.png" }), "/models");
    const idx = args.indexOf("--output-prefix");
    expect(idx).toBeGreaterThan(-1);
    expect(args[idx + 1]).toBe("my_result");
  });

  it("leaves output unchanged when no .png suffix", () => {
    const args = buildEditImageArgs(baseOpts("qwen", { output: "my_result" }), "/models");
    const idx = args.indexOf("--output-prefix");
    expect(idx).toBeGreaterThan(-1);
    expect(args[idx + 1]).toBe("my_result");
  });
});

describe("US-002-AC04: qwen optional flags", () => {
  it("includes --image2 when provided", () => {
    const args = buildEditImageArgs(baseOpts("qwen", { image2: "second.jpg" }), "/models");
    expect(args).toContain("--image2");
    expect(args[args.indexOf("--image2") + 1]).toBe("second.jpg");
  });

  it("includes --image3 when provided", () => {
    const args = buildEditImageArgs(baseOpts("qwen", { image3: "third.jpg" }), "/models");
    expect(args).toContain("--image3");
    expect(args[args.indexOf("--image3") + 1]).toBe("third.jpg");
  });

  it("includes --no-lora when noLora is true", () => {
    const args = buildEditImageArgs(baseOpts("qwen", { noLora: true }), "/models");
    expect(args).toContain("--no-lora");
  });

  it("excludes --image2 when not provided", () => {
    const args = buildEditImageArgs(baseOpts("qwen"), "/models");
    expect(args).not.toContain("--image2");
  });
});

describe("US-002-AC04: seed handling", () => {
  it("includes --seed when provided for flux model", () => {
    const args = buildEditImageArgs(baseOpts("flux_4b_base", { seed: "42" }), "/models");
    expect(args).toContain("--seed");
    expect(args[args.indexOf("--seed") + 1]).toBe("42");
  });

  it("includes --seed when provided for qwen", () => {
    const args = buildEditImageArgs(baseOpts("qwen", { seed: "99" }), "/models");
    expect(args).toContain("--seed");
    expect(args[args.indexOf("--seed") + 1]).toBe("99");
  });

  it("omits --seed when not provided", () => {
    const args = buildEditImageArgs(baseOpts("flux_4b_base"), "/models");
    expect(args).not.toContain("--seed");
  });
});

// US-002-AC05/AC06: getScript returns correct script for each model.
describe("US-002-AC05: getScript returns flux script for flux_4b_base", () => {
  it("returns the correct script path", () => {
    expect(getScript("edit", "image", "flux_4b_base")).toBe("runtime/image/edit/flux/4b_base.py");
  });
});

describe("US-002-AC06: getScript returns qwen script for qwen", () => {
  it("returns the correct script path", () => {
    expect(getScript("edit", "image", "qwen")).toBe("runtime/image/edit/qwen/edit_2511.py");
  });
});

describe("US-002: getScript returns correct scripts for all edit image models", () => {
  it("flux_4b_distilled", () => {
    expect(getScript("edit", "image", "flux_4b_distilled")).toBe("runtime/image/edit/flux/4b_distilled.py");
  });

  it("flux_9b_base", () => {
    expect(getScript("edit", "image", "flux_9b_base")).toBe("runtime/image/edit/flux/9b_base.py");
  });

  it("flux_9b_distilled", () => {
    expect(getScript("edit", "image", "flux_9b_distilled")).toBe("runtime/image/edit/flux/9b_distilled.py");
  });

  it("flux_9b_kv", () => {
    expect(getScript("edit", "image", "flux_9b_kv")).toBe("runtime/image/edit/flux/9b_kv.py");
  });
});

#!/usr/bin/env bun
import { Command } from "commander";
import { registerCreate } from "./commands/create";
import { registerEdit } from "./commands/edit";
import { registerInstall } from "./commands/install";

const program = new Command();

// Reformat commander's missing-required-option errors to "Error: --flag is required"
function formatRequiredFlagError(msg: string): string {
  return msg.replace(
    /error: required option '(--[a-z-]+)[^']*' not specified/,
    "Error: $1 is required"
  );
}

program
  .name("parallax")
  .description("Parallax CLI — comfy-diffusion media generation")
  .version("0.1.0")
  .addHelpText("before", "parallax v0.1.0\n")
  .configureOutput({
    writeErr: (str) => process.stderr.write(formatRequiredFlagError(str)),
  });

// ── create ────────────────────────────────────────────────────────────────────

registerCreate(program);

// ── edit ──────────────────────────────────────────────────────────────────────

registerEdit(program);

// ── install ───────────────────────────────────────────────────────────────────

registerInstall(program);

if (process.argv.length <= 2) {
  program.help();
}

program.parse();

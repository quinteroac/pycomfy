#!/usr/bin/env bun
import { Command } from "commander";
import { registerInstall } from "./commands/install";
import { registerCreate } from "./commands/create";
import { registerEdit } from "./commands/edit";
import { registerUpscale } from "./commands/upscale";
import { formatRequiredFlagError } from "./utils";

const program = new Command();
program
  .name("parallax")
  .description("Parallax CLI — comfy-diffusion media generation")
  .version("0.1.0")
  .addHelpText("before", "parallax v0.1.0\n")
  .configureOutput({ writeErr: (str) => process.stderr.write(formatRequiredFlagError(str)) });
registerInstall(program);
registerCreate(program);
registerEdit(program);
registerUpscale(program);
if (process.argv.length <= 2) program.help();
program.parse();

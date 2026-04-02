#!/usr/bin/env bun
import { Command } from "commander";

const program = new Command();

program
  .name("parallax")
  .description("Parallax CLI — comfy-diffusion media generation")
  .version("0.1.0");

// TODO: add commands (generate, edit, download, ...)

program.parse();

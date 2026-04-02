#!/usr/bin/env bun
import { Command } from "commander";

const program = new Command();

program
  .name("parallax")
  .description("Parallax CLI — comfy-diffusion media generation")
  .version("0.1.0")
  .addHelpText("before", "parallax v0.1.0\n");

program
  .command("create")
  .description("Generate a new image from a text prompt")
  .argument("<prompt>", "Text prompt describing the image to generate")
  .option("-o, --output <path>", "Output file path", "output.png")
  .action((_prompt: string, _options: { output: string }) => {
    console.error("Not yet implemented — coming soon.");
    process.exit(1);
  });

program
  .command("edit")
  .description("Edit an existing image using a text prompt")
  .argument("<image>", "Path to the input image")
  .argument("<prompt>", "Text prompt describing the desired edits")
  .option("-o, --output <path>", "Output file path", "output.png")
  .action((_image: string, _prompt: string, _options: { output: string }) => {
    console.error("Not yet implemented — coming soon.");
    process.exit(1);
  });

if (process.argv.length <= 2) {
  program.help();
}

program.parse();

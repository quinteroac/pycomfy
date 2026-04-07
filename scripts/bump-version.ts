#!/usr/bin/env bun

import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { execSync } from "node:child_process";

type BumpKind = "major" | "minor" | "patch";

const VALID_BUMPS = new Set<BumpKind>(["major", "minor", "patch"]);

function parseVersion(raw: string): [number, number, number] {
  const match = raw.match(/^(\d+)\.(\d+)\.(\d+)$/);
  if (!match) {
    throw new Error(`Invalid version format: ${raw}. Expected x.y.z`);
  }

  return [Number(match[1]), Number(match[2]), Number(match[3])];
}

function bumpVersion(current: string, kind: BumpKind): string {
  const [major, minor, patch] = parseVersion(current);

  if (kind === "major") {
    return `${major + 1}.0.0`;
  }
  if (kind === "minor") {
    return `${major}.${minor + 1}.0`;
  }
  return `${major}.${minor}.${patch + 1}`;
}

function normalizeInput(value: string): string {
  return value.startsWith("v") ? value.slice(1) : value;
}

function run(command: string): string {
  return execSync(command, {
    stdio: ["ignore", "pipe", "pipe"],
    encoding: "utf8",
  }).trim();
}

function gitRelease(nextVersion: string): void {
  const tag = `v${nextVersion}`;
  const branch = run("git rev-parse --abbrev-ref HEAD");

  const localTagExists = run(`git tag --list ${tag}`) === tag;
  if (localTagExists) {
    throw new Error(`Tag already exists locally: ${tag}`);
  }

  run("git add pyproject.toml package.json cli/_version.py");
  run(`git commit -m "chore(release): ${tag}"`);
  run(`git tag -a ${tag} -m "Release ${tag}"`);
  run(`git push origin ${branch}`);
  run(`git push origin ${tag}`);
}

function updatePackageJsonVersion(content: string, nextVersion: string): { updated: string; previous: string } {
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(content) as Record<string, unknown>;
  } catch {
    throw new Error("Could not parse package.json");
  }

  const previous = typeof parsed.version === "string" ? parsed.version : "";
  parsed.version = nextVersion;
  const updated = `${JSON.stringify(parsed, null, 2)}\n`;
  return { updated, previous };
}

function updatePyprojectVersion(content: string, nextVersionInput: string): { updated: string; previous: string; next: string } {
  const projectHeader = "[project]";
  const projectStart = content.indexOf(projectHeader);
  if (projectStart === -1) {
    throw new Error("Could not find [project] section in pyproject.toml");
  }

  const nextSectionStart = content.indexOf("\n[", projectStart + projectHeader.length);
  const sectionEnd = nextSectionStart === -1 ? content.length : nextSectionStart + 1;

  const projectSection = content.slice(projectStart, sectionEnd);
  const beforeSection = content.slice(0, projectStart);
  const afterSection = content.slice(sectionEnd);

  const versionLineRegex = /^(\s*version\s*=\s*["'])(\d+\.\d+\.\d+)(["']\s*)$/m;
  const versionMatch = projectSection.match(versionLineRegex);

  if (!versionMatch) {
    throw new Error("Could not find project version line in [project] section");
  }

  const previous = versionMatch[2];
  const normalized = normalizeInput(nextVersionInput);
  const next = VALID_BUMPS.has(normalized as BumpKind)
    ? bumpVersion(previous, normalized as BumpKind)
    : normalizeInput(nextVersionInput);

  parseVersion(next);

  const nextProjectSection = projectSection.replace(versionLineRegex, `$1${next}$3`);
  const updated = `${beforeSection}${nextProjectSection}${afterSection}`;

  return { updated, previous, next };
}

function main(): void {
  const envKind = (process.env.BUMP_KIND ?? "").trim().toLowerCase();
  const cliArg = process.argv[2]?.trim();
  const arg = envKind || cliArg || "patch";

  const pyprojectPath = resolve(process.cwd(), "pyproject.toml");
  const pyprojectContent = readFileSync(pyprojectPath, "utf8");
  const packageJsonPath = resolve(process.cwd(), "package.json");
  const packageJsonContent = readFileSync(packageJsonPath, "utf8");
  const versionPyPath = resolve(process.cwd(), "cli", "_version.py");
  const versionPyContent = readFileSync(versionPyPath, "utf8");

  const { updated, previous, next } = updatePyprojectVersion(pyprojectContent, arg);
  const { updated: updatedPackageJson } = updatePackageJsonVersion(packageJsonContent, next);
  const updatedVersionPy = versionPyContent.replace(
    /^(__version__\s*=\s*["'])([^"']+)(["'])$/m,
    `$1${next}$3`,
  );

  if (previous === next) {
    console.log(`Version already ${next} (no changes).`);
    return;
  }

  writeFileSync(pyprojectPath, updated, "utf8");
  writeFileSync(packageJsonPath, updatedPackageJson, "utf8");
  writeFileSync(versionPyPath, updatedVersionPy, "utf8");
  console.log(`Bumped version: ${previous} -> ${next}`);

  gitRelease(next);
  console.log(`Released ${next}: committed, tagged, and pushed.`);
}

main();

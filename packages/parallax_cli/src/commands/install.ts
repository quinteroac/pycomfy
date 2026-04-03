// Interactive installer command.
// Interactive TTY flow uses @clack/prompts; non-interactive mode (--non-interactive or no TTY) uses flag values / defaults.

import { Command } from "commander";
import { existsSync } from "fs";
import { homedir } from "os";
import { join } from "path";
import { configExists, readConfig, writeConfig } from "../config";

const DEFAULT_INSTALL_DIR = join(homedir(), ".parallax");
const DEFAULT_MODELS_DIR = join(homedir(), "parallax-models");
const DEFAULT_VARIANT = "cpu";
const DEFAULT_UV_PATH = join(homedir(), ".local", "bin", "uv");

interface InstallOpts {
  nonInteractive?: boolean;
  installDir: string;
  modelsDir: string;
  variant: string;
}

export function registerInstall(program: Command): void {
  program
    .command("install")
    .description("Interactive environment setup wizard")
    .option("--non-interactive", "Skip all prompts and use flag values or defaults")
    .option("--install-dir <path>", "Directory to install Parallax into", DEFAULT_INSTALL_DIR)
    .option("--models-dir <path>", "Directory to store models", DEFAULT_MODELS_DIR)
    .option("--variant <variant>", "Torch variant: cuda or cpu", DEFAULT_VARIANT)
    .action(async (opts: InstallOpts) => {
      const nonInteractive = opts.nonInteractive === true || !process.stdout.isTTY;

      if (nonInteractive) {
        await runNonInteractive(opts);
      } else {
        await runInteractive(opts);
      }
    });
}

async function runNonInteractive(opts: InstallOpts): Promise<void> {
  const { installDir, modelsDir, variant } = opts;

  applyConfig(installDir, modelsDir, variant);

  console.log("[parallax] Installation configured:");
  console.log(`  install-dir: ${installDir}`);
  console.log(`  models-dir:  ${modelsDir}`);
  console.log(`  variant:     ${variant}`);
  console.log("[parallax] Configuration saved to ~/.config/parallax/config.json");
}

async function runInteractive(opts: InstallOpts): Promise<void> {
  const { intro, outro, text, select, confirm, spinner, isCancel, cancel } = await import("@clack/prompts");

  intro("Parallax — environment setup");

  // Confirm before reinstalling if config already exists.
  if (configExists()) {
    const shouldReinstall = await confirm({ message: "Existing configuration found. Reinstall?" });
    if (isCancel(shouldReinstall) || !shouldReinstall) {
      cancel("Installation cancelled.");
      process.exit(0);
    }
  }

  const installDirAnswer = await text({
    message: "Install directory",
    placeholder: opts.installDir,
    defaultValue: opts.installDir,
  });
  if (isCancel(installDirAnswer)) {
    cancel("Installation cancelled.");
    process.exit(0);
  }

  const modelsDirAnswer = await text({
    message: "Models directory",
    placeholder: opts.modelsDir,
    defaultValue: opts.modelsDir,
  });
  if (isCancel(modelsDirAnswer)) {
    cancel("Installation cancelled.");
    process.exit(0);
  }

  const variantAnswer = await select({
    message: "Torch variant",
    options: [
      { value: "cpu", label: "CPU (no GPU required)" },
      { value: "cuda", label: "CUDA (NVIDIA GPU)" },
    ],
    initialValue: opts.variant,
  });
  if (isCancel(variantAnswer)) {
    cancel("Installation cancelled.");
    process.exit(0);
  }

  // Detect or install uv.
  const uvPath = await detectOrInstallUv(spinner);

  // Set up Python environment with uv venv + uv sync.
  const s = spinner();
  s.start("Setting up Python environment…");
  const venvPath = join(installDirAnswer as string, ".venv");
  const venvProc = Bun.spawn([uvPath, "venv", venvPath], {
    cwd: installDirAnswer as string,
    stdout: "pipe",
    stderr: "pipe",
  });
  const venvExit = await venvProc.exited;
  if (venvExit !== 0) {
    s.stop("Failed to create virtual environment.");
    cancel("Setup failed. Check that uv is working and the install directory is valid.");
    process.exit(1);
  }

  const syncProc = Bun.spawn([uvPath, "sync", "--extra", variantAnswer as string], {
    cwd: installDirAnswer as string,
    stdout: "pipe",
    stderr: "pipe",
  });
  const syncExit = await syncProc.exited;
  if (syncExit !== 0) {
    s.stop("Failed to sync dependencies.");
    cancel("uv sync failed. Check your pyproject.toml and network connectivity.");
    process.exit(1);
  }
  s.stop("Python environment ready.");

  applyConfig(
    installDirAnswer as string,
    modelsDirAnswer as string,
    variantAnswer as string,
    uvPath,
  );

  outro("Listo. Ejecuta: parallax create image --help");
}

async function detectOrInstallUv(
  spinnerFn: () => { start: (msg: string) => void; stop: (msg: string) => void },
): Promise<string> {
  // Check ~/.local/bin/uv first.
  if (existsSync(DEFAULT_UV_PATH)) return DEFAULT_UV_PATH;

  // Check if uv is available in PATH.
  const inPath = Bun.which("uv");
  if (inPath) return inPath;

  // Download uv to ~/.local/bin.
  const s = spinnerFn();
  s.start("Downloading uv to ~/.local/bin…");
  const installProc = Bun.spawn(["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], {
    stdout: "pipe",
    stderr: "pipe",
  });
  const installExit = await installProc.exited;
  if (installExit !== 0 || !existsSync(DEFAULT_UV_PATH)) {
    s.stop("Failed to download uv.");
    throw new Error("Could not install uv automatically. Please install it manually: https://docs.astral.sh/uv/");
  }
  s.stop("uv installed to ~/.local/bin.");
  return DEFAULT_UV_PATH;
}

function applyConfig(installDir: string, modelsDir: string, variant: string, uvPath?: string): void {
  const stored = readConfig();
  writeConfig({
    ...stored,
    repoRoot: installDir,
    modelsDir,
    variant,
    installedAt: new Date().toISOString(),
    ...(uvPath !== undefined && { uvPath }),
  });
}


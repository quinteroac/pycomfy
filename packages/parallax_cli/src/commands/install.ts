// Interactive installer command.
// Interactive TTY flow uses @clack/prompts; non-interactive mode (--non-interactive or no TTY) uses flag values / defaults.

import { Command } from "commander";
import { homedir } from "os";
import { join } from "path";
import { readConfig, writeConfig } from "../config";

const DEFAULT_INSTALL_DIR = join(homedir(), ".parallax");
const DEFAULT_MODELS_DIR = join(homedir(), ".parallax", "models");
const DEFAULT_VARIANT = "cpu";

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
  const { intro, outro, text, select, isCancel, cancel } = await import("@clack/prompts");

  intro("Parallax — environment setup");

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

  applyConfig(
    installDirAnswer as string,
    modelsDirAnswer as string,
    variantAnswer as string,
  );

  outro("Configuration saved. Run `parallax create` to get started.");
}

function applyConfig(installDir: string, modelsDir: string, variant: string): void {
  const stored = readConfig();
  writeConfig({
    ...stored,
    repoRoot: installDir,
    modelsDir,
    variant,
    installedAt: new Date().toISOString(),
  });
}

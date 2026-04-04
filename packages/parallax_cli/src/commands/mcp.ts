// MCP installer command.
// `parallax mcp install` — interactive prompt to configure AI clients with the MCP server.
// Non-interactive mode (--non-interactive or no TTY) uses --clients flag value or installs all.

import { Command } from "commander";
import { applyClientConfig, getMcpServerEntry, type ApplyResult } from "../mcp_config";

export const SUPPORTED_CLIENTS = ["claude", "gemini", "github-copilot", "codex"] as const;
export type SupportedClient = (typeof SUPPORTED_CLIENTS)[number];

export const CLIENT_LABELS: Record<SupportedClient, string> = {
  claude: "Claude",
  gemini: "Gemini",
  "github-copilot": "GitHub Copilot",
  codex: "Codex",
};

interface McpInstallOpts {
  nonInteractive?: boolean;
  clients?: string;
}

export function registerMcp(program: Command): void {
  const mcp = program
    .command("mcp")
    .description("MCP server management");

  mcp
    .command("install")
    .description("Configure AI clients to use the Parallax MCP server")
    .option("--non-interactive", "Skip prompts and use --clients flag or install all")
    .option(
      "--clients <list>",
      `Comma-separated list of clients to install (${SUPPORTED_CLIENTS.join(", ")})`,
    )
    .action(async (opts: McpInstallOpts) => {
      const nonInteractive = opts.nonInteractive === true || !process.stdout.isTTY;

      if (nonInteractive) {
        await runNonInteractive(opts);
      } else {
        await runInteractive();
      }
    });
}

async function runInteractive(): Promise<void> {
  const { intro, outro, multiselect, isCancel, cancel } = await import("@clack/prompts");

  intro("Parallax MCP — client setup");

  const selected = await multiselect({
    message: "Select AI clients to configure:",
    options: SUPPORTED_CLIENTS.map((value) => ({
      value,
      label: CLIENT_LABELS[value],
    })),
    required: true,
  });

  if (isCancel(selected)) {
    cancel("MCP installation cancelled.");
    process.exit(0);
  }

  const clients = selected as SupportedClient[];
  const results = installClients(clients);
  const succeeded = results.filter((r) => r.success);
  const failed = results.filter((r) => !r.success);

  if (failed.length > 0) {
    for (const r of failed) {
      console.error(`  [error] ${CLIENT_LABELS[r.client]}: ${r.error}`);
    }
  }

  outro(
    `Done. Configured: ${succeeded.map((r) => CLIENT_LABELS[r.client]).join(", ")}` +
      (failed.length > 0 ? ` (${failed.length} failed)` : ""),
  );
}

async function runNonInteractive(opts: McpInstallOpts): Promise<void> {
  const clients: SupportedClient[] = opts.clients
    ? (opts.clients
        .split(",")
        .map((s) => s.trim())
        .filter((s): s is SupportedClient =>
          SUPPORTED_CLIENTS.includes(s as SupportedClient),
        ))
    : [...SUPPORTED_CLIENTS];

  const results = installClients(clients);
  const succeeded = results.filter((r) => r.success);
  const failed = results.filter((r) => !r.success);

  if (failed.length > 0) {
    for (const r of failed) {
      console.error(`  [error] ${CLIENT_LABELS[r.client]}: ${r.error}`);
    }
  }

  console.log("[parallax] MCP clients configured:");
  for (const r of succeeded) {
    console.log(`  ${CLIENT_LABELS[r.client]} → ${r.configPath}`);
  }
}

function installClients(clients: SupportedClient[]): ApplyResult[] {
  const serverEntry = getMcpServerEntry();
  return clients.map((client) => applyClientConfig(client, serverEntry));
}

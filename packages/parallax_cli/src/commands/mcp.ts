// MCP installer command.
// `parallax mcp install` — interactive prompt to configure AI clients with the MCP server.
// Non-interactive mode (--non-interactive or no TTY) uses --clients flag value or installs all.

import { Command } from "commander";

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
  installClients(clients);

  outro(`Done. Configured: ${clients.map((c) => CLIENT_LABELS[c]).join(", ")}`);
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

  installClients(clients);

  console.log("[parallax] MCP clients configured:");
  for (const client of clients) {
    console.log(`  ${CLIENT_LABELS[client]}`);
  }
}

function installClients(clients: SupportedClient[]): void {
  // Placeholder: each client config writer will be implemented per-client.
  // For now we log the intent — the actual config file writes come in follow-up stories.
  for (const client of clients) {
    console.log(`[parallax] Configuring ${CLIENT_LABELS[client]}…`);
  }
}

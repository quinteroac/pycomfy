// MCP configuration writer — resolves config file paths per client/OS and merges the
// parallax server entry into the target JSON file.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { homedir } from "os";
import { dirname, join, resolve } from "path";

import type { SupportedClient } from "./commands/mcp";

export interface McpServerEntry {
  command: string;
  args: string[];
}

export interface ApplyResult {
  client: SupportedClient;
  success: boolean;
  configPath: string;
  error?: string;
}

// Resolve the absolute path to the parallax_mcp entry point, working in both dev
// (bun run src/index.ts) and compiled-binary modes.
export function getMcpServerEntry(): McpServerEntry {
  const mcpPath = resolve(import.meta.dir, "../../parallax_mcp/src/index.ts");
  return { command: "bun", args: ["run", mcpPath] };
}

/**
 * Return the OS-specific config file path for the given client.
 *
 * @param client    - One of the supported client identifiers.
 * @param platform  - Defaults to process.platform; pass explicitly for tests.
 * @param home      - Defaults to os.homedir(); pass explicitly for tests.
 * @param appData   - Defaults to %APPDATA% (Windows only); pass explicitly for tests.
 */
export function getConfigPath(
  client: SupportedClient,
  platform: string = process.platform,
  home: string = homedir(),
  appData: string = process.env.APPDATA ?? join(home, "AppData", "Roaming"),
): string {
  switch (client) {
    case "claude":
      if (platform === "win32") {
        return join(appData, "Claude", "claude_desktop_config.json");
      }
      if (platform === "darwin") {
        return join(home, "Library", "Application Support", "Claude", "claude_desktop_config.json");
      }
      return join(home, ".config", "Claude", "claude_desktop_config.json");

    case "gemini":
      return join(home, ".gemini", "settings.json");

    case "github-copilot":
      return join(home, ".copilot", "mcp-config.json");

    case "codex":
      if (platform === "win32") {
        return join(process.env.USERPROFILE ?? home, ".cursor", "mcp.json");
      }
      return join(home, ".cursor", "mcp.json");
  }
}

/**
 * Merge the parallax MCP server entry into the client config file.
 * Creates the file (and parent directories) if they do not exist.
 * Existing keys in the config are preserved.
 *
 * @param client      - Target client.
 * @param serverEntry - The MCP server command/args to write.
 * @param configPath  - Explicit path override (defaults to getConfigPath result; useful for tests).
 */
export function applyClientConfig(
  client: SupportedClient,
  serverEntry: McpServerEntry,
  configPath: string = getConfigPath(client),
): ApplyResult {
  try {
    const dir = dirname(configPath);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }

    let config: Record<string, unknown> = {};
    if (existsSync(configPath)) {
      config = JSON.parse(readFileSync(configPath, "utf8"));
    }

    const mcpServers = (config["mcpServers"] ?? {}) as Record<string, unknown>;
    mcpServers["parallax"] = serverEntry;
    config["mcpServers"] = mcpServers;

    writeFileSync(configPath, JSON.stringify(config, null, 2) + "\n", "utf8");
    return { client, success: true, configPath };
  } catch (err) {
    return {
      client,
      success: false,
      configPath,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

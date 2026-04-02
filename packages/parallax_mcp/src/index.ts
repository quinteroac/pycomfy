import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new McpServer({
  name: "parallax-mcp",
  version: "0.1.0",
});

// TODO: register tools (generate_image, edit_image, ...)

const transport = new StdioServerTransport();
await server.connect(transport);

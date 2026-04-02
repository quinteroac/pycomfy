import { Elysia } from "elysia";

const app = new Elysia()
  .get("/health", () => ({ status: "ok" }))
  // TODO: add routes
  .listen(3000);

console.log(`parallax_ms running at ${app.server?.hostname}:${app.server?.port}`);

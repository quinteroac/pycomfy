/**
 * Minimal type declarations for Node.js built-in modules used in vitest.
 * Vitest runs in Node.js, so fs/url/path are available at runtime even though
 * @types/node is not a project dependency.
 */
declare module "fs" {
  function readFileSync(path: string, encoding: "utf-8" | BufferEncoding): string;
  export { readFileSync };
}

declare module "url" {
  function fileURLToPath(url: string | URL): string;
  export { fileURLToPath };
}

declare module "path" {
  function dirname(p: string): string;
  function join(...paths: string[]): string;
  export { dirname, join };
}

import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["browser-src/**/*.test.ts"],
    exclude: ["**/node_modules/**", "**/target/**", "**/dist/**"],
    passWithNoTests: true,
  },
});

// @vitest-environment jsdom
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AppState, Project } from "./types";
import { testProject } from "./test/fixtures";

beforeEach(() => {
  localStorage.clear();
  vi.resetModules();
});

describe("demo RAW preference", () => {
  it("normalizes a legacy saved project and its summary to separate RAW and JPEG imports", async () => {
    const legacy = testProject();
    delete (legacy as Partial<Project>).preferRaw;
    localStorage.setItem(
      "photo-select-explicit-demo-v1",
      JSON.stringify(legacy),
    );
    const { demoInvoke } = await import("./demo");
    const project = await demoInvoke<Project>("open_project");
    const state = await demoInvoke<AppState>("get_app_state");
    expect(project.preferRaw).toBe(false);
    expect(state.projects[0].preferRaw).toBe(false);
  });
  it.each([false, true])(
    "retains preferRaw=%s on a created demo project and its summary",
    async (preferRaw) => {
      const { demoInvoke } = await import("./demo");
      const project = await demoInvoke<Project>("create_project", {
        name: "New trip",
        sourceDir: "/photos",
        includeSubfolders: false,
        automaticSelectionEnabled: false,
        preferRaw,
      });
      const state = await demoInvoke<AppState>("get_app_state");
      expect(project.preferRaw).toBe(preferRaw);
      expect(state.projects[0].preferRaw).toBe(preferRaw);
    },
  );
});

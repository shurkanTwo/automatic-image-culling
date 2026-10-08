// @vitest-environment jsdom
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import Viewer from "./Viewer";
import type { Photo } from "../types";
vi.mock("../api", () => ({ photoSrc: (path: string) => path }));
afterEach(cleanup);
beforeEach(() => {
  Object.defineProperty(window, "devicePixelRatio", {
    configurable: true,
    value: 1,
  });
});
const photo: Photo = {
  id: "one",
  path: "/one.raw",
  filename: "one.raw",
  previewPath: "/preview.jpg",
  thumbnailPath: "/thumbnail.jpg",
  detailPath: null,
  captureTime: "",
  width: 1944,
  height: 1296,
  camera: null,
  groupId: null,
  qualityScore: 70,
  hints: [],
  rating: 0,
  ratingTouched: false,
  decision: "undecided",
  reviewed: false,
  tags: [],
  analysisError: null,
};
function loaded(image: HTMLElement, width: number, height: number) {
  Object.defineProperty(image, "naturalWidth", {
    configurable: true,
    value: width,
  });
  Object.defineProperty(image, "naturalHeight", {
    configurable: true,
    value: height,
  });
  fireEvent.load(image);
}
function setPaneSize(
  container: HTMLElement,
  width = 484,
  height = 400,
  index = 0,
) {
  const pane = container.querySelectorAll(".photo-canvas")[index]!;
  Object.defineProperty(pane, "clientWidth", {
    configurable: true,
    value: width,
  });
  Object.defineProperty(pane, "clientHeight", {
    configurable: true,
    value: height,
  });
}
function scale(container: HTMLElement, index = 0) {
  return Number(
    container
      .querySelectorAll(".transformed-photo")
      [index]?.getAttribute("style")
      ?.match(/scale\(([^)]+)\)/)?.[1],
  );
}
describe("actual pixel inspection", () => {
  it.each([1, 1.5, 2])(
    "maps one source pixel to one physical display pixel at DPR %s",
    async (dpr) => {
      Object.defineProperty(window, "devicePixelRatio", {
        configurable: true,
        value: dpr,
      });
      const { container } = render(
        <Viewer
          photos={[{ ...photo, detailPath: "/full.jpg" }]}
          recommended={new Set()}
          onDetail={vi.fn(async () => true)}
          loadingDetail={false}
          onActive={vi.fn()}
        />,
      );
      setPaneSize(container);
      loaded(screen.getByRole("img"), 1936, 1296);
      fireEvent.click(screen.getByRole("button", { name: "100%" }));
      await waitFor(() => expect(scale(container)).toBeCloseTo(4 / dpr));
      expect(484 * scale(container) * dpr).toBeCloseTo(1936);
      expect(
        screen.getByText("100%", { selector: ".zoom-label" }),
      ).toBeTruthy();
    },
  );
  it("normalizes differing comparison images independently and retains linked relative zoom", async () => {
    Object.defineProperty(window, "devicePixelRatio", {
      configurable: true,
      value: 2,
    });
    const second = {
      ...photo,
      id: "two",
      filename: "two.raw",
      detailPath: "/full-two.jpg",
    };
    const { container } = render(
      <Viewer
        photos={[{ ...photo, detailPath: "/full.jpg" }, second]}
        recommended={new Set()}
        onDetail={vi.fn(async () => true)}
        loadingDetail={false}
        onActive={vi.fn()}
      />,
    );
    setPaneSize(container);
    setPaneSize(container, 600, 300, 1);
    loaded(screen.getAllByRole("img")[0], 1936, 1296);
    loaded(screen.getAllByRole("img")[1], 1000, 2000);
    fireEvent.click(screen.getByRole("button", { name: "100%" }));
    await waitFor(() => expect(scale(container)).toBeCloseTo(2));
    expect(scale(container, 1)).toBeCloseTo(1 / (0.15 * 2));
    expect(484 * scale(container) * 2).toBeCloseTo(1936);
    expect(300 * scale(container, 1) * 2).toBeCloseTo(2000);
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    expect(scale(container)).toBeCloseTo(3);
    expect(scale(container, 1)).toBeCloseTo(5);
    expect(screen.getByText("150%", { selector: ".zoom-label" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Fit" }));
    expect(scale(container)).toBe(1);
    expect(scale(container, 1)).toBe(1);
  });
  it("keeps actual pixel size when display scaling or viewport dimensions change", async () => {
    const { container } = render(
      <Viewer
        photos={[{ ...photo, detailPath: "/full.jpg" }]}
        recommended={new Set()}
        onDetail={vi.fn(async () => true)}
        loadingDetail={false}
        onActive={vi.fn()}
      />,
    );
    setPaneSize(container);
    loaded(screen.getByRole("img"), 1936, 1296);
    fireEvent.click(screen.getByRole("button", { name: "100%" }));
    await waitFor(() => expect(scale(container)).toBe(4));
    Object.defineProperty(window, "devicePixelRatio", {
      configurable: true,
      value: 2,
    });
    setPaneSize(container, 968, 800);
    fireEvent(window, new Event("resize"));
    await waitFor(() => expect(scale(container)).toBe(1));
    expect(968 * scale(container) * window.devicePixelRatio).toBeCloseTo(1936);
  });
  it("retries a failed cached detail after regeneration even when the cache path stays the same", async () => {
    const onDetail = vi.fn(async () => true);
    render(
      <Viewer
        photos={[{ ...photo, detailPath: "/full.jpg" }]}
        recommended={new Set()}
        onDetail={onDetail}
        loadingDetail={false}
        onActive={vi.fn()}
      />,
    );
    fireEvent.error(screen.getByRole("img"));
    expect(screen.queryByRole("img")).toBeNull();
    fireEvent.click(
      screen.getByRole("button", { name: "Full-resolution detail" }),
    );
    const reloaded = await screen.findByRole("img");
    expect(reloaded.getAttribute("src")).toBe("/full.jpg?detailRevision=1");
  });
  it("prepares full-resolution detail before zooming and uses loaded detail dimensions rather than metadata", async () => {
    let finish!: (value: boolean) => void;
    const onDetail = vi.fn(
      () =>
        new Promise<boolean>((resolve) => {
          finish = resolve;
        }),
    );
    const props = {
      photos: [photo],
      recommended: new Set<string>(),
      onDetail,
      loadingDetail: false,
      onActive: vi.fn(),
    };
    const { container, rerender } = render(<Viewer {...props} />);
    setPaneSize(container);
    loaded(screen.getByRole("img"), 960, 640);
    expect(screen.getByText("Preview")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "100%" }));
    expect(onDetail).toHaveBeenCalledWith([photo]);
    expect(
      screen
        .getByRole("button", { name: /Preparing 100%/ })
        .hasAttribute("disabled"),
    ).toBe(true);
    expect(
      container.querySelector(".transformed-photo")?.getAttribute("style"),
    ).toContain("scale(1)");
    await act(async () => {
      finish(true);
    });
    rerender(
      <Viewer {...props} photos={[{ ...photo, detailPath: "/full.jpg" }]} />,
    );
    loaded(screen.getByRole("img"), 1936, 1296);
    await waitFor(() =>
      expect(
        container.querySelector(".transformed-photo")?.getAttribute("style"),
      ).toContain("scale(4)"),
    );
    expect(screen.getByText("Full-resolution")).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "100%" }).hasAttribute("disabled"),
    ).toBe(false);
  });
  it("uses a cached full-resolution image at true pixel size without another detail request", async () => {
    const onDetail = vi.fn(async () => true);
    const { container } = render(
      <Viewer
        photos={[{ ...photo, detailPath: "/full.jpg" }]}
        recommended={new Set()}
        onDetail={onDetail}
        loadingDetail={false}
        onActive={vi.fn()}
      />,
    );
    setPaneSize(container, 1000, 700);
    loaded(screen.getByRole("img"), 500, 350);
    fireEvent.click(screen.getByRole("button", { name: "100%" }));
    await waitFor(() =>
      expect(
        container.querySelector(".transformed-photo")?.getAttribute("style"),
      ).toContain("scale(0.5)"),
    );
    expect(onDetail).not.toHaveBeenCalled();
  });
  it("returns to a usable preview when full-resolution preparation fails", async () => {
    const onDetail = vi.fn(async () => false);
    render(
      <Viewer
        photos={[photo]}
        recommended={new Set()}
        onDetail={onDetail}
        loadingDetail={false}
        onActive={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "100%" }));
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "100%" }).hasAttribute("disabled"),
      ).toBe(false),
    );
    expect(screen.getByText("Preview")).toBeTruthy();
  });
});

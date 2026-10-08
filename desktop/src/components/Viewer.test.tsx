// @vitest-environment jsdom
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import Viewer from "./Viewer";
import type { Photo } from "../types";
vi.mock("../api", () => ({ photoSrc: (path: string) => path }));
afterEach(cleanup);
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
function setPaneSize(container: HTMLElement, width = 484, height = 400) {
  const pane = container.querySelector(".photo-canvas")!;
  Object.defineProperty(pane, "clientWidth", {
    configurable: true,
    value: width,
  });
  Object.defineProperty(pane, "clientHeight", {
    configurable: true,
    value: height,
  });
}
describe("actual pixel inspection", () => {
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

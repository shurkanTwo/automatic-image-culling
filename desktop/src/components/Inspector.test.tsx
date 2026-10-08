// @vitest-environment jsdom
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import Inspector from "./Inspector";
import { testPhoto } from "../test/fixtures";
afterEach(cleanup);
describe("tag drafts", () => {
  it("preserves a newer focused draft when the preceding save refreshes photo tags", async () => {
    let finish!: (value: boolean) => void;
    const onSaveTags = vi.fn(
      () =>
        new Promise<boolean>((resolve) => {
          finish = resolve;
        }),
    );
    const photo = testPhoto(0);
    const props = {
      photo,
      selectedCount: 1,
      selectionKey: photo.id,
      collections: [],
      onSaveTags,
      onEdit: vi.fn(),
      onCollection: vi.fn(),
      onReveal: vi.fn(),
    };
    const { rerender } = render(<Inspector {...props} />);
    const input = screen.getByLabelText("Tags") as HTMLTextAreaElement;
    fireEvent.change(input, { target: { value: "family, one" } });
    fireEvent.blur(input);
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: "family, one, two" } });
    rerender(
      <Inspector {...props} photo={{ ...photo, tags: ["family", "one"] }} />,
    );
    expect(input.value).toBe("family, one, two");
    await act(async () => {
      finish(true);
    });
    expect(input.value).toBe("family, one, two");
    fireEvent.blur(input);
    expect(onSaveTags).toHaveBeenLastCalledWith(
      ["family", "one", "two"],
      false,
    );
  });
  it("does not commit an unchanged bulk field or copy the active photo tags to other photos", () => {
    const onSaveTags = vi.fn(async () => true);
    render(
      <Inspector
        photo={testPhoto(0)}
        selectedCount={2}
        selectionKey="photo-0|photo-1"
        collections={[]}
        onSaveTags={onSaveTags}
        onEdit={vi.fn()}
        onCollection={vi.fn()}
        onReveal={vi.fn()}
      />,
    );
    const input = screen.getByLabelText("Tags") as HTMLTextAreaElement;
    expect(input.value).toBe("");
    fireEvent.blur(input);
    expect(onSaveTags).not.toHaveBeenCalled();
  });
});

export const COMMIT_DRAFTS_EVENT = "photo-select-commit-drafts";

export function commitDrafts(): void {
  if (document.activeElement instanceof HTMLElement)
    document.activeElement.blur();
  document.dispatchEvent(new Event(COMMIT_DRAFTS_EVENT));
}

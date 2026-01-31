# Coding Standards

Project rules focused on clarity and predictable behavior. Follow these unless a documented exception is required.

- **Style and readability**

  - Keep functions small and focused; avoid deep nesting and long parameter lists.
  - Prefer explicit, readable logic over clever shortcuts.
  - Keep one statement per line; break complex conditions into named variables.

- **Interfaces**

  - Use positional args for required inputs and keyword args for optional behavior.
  - Avoid `*args/**kwargs` unless the API is genuinely variadic.
  - Mark non-public helpers with a leading underscore and respect it in other modules.

- **Control flow**

  - Use early returns for invalid input and keep the main success path simple.
  - Extract helpers instead of building tall `if/else` pyramids.

- **Python usage**

  - Use comprehensions and generators for building collections, not for side effects.
  - Use `enumerate` for indexes; avoid mutating lists while iterating.
  - Prefer `dict.get()` and `in` for dictionary access; use `is None` for `None` checks.

- **Data and I/O**

  - Join strings with `''.join(...)` in loops; avoid repeated concatenation.
  - Always use `with open(...)` for file handling.

- **Tooling and tests**

  - Run a formatter (e.g., `black`) and keep linting enabled.
  - Add or update tests with behavior changes; keep checks fast and deterministic.

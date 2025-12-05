# Coding Standards

Reference guide for this project. Default to clarity, explicit intent, and the least surprising solution. These rules are mandatory unless a clearly documented exception is needed.

- **Follow the basics**

  - Conform to PEP 8; run a formatter (e.g., `black`) before committing.
  - Prefer type hints and docstrings for public functions and modules.
  - Keep functions cohesive; avoid long parameter lists and deep nesting.
  - Readability beats cleverness; the Zen of Python applies.

- **Be explicit**

  - Accept arguments explicitly instead of packing/unpacking magic (avoid `*args/**kwargs` unless truly needed).
  - Use straightforward returns; one primary success exit, early returns only for errors or guard clauses.
  - Avoid “black magic” (monkey-patching import hooks or constructors) unless absolutely required and documented.

- **One statement per line**

  - Do not chain unrelated statements on a single line; expand complex conditions into named intermediates when it improves clarity.

- **Function signatures**

  - Use positional arguments for essential inputs with natural ordering.
  - Use keyword arguments with sensible defaults for optional behavior; resist adding “just in case” options (YAGNI).
  - Reach for `*args` only when the extensible positional list is the intent; otherwise pass a sequence explicitly.
  - Use `**kwargs` only when truly open-ended; prefer explicit keywords.

- **Responsible interfaces**

  - Mark non-public helpers with a leading underscore; respect others’ underscored members.
  - Avoid relying on side effects that aren’t obvious from the signature or docstring.

- **Control flow and returns**

  - Early-return on invalid input; keep the main success path single-exit where practical.
  - Keep branching shallow; extract helpers instead of nesting.

- **Pythonic idioms**

  - Use unpacking (including extended unpacking) for clarity; swap with `a, b = b, a`.
  - Use `__` for throwaway variables when needed to avoid clobbering `_`.
  - Build lists with comprehensions only when creating a new list is required; never use comprehensions for side effects.
  - Prefer generator expressions when the result is only iterated once.
  - Do not mutate lists while iterating them; create filtered copies or reassign via slice (`seq[:] = [...]`) if you must preserve the reference.
  - Use `enumerate` instead of manual counters.

- **Collections and performance**

  - Choose sets/dicts over lists for frequent membership tests or large collections.
  - Create uniform-length lists with `[value] * n` only for immutables; for mutables, use comprehensions (e.g., `[[] for __ in range(n)]`).

- **String and data handling**

  - Join strings with `''.join(seq)`; avoid repeated concatenation in loops.
  - Use truthiness directly (`if attr:`) and explicit `is None` when checking for `None`.
  - Access dictionaries with `in` or `dict.get()`; never use `has_key`.

- **File I/O and long lines**

  - Always use context managers for files (`with open(...) as f:`).
  - Break long logical lines with parentheses, not backslashes.

- **Tooling and automation**

  - Keep formatting automated (e.g., `black`, `autopep8`, or `yapf`) and lint regularly.
  - Add tests alongside changes; prefer fast, deterministic checks in CI.

- **Examples of “good” vs “bad” condensed**
  - Explicit args: prefer `def fn(x, y): return {"x": x, "y": y}` over unpacking `*args` and `locals()`.
  - Statements: avoid `print("one"); print("two")`; place each statement on its own line.
  - Conditions: split complex inline conditions into named variables before the `if`.
  - Collections: filter with comprehensions/generators, not by removing items mid-iteration.
  - Reads: `with open(path) as f: ...` instead of manual open/close.

Source: https://docs.python-guide.org/writing/style/

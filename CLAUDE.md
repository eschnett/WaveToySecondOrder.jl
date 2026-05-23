# Instructions for AI agents

Adhere to these preferences if possible:

- Use CairoMakie.jl for plotting. The figure width should be no more
  than 800 pixels to fit on a terminal window.
- When writing short throw-away scripts that are executed e.g. in
  /tmp, do not use `#` for comments if the file is generated via a
  shell command. Such comments can trigger alerts because they are
  also shell comments. Use another mechanism for comments, e.g. output
  them via `println`, or keep them as triple-quited string in the
  source code, or similar.

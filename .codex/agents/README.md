# Project-specific Codex agents

These five small, read-only agent profiles are adapted from [VoltAgent/awesome-codex-subagents](https://github.com/VoltAgent/awesome-codex-subagents), supplied by the project owner. The upstream repository is MIT-licensed; its license is included below. The instructions are shortened for this project, and the model settings use lower-cost Luna models to match the request for token-efficient, bounded delegation.

Codex does not automatically run these agents. They are available for explicit delegation in this project after the current Codex session reloads the project configuration. Use `context-manager` for a compact handoff, `quant-analyst` or `data-scientist` for model/data analysis, `ui-ux-tester` for interface review, and `reviewer` for a code review.

## Upstream license

MIT License

Copyright (c) 2026 VoltAgent

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

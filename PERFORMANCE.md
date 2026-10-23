# Dashboard performance

Only the selected report tab is rendered. All tabs, tables, ledgers, and provenance remain available, and selecting a tab never reruns the simulation. Reports are still created only by the Run button.

Price-frame and date-bound caches are bounded to 16 entries. Their keys include full file paths, nanosecond modification times, and sizes so explicit refreshes invalidate promptly. A single validation sample loads one ticker; an empty custom-strategy directory skips that read. The same validation gates run when custom files exist.

Altair/report-pipeline imports are deferred to their views/actions. The Yahoo client is imported only for downloads; the public `src.data.yf` alias remains available. Retries, validation, cache writes, fills, costs, strategies, and reporting formulas are unchanged. No automatic refresh is added.

Use Streamlit 1.60+ for state-tracked tabs. The full suite verifies the execution/validation contracts and added cases cover deferred tabs and immediate cache-file invalidation.

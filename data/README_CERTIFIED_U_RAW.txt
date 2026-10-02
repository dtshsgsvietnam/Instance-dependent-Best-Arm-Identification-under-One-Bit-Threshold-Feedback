The final certified-U stress run was executed on the separated five-arm instance
[0.90, 0.50, 0.30, 0.20, 0.10] with beta=0.5, deltas
{0.10, 0.05, 0.02, 0.01}, 200 replications, cap=200000,
refresh=decide=50, and the proof-certified U boundary.

All recorded rows were censored at the cap.  The raw server-side trials.csv
was not uploaded into the chat session used to assemble this handoff, so the
package contains the exact aggregate result in U_certified_easy5_summary.csv
but not the 800 raw delta-level rows.

To make the handoff fully archival, copy the original file into:
    data/U_certified_easy5_final.csv

The expected source run directory was named:
    U_CERTIFIED_EASY5_FINAL/

Do not synthesize the missing seeds/runtime columns.  Preserve the original
trials.csv if it is added later.

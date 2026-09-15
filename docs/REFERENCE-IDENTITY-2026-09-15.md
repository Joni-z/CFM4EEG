# Reference identity correction, 2026-09-15

User corrected a mistaken comparison: "paper duplex" means cf2_v1d192,
not the earlier paclock_duplex. Verified against live Overleaf sections/results.tex
and appendix.tex, AMD scripts/gen_tables.py mapping and targeted raw results.

Paper folded duplex cf2_v1d192:
- TUEV test kappa .6546, TWO seeds (0=.6804138482423736, 2=.6287428633331578).
- CHB test PR .6691 +/- .0413, three seeds, from current paper generated table.

Earlier tri-axial paclock_duplex:
- TUEV test kappa .6895 +/- .0357, three seeds.
- CHB test PR .6985 approximately, three seeds.

New nb16_carrier seed0:
- TUEV test kappa .7213771274730753; CHB test PR .6914489029764315.
This is numerically above the PAPER reference on both primary test metrics;
it is not replicated superiority and differs in architecture/recipe.
Do not call it useless merely because it misses the older CHB validation gate.
Do not retroactively change preregistered validation gates using these test scores.
All comparisons must name the exact configuration and seed count.

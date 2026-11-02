# Independent phone-photo collection

Goal: test the validation-selected model on a small, separately collected set with different backgrounds and lighting. Do not train or tune on this set.

1. Target 20 **distinct physical items per class**, 120 items total. Photograph each item once for the primary report. If you keep multiple views, track an item ID and report the dependency; they are not independent items.
2. Use the six original labels. Keep ambiguous mixed-material objects out of the primary six-class evaluation; log them separately for qualitative inspection. The benchmark's trash label is imperfect, so document how you assigned it.
3. Vary lighting and background naturally. Keep one prominent item per photo. Avoid creating only exceptionally easy examples or relabeling based on predictions.
4. Label the images before looking at model outputs. Record the collection date, background, lighting, material label and item ID in a separate CSV for your report. Avoid identifying people or private information in the frame.
5. Store files under `data/phone-photos/cardboard/`, `glass/`, `metal/`, `paper/`, `plastic/`, and `trash/`.
6. Run `python -m vg lock-challenge --images data/phone-photos`. The command requires all six categories, rejects exact duplicate photos and likely overlaps with the benchmark, and writes image hashes.
7. Run `python -m vg evaluate-challenge --images data/phone-photos`. Review the separate report under the selected run's `challenge-evaluation/` folder. Do not combine these scores with the benchmark test score.

If any failure guides model improvements, retire this collection as a held-out set, call it development data and collect a fresh evaluation set. A convenience sample of 120 items is informative but cannot establish performance for all locations, cameras or waste streams.

Current status: no independent phone photos have been supplied. The software supports evaluation; the actual study remains pending.

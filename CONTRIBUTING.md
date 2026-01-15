# Contributing to geograph-type-classifier

We welcome contributions to improve the accuracy of the Geograph Type predictions!

### How you can help:
1. **Label More 'From Drone' / 'From Above' Samples:** The model currently lacks sufficient data for low-level aerial shots. If you find images in the Geograph archive that fit these categories, please submit their IDs.
2. **Refining the 'Cross Far' Heuristic:** If you find cases where the model misidentifies distance perspective, let us know so we can adjust the distance-embedding logic.
3. **Pipeline Optimization:** Suggestions for faster JSONL parsing or more efficient CPU training are always welcome.

### Submission Process:
- For data additions: Open an Issue with a list of Geograph Image IDs and the suggested Tag.
- For code changes: Submit a Pull Request with a brief description of the improvement.

Together, we can build a more searchable and better-categorized Geograph archive.

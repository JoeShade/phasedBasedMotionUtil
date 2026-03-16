# Current Deviations from `systemDesign.md`

- The worker still reconstructs the full decoded clip in memory before phase processing instead of running a fully bounded chunk-to-numeric pipeline with reusable lossless chunk blocks. The decode subprocess itself is long-lived and bounded per read, but the later numeric path is still more memory-hungry than the final architecture target.

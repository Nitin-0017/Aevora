# NASA C-MAPSS FD001 acquisition

Verified 2026-10-23: NASA PCoE's [dataset repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
lists **6. Turbofan Engine Degradation Simulation** and links this
[archive](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip).
Use that entry, not the newer Simulation-2/N-CMAPSS collection.
If the link changes, return to NASA's page instead of substituting an unverified mirror.

Citation: A. Saxena and K. Goebel (2008), “Turbofan Engine Degradation Simulation
Data Set,” NASA Prognostics Data Repository, NASA Ames Research Center.
These are simulated engine trajectories, not measurements from a deployed factory.

## Manual acquisition

Download the NASA-linked archive in your browser, extract it (including any nested
CMAPSSData archive), and place these files under the configured raw data directory:

```text
data/raw/CMAPSSData/
  train_FD001.txt
  test_FD001.txt
  RUL_FD001.txt
```

`configs/default.toml` controls `raw_data_dir`. Relative paths resolve against
`project_root`, itself relative to the config file. Keep data under the ignored
`data/` directory, or outside the repository. Check `git check-ignore` before
using any custom in-repository location. Raw files must never be committed.

The trajectory schema is unit ID, cycle, three operating settings, and 21 sensors
(26 whitespace-separated numeric columns). Training histories end at failure;
test histories stop earlier. The target file has one remaining-life value per
test engine, in increasing unit-ID order. It contains cycles, not hours.

Phase 2 EDA uses training data only. Do not open official RUL targets for EDA.
Target parsing will be tested using temporary unit-test fixtures; those fixtures
are not substitutes for NASA data or evidence for empirical findings.

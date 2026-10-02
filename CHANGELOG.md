# Changelog

Changes to released versions of PolyProton are documented here.

## [Unreleased]

## [1.0.0] - 2026-10-02

Initial public release.

### Added

- Automated SPI monomer construction and conformer generation.
- MACE-OFF conformer optimization and energy-based selection.
- ORCA and Multiwfn calculations for RESP charge preparation.
- GAFF2 parameter preparation and polymer-chain construction.
- Fixed and mixed polymer-chain lengths.
- Packmol construction of dry, hydrated, and hydronium-containing systems.
- Amber MPI and CUDA execution with configurable computational settings.
- 6-step and 12-step dry equilibration protocols.
- Sequential hydration and conductivity simulation workflows.
- Structural, water-cluster, and hydronium transport analysis.
- Example CPU and GPU SLURM submission scripts.
- Documentation and BSD 3-Clause licensing.

### Known limitations

- Molecular construction is specialized for sulfonated polyimides.
- Automatic restart/resume support is not provided.
- New calculations should use clean working directories.
- Working and scratch paths should not contain spaces.
- Simulations use a fixed-charge GAFF2 description.
- Proton conductivity includes vehicular hydronium diffusion;
  explicit Grotthuss proton transfer is not represented.

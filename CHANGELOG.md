# Changelog

Changes to released versions of PolyProton are documented here.

## [Unreleased]

### Added

- PSMILES based polymer monomer unit definition using RDKit
- Support for explicit polymerization endpoint using `[*:1]` and `[*:2]`.
- Automatic detection of sulfonic acid groups from the polymer PSMILES.
- Automatic determination of protonated and deprotonated monomer unit charges.

### Changed
- Replaced the SPI-specific monomer assembly from separate dianhydride, benzene, 
and side-chain SMILES with a single PSMILES monomer unit definition.
- Polymer labels are now required to contain exactly three alphanumeric characters
because they are also used as PDB residue names.
- Polymer head and tail atoms are determined directly from the PSMILES polymerization
endpoints.
- Sulfonic acid deprotonation charge is determined automatically instead of using a
fixed `-2` repeat-unit charge.
- Hydration water counts are based on the number of detected sulfonic acid sites
rather than the total number of sulfur atoms.
- Conductivity-system hydronium counts are validated against the expected number of
sulfonic acid sites.
- - Amber residue charges used during PREPGEN are derived from the detected monomer unit chemistry

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

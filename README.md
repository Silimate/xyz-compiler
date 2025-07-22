# `xyz` boolean logic optimizer

## Basic Instructions

`xyz` builds into an executable via `cargo build —-release`. This `xyz-compiler` repository holds additional components that extend `prjunnamed`.

Please see here for an example of Yosys integration: https://github.com/povik/xyz-compiler/blob/main/yosys_flow/flow.tcl.

In this script an "xyz" procedure is defined which resembles the Yosys "abc" command. It iterates over modules and independently processes them.

Multipliers, adders, and sequential elements should pass through this procedure without mapping. Other logic elements get optimized and mapped to a basic CMOS library (https://github.com/povik/xyz-compiler/blob/main/yosys_flow/yosys_cmos.lib) which is translated to Yosys internal cells.

This is tested for logic equivalence on the EPFL combinational benchmark: https://github.com/lsils/benchmarks/tree/7770275a0e07a27b8ea9f65b6a3f767282fb8226

Depending on how you use the command you may run into these issues: https://gist.github.com/povik/9c2e4a8ceacce4898fe60cc4863b81b7

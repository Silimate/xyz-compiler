use bumpalo::Bump;
use prjunnamed_netlist::{Cell, ControlNet, Design, Net, Target, Value};
use std::{collections::BTreeMap, fs::File, io::BufWriter, io::Write, sync::Arc};
use xyz_compiler::target::SCLTarget;
use xyz_compiler::{cm, sm};

fn read_input(target: Option<Arc<dyn Target>>, filename: String) -> (Design, String) {
    if filename.ends_with(".uir") {
        return (
            prjunnamed_netlist::parse(target, &std::fs::read_to_string(filename).unwrap()).unwrap(),
            "top".to_owned(),
        );
    } else if filename.ends_with(".json") {
        let designs =
            prjunnamed_yosys_json::import(target, &mut File::open(filename).unwrap()).unwrap();
        assert_eq!(
            designs.len(),
            1,
            "On Yosys JSON import a single module is expected"
        );
        let top_name = designs.keys().cloned().next().unwrap();
        return (designs.into_values().next().unwrap(), top_name);
    } else {
        panic!("unrecognized file type: {filename}")
    }
}

fn write_output(mut design: Design, filename: String, top_name: String) {
    if let Some(target) = design.target() {
        target.export(&mut design);
    }

    let mut output = File::create(filename.clone()).unwrap();
    let stats = design.statistics();

    if filename.ends_with(".uir") {
        write!(output, "{design}").unwrap()
    } else if filename.ends_with(".json") {
        prjunnamed_yosys_json::export(&mut output, BTreeMap::from([(top_name, design)])).unwrap()
    } else if filename.ends_with(".v") {
        let alloc = Bump::new();
        let unit = prjunnamed_verilogout::export(&design, &alloc);
        write!(output, "{}", unit).unwrap()
    }

    eprintln!("Output cell statistics:");
    for (kind, amount) in stats {
        eprintln!("{:>7} {}", amount, kind);
    }
}

fn main() {
    let mut input_fn = String::new();
    let mut library_fn = String::new();
    let mut output_fn = String::new();
    let mut disable_dar_opt = false;
    let mut no_sequential_mapping = false;
    let mut no_lower_arith = false;

    {
        let mut parser = argparse::ArgumentParser::new();
        parser
            .refer(&mut input_fn)
            .add_argument("INPUT", argparse::Store, "Input file")
            .required();
        parser
            .refer(&mut library_fn)
            .add_argument("LIB", argparse::Store, "Library")
            .required();
        parser
            .refer(&mut output_fn)
            .add_argument("OUTPUT", argparse::Store, "Output file")
            .required();
        parser.refer(&mut disable_dar_opt).add_option(
            &["--disable-dar-opt"],
            argparse::StoreTrue,
            "Disable DAG-aware rewriting optimization",
        );
        parser.refer(&mut no_sequential_mapping).add_option(
            &["--no-sequential-mapping"],
            argparse::StoreTrue,
            "Disable technology mapping of sequentials",
        );
        parser.refer(&mut no_lower_arith).add_option(
            &["--no-lower-arith"],
            argparse::StoreTrue,
            "Disable lowering of arithmetic",
        );
        parser.parse_args_or_exit();
    }

    let target;
    {
        let mut target_staging = SCLTarget::new();
        target_staging.read_liberty(std::fs::read_to_string(library_fn).unwrap());
        target = Arc::new(target_staging);
    }

    let mut design;
    let top_name;
    (design, top_name) = read_input(Some(target.clone()), input_fn);
    if let Some(target) = design.target() {
        target.import(&mut design).unwrap();
    }
    prjunnamed_generic::unname(&mut design);

    eprintln!("Optimizing...");
    prjunnamed_generic::decision(&mut design);
    prjunnamed_generic::canonicalize(&mut design);

    if no_lower_arith {
        design.rewrite(&[
            &prjunnamed_generic::LowerLt,
            &prjunnamed_generic::LowerShift,
        ]);
    } else {
        design.rewrite(&[
            &prjunnamed_generic::LowerLt,
            &prjunnamed_generic::LowerMul,
            &prjunnamed_generic::LowerShift,
        ]);
    }

    // Fix up for limitation of Yosys export
    for cell_ref in design.iter_cells() {
        if let Cell::Dff(ff) = &*cell_ref.get() {
            let _guard = design.use_metadata_from(&[cell_ref]);
            if ff.has_clear() && ff.has_reset() {
                let mut ff = ff.clone();
                ff.unmap_reset(&design);
                cell_ref.replace(Cell::Dff(ff));
            }
        }
    }
    design.compact();

    design.rewrite(&[
        &prjunnamed_generic::LowerEq,
        &prjunnamed_generic::LowerMux,
        &prjunnamed_generic::SimpleAigOpt,
        &prjunnamed_generic::Normalize,
    ]);

    prjunnamed_generic::chain_rebalance(&mut design);
    prjunnamed_generic::canonicalize(&mut design);
    prjunnamed_generic::tree_rebalance(&mut design);

    if !disable_dar_opt {
        eprintln!("Optimizing (DAR)...");
        for _ in 0..3 {
            xyz_compiler::dar::rewrite(&design);
            design.compact();
            design.rewrite(&[
                &prjunnamed_generic::SimpleAigOpt,
                &prjunnamed_generic::Normalize,
            ]);
            design.compact();
        }
    }

    // lower xors ahead of technology mapping
    for cell in design.iter_cells() {
        let _guard = design.use_metadata_from(&[cell]);
        match &*cell.get() {
            Cell::Xor(arg1, arg2) => {
                design.replace_value(
                    cell.output(),
                    arg1.iter()
                        .zip(arg2.iter())
                        .map(|(net1, net2)| {
                            design.add_aig(
                                ControlNet::Neg(
                                    design.add_aig(ControlNet::Pos(net1), ControlNet::Pos(net2)),
                                ),
                                ControlNet::Neg(
                                    design.add_aig(ControlNet::Neg(net1), ControlNet::Neg(net2)),
                                ),
                            )
                        })
                        .collect::<Value>(),
                );
                cell.unalive();
            }
            _ => {}
        }
    }
    design.compact();

    eprintln!("Mapping...");
    if !no_sequential_mapping {
        eprintln!("Sequential..");
        sm::map(&design, target.clone());
        design.compact();
    }

    eprintln!("Combinational..");
    cm::map(&design, target.clone());
    design.compact();

    eprintln!("Writing result..");
    write_output(design, output_fn, top_name);
}

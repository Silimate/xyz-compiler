// sequential mapper
use crate::target::{parse_pin, LibraryCell, SCLTarget, Statement};
use prjunnamed_netlist::{Cell, Design, TargetCell, Trit, Value};
use std::{collections::HashMap, sync::Arc};

#[derive(Default, Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct FeatureSet {
    clocked: bool,
    clock_polarity: bool,
    has_enable: bool,
    enable_polarity: bool,
    has_reset: bool,
    reset_polarity: bool,
    has_clear: bool,
    clear_polarity: bool,
    has_set: bool,
    set_polarity: bool,
}

impl FeatureSet {
    fn without_enable(&self) -> FeatureSet {
        let mut ret = self.clone();
        ret.has_reset = false;
        ret.reset_polarity = false;
        ret
    }

    fn without_set(&self) -> FeatureSet {
        let mut ret = self.clone();
        ret.has_set = false;
        ret.set_polarity = false;
        ret
    }

    fn without_clear(&self) -> FeatureSet {
        let mut ret = self.clone();
        ret.has_clear = false;
        ret.clear_polarity = false;
        ret
    }

    fn without_reset(&self) -> FeatureSet {
        let mut ret = self.clone();
        ret.has_reset = false;
        ret.reset_polarity = false;
        ret
    }
}

#[derive(Default, Debug, Eq, PartialEq, Clone)]
pub(crate) struct PinPositions {
    clock_in: Option<usize>,
    data_in: Option<usize>,
    enable_in: Option<usize>,
    reset_in: Option<usize>,
    clear_in: Option<usize>,
    set_in: Option<usize>,
    pub(crate) data_out: Option<usize>,
    pub(crate) data_negated_out: Option<usize>,
}

impl PinPositions {
    fn no_described_inputs(&self) -> usize {
        (self.clock_in.is_some() as usize)
            + (self.data_in.is_some() as usize)
            + (self.enable_in.is_some() as usize)
            + (self.reset_in.is_some() as usize)
            + (self.clear_in.is_some() as usize)
            + (self.set_in.is_some() as usize)
    }
}

#[derive(Debug, Clone)]
pub struct MapTarget<'a> {
    cell: &'a LibraryCell,
    features: FeatureSet,
    pub(crate) pins: PinPositions,
}

impl<'a> MapTarget<'a> {
    fn instantiate(&self) -> TargetCell {
        let feats = &self.features;
        let mut ret = TargetCell::new(&self.cell.name, &self.cell.prototype);
        ret.inputs = Value::undef(self.cell.prototype.input_len);
        if feats.has_reset {
            ret.inputs[self.pins.reset_in.unwrap()] = (!feats.reset_polarity).into();
        }
        if feats.has_set {
            ret.inputs[self.pins.set_in.unwrap()] = (!feats.set_polarity).into();
        }
        if feats.has_clear {
            ret.inputs[self.pins.clear_in.unwrap()] = (!feats.clear_polarity).into();
        }
        if feats.has_enable {
            ret.inputs[self.pins.enable_in.unwrap()] = (feats.enable_polarity).into();
        }
        ret
    }
}

fn detect_cell<'a>(cell: &'a LibraryCell) -> Option<(FeatureSet, MapTarget<'a>)> {
    use Statement::{Group, SimpleAttr};

    let ff_ast = cell.ast.lookup("ff")?;
    let Group(_, ff_attrs, _) = ff_ast else {
        return None;
    };
    let [ff_pin_expr, ff_pin_negated_expr] = &ff_attrs[..] else {
        return None;
    };
    let mut feats = FeatureSet::default();
    let mut pins = PinPositions::default();

    // assert cell has single-bit inputs only
    for input in cell.prototype.inputs.iter() {
        if input.len() != 1 {
            return None;
        }
    }

    if let Some(clock_ast) = ff_ast.lookup("clocked_on") {
        let SimpleAttr(_, expr) = clock_ast else {
            return None;
        };
        let (negated, pin) = parse_pin(expr)?;
        pins.clock_in = Some(cell.prototype.inputs.iter().position(|p| p.name == pin)?);
        feats.clocked = true;
        feats.clock_polarity = !negated;
    } else {
        // clock is non-optional until we support latches
        return None;
    }

    if let Some(clear_ast) = ff_ast.lookup("clear") {
        let SimpleAttr(_, expr) = clear_ast else {
            return None;
        };
        let (negated, pin) = parse_pin(expr)?;
        pins.clear_in = Some(cell.prototype.inputs.iter().position(|p| p.name == pin)?);
        feats.has_clear = true;
        feats.clear_polarity = !negated;
    }
    if let Some(set_ast) = ff_ast.lookup("preset") {
        let SimpleAttr(_, expr) = set_ast else {
            return None;
        };
        let (negated, pin) = parse_pin(expr)?;
        pins.set_in = Some(cell.prototype.inputs.iter().position(|p| p.name == pin)?);
        feats.has_set = true;
        feats.set_polarity = !negated;
    }

    let next_state_ast = ff_ast.lookup("next_state")?;
    let SimpleAttr(_, expr) = next_state_ast else {
        return None;
    };
    let (negated, pin) = parse_pin(expr)?;
    if negated {
        return None;
    }
    pins.data_in = Some(cell.prototype.inputs.iter().position(|p| p.name == pin)?);

    pins.data_out = Some(cell.output_pin_asts.iter().position(|ast| {
        let Some(SimpleAttr(_, expr)) = ast.lookup("function") else {
            return false;
        };
        let Some((negated, pin)) = parse_pin(expr) else {
            return false;
        };
        return !negated && pin == *ff_pin_expr;
    })?);

    pins.data_negated_out = cell.output_pin_asts.iter().position(|ast| {
        let Some(SimpleAttr(_, expr)) = ast.lookup("function") else {
            return false;
        };
        let Some((negated, pin)) = parse_pin(expr) else {
            return false;
        };
        return (negated && pin == *ff_pin_expr) || (!negated && pin == *ff_pin_negated_expr);
    });

    // TODO: reset inference
    // TODO: enable inference

    if pins.no_described_inputs() != cell.prototype.input_len {
        return None;
    }

    Some((
        feats,
        MapTarget {
            cell: cell,
            pins: pins,
            features: feats,
        },
    ))
}

pub struct TargetIndex<'a> {
    classes: HashMap<FeatureSet, MapTarget<'a>>,
    pub(crate) per_cell: HashMap<String, MapTarget<'a>>,
}

impl<'a> MapTarget<'a> {
    fn beats(&self, other: Option<&MapTarget>) -> bool {
        other.is_none() || other.unwrap().cell.area > self.cell.area
    }
}

impl<'a> TargetIndex<'a> {
    pub fn create(library: &'a SCLTarget) -> Self {
        let mut per_cell: HashMap<String, MapTarget<'a>> = HashMap::new();
        let mut feature_classes: HashMap<FeatureSet, MapTarget<'a>> = HashMap::new();

        for (name, cell) in library.cells.iter() {
            let Some((feats, map_target)) = detect_cell(cell) else {
                continue;
            };

            per_cell.insert(name.to_string(), map_target.clone());
            if let Some(existing) = feature_classes.get(&feats) {
                if existing.cell.area < cell.area {
                    continue;
                }
            }
            feature_classes.insert(feats, map_target);
        }

        loop {
            let mut settled = true;
            for (feats, target) in feature_classes.clone().iter() {
                for lessened_feats in [
                    feats.without_enable(),
                    feats.without_reset(),
                    feats.without_set(),
                    feats.without_clear(),
                ] {
                    if target.beats(feature_classes.get(&lessened_feats)) {
                        settled = false;
                        feature_classes.insert(lessened_feats, target.clone());
                    }
                }
            }
            if settled {
                break;
            }
        }

        Self {
            per_cell: per_cell,
            classes: feature_classes,
        }
    }
}

pub fn map(design: &Design, target: Arc<SCLTarget>) {
    let index = TargetIndex::create(target.as_ref());

    for cell in design.iter_cells() {
        let Cell::Dff(flop_data) = &*cell.get() else {
            continue;
        };
        let mut flop_data = flop_data.clone();

        let _guard = design.use_metadata_from(&[cell]);

        if !flop_data.enable.is_always(true) {
            flop_data.unmap_enable(design, &cell.output());
        }

        if !flop_data.reset.is_always(false) {
            flop_data.unmap_reset(design);
        }

        for idx in 0..flop_data.data.len() {
            let slice = flop_data.slice(idx..idx + 1);

            let mut feats = FeatureSet::default();
            feats.clocked = true;
            feats.clock_polarity = slice.clock.is_positive();
            if !slice.clear.is_always(false) {
                if slice.clear_value[0] == Trit::Zero {
                    feats.has_clear = true;
                    feats.clear_polarity = slice.clear.is_positive();
                } else if slice.clear_value[0] == Trit::One {
                    feats.has_set = true;
                    feats.set_polarity = slice.clear.is_positive();
                }
            }

            let Some(target) = index.classes.get(&feats) else {
                panic!("unmappable cell: {}", design.display_cell(cell));
            };

            let mut target_cell = target.instantiate();
            if !slice.clear.is_always(false) {
                if slice.clear_value[0] == Trit::Zero {
                    target_cell.inputs[target.pins.clear_in.unwrap()] = slice.clear.net();
                } else if slice.clear_value[0] == Trit::One {
                    target_cell.inputs[target.pins.set_in.unwrap()] = slice.clear.net();
                }
            }

            target_cell.inputs[target.pins.clock_in.unwrap()] = slice.clock.net();
            target_cell.inputs[target.pins.data_in.unwrap()] = slice.data.unwrap_net();
            let out = design.add_target(target_cell);
            design.replace_net(cell.output()[idx], out[target.pins.data_out.unwrap()]);
        }

        cell.unalive();
    }
}

#[cfg(test)]
mod test {
    use crate::sm::{detect_cell, map, FeatureSet, PinPositions};
    use crate::target::SCLTarget;
    use prjunnamed_netlist::{assert_isomorphic, parse, ControlNet, Design, FlipFlop, Trit, Value};
    use std::sync::Arc;

    #[test]
    fn test_detect() {
        let mut target = SCLTarget::new();
        target.read_liberty(
            r#"
			library(small) {
				cell(ff) {
					area : 3;
					ff(IQ, IQN) {
						next_state : "D" ;
						clocked_on : "CLK" ;
					}
					pin(D) {
                        capacitance: 1.0;
						direction : input;
					}
					pin(CLK) {
                        capacitance: 1.0;
						direction : input;
					}
					pin(Q) {
                        capacitance: 1.0;
						direction : output;
						function : "IQ" ;
					}
				}
                cell(ff2) {
                    area : 3;
                    ff(IQ, IQN) {
                        next_state : "D" ;
                        clocked_on : "CLK'" ;
                    }
                    pin(D) {
                        capacitance: 1.0;
                        direction : input;
                    }
                    pin(CLK) {
                        capacitance: 1.0;
                        direction : input;
                    }
                    pin(Q) {
                        capacitance: 1.0;
                        direction : output;
                        function : "IQ" ;
                    }
                    pin(QN) {
                        capacitance: 1.0;
                        direction : output;
                        function : "IQN" ;
                    }
                }

                cell(ff3) {
                    area : 3;
                    ff(IQ, IQN) {
                        next_state : "D" ;
                        clocked_on : "CLK" ;
                        clear : "!RST_N" ;
                    }
                    pin(CLK) {
                        capacitance: 1.0;
                        direction : input;
                    }
                    pin(RST_N) {
                        capacitance: 1.0;
                        direction : input;
                    }
                    pin(D) {
                        capacitance: 1.0;
                        direction : input;
                    }
                    pin(Q) {
                        capacitance: 1.0;
                        direction : output;
                        function : "IQ" ;
                    }
                    pin(QN) {
                        capacitance: 1.0;
                        direction : output;
                        function : "IQN" ;
                    }
                }
			}"#
            .to_string(),
        );

        let detection = detect_cell(target.cells.get("ff").unwrap()).unwrap();
        assert_eq!(
            detection.0,
            FeatureSet {
                clocked: true,
                clock_polarity: true,
                has_enable: false,
                enable_polarity: false,
                has_reset: false,
                reset_polarity: false,
                has_clear: false,
                clear_polarity: false,
                has_set: false,
                set_polarity: false
            }
        );
        assert_eq!(
            detection.1.pins,
            PinPositions {
                clock_in: Some(1),
                data_in: Some(0),
                enable_in: None,
                reset_in: None,
                clear_in: None,
                set_in: None,
                data_out: Some(0),
                data_negated_out: None
            }
        );

        let detection = detect_cell(target.cells.get("ff2").unwrap()).unwrap();
        assert_eq!(
            detection.0,
            FeatureSet {
                clocked: true,
                clock_polarity: false,
                has_enable: false,
                enable_polarity: false,
                has_reset: false,
                reset_polarity: false,
                has_clear: false,
                clear_polarity: false,
                has_set: false,
                set_polarity: false
            }
        );
        assert_eq!(
            detection.1.pins,
            PinPositions {
                clock_in: Some(1),
                data_in: Some(0),
                enable_in: None,
                reset_in: None,
                clear_in: None,
                set_in: None,
                data_out: Some(0),
                data_negated_out: Some(1)
            }
        );

        let detection = detect_cell(target.cells.get("ff3").unwrap()).unwrap();
        assert_eq!(
            detection.0,
            FeatureSet {
                clocked: true,
                clock_polarity: true,
                has_enable: false,
                enable_polarity: false,
                has_reset: false,
                reset_polarity: false,
                has_clear: true,
                clear_polarity: false,
                has_set: false,
                set_polarity: false
            }
        );
        assert_eq!(
            detection.1.pins,
            PinPositions {
                clock_in: Some(0),
                data_in: Some(2),
                enable_in: None,
                reset_in: None,
                clear_in: Some(1),
                set_in: None,
                data_out: Some(0),
                data_negated_out: Some(1)
            }
        );
    }

    #[test]
    fn test_simple_map() {
        let mut target = SCLTarget::new();
        target.read_liberty(
            r#"
            library(small) {
                cell(ff) {
                    area : 3;
                    ff(IQ, IQN) {
                        next_state : "D" ;
                        clocked_on : "CLK" ;
                    }
                    pin(D) {
                        capacitance : 1.0;
                        direction : input;
                    }
                    pin(CLK) {
                        capacitance : 1.0;
                        direction : input;
                    }
                    pin(Q) {
                        capacitance : 1.0;
                        direction : output;
                        function : "IQ" ;
                    }
                }
                cell(ff4) {
                    area : 3;
                    ff(IQ, IQN) {
                        next_state : "D" ;
                        clocked_on : "CLK" ;
                        preset : "!RST_N" ;
                    }
                    pin(CLK) {
                        capacitance : 1.0;
                        direction : input;
                    }
                    pin(RST_N) {
                        capacitance : 1.0;
                        direction : input;
                    }
                    pin(D) {
                        capacitance : 1.0;
                        direction : input;
                    }
                    pin(Q) {
                        capacitance : 1.0;
                        direction : output;
                        function : "IQ" ;
                    }
                    pin(QN) {
                        capacitance : 1.0;
                        direction : output;
                        function : "IQN" ;
                    }
                }

                cell(ff3) {
                    area : 3;
                    ff(IQ, IQN) {
                        next_state : "D" ;
                        clocked_on : "CLK" ;
                        clear : "!RST_N" ;
                    }
                    pin(CLK) {
                        capacitance : 1.0;
                        direction : input;
                    }
                    pin(RST_N) {
                        capacitance : 1.0;
                        direction : input;
                    }
                    pin(D) {
                        capacitance : 1.0;
                        direction : input;
                    }
                    pin(Q) {
                        capacitance : 1.0;
                        direction : output;
                        function : "IQ" ;
                    }
                    pin(QN) {
                        capacitance : 1.0;
                        direction : output;
                        function : "IQN" ;
                    }
                }
            }"#
            .to_string(),
        );
        let target_finalized = Arc::new(target);
        let mut d = Design::with_target(Some(target_finalized.clone()));
        let a = d.add_input1("a");
        let c = d.add_input1("c");
        let clr = d.add_input1("clr");
        let set = d.add_input1("set");
        let ff1 =
            FlipFlop::new(Value::from(a), c).with_clear_value(ControlNet::Neg(clr), Trit::Zero);
        let ff2 =
            FlipFlop::new(Value::from(a), c).with_clear_value(ControlNet::Neg(set), Trit::One);
        let q1 = d.add_dff(ff1).unwrap_net();
        let q2 = d.add_dff(ff2).unwrap_net();
        d.add_output("q1", q1);
        d.add_output("q2", q2);
        d.apply();

        map(&d, target_finalized.clone());
        d.apply();

        let mut gold = parse(Some(target_finalized), {
            r#"
            %0:1 = input "a"
            %1:1 = input "c"
            %2:1 = input "clr"
            %3:1 = input "set"
            ; drives "q1"+0
            %8:_ = target "ff3" {
              input "CLK" = %1
              input "RST_N" = %2
              input "D" = %0
              %8:1 = output "Q"
              %9:1 = output "QN"
            }
            ; drives "q2"+0
            %10:_ = target "ff4" {
              input "CLK" = %1
              input "RST_N" = %3
              input "D" = %0
              %10:1 = output "Q"
              %11:1 = output "QN"
            }
            %6:0 = output "q1" %8
            %7:0 = output "q2" %10
        "#
        })
        .unwrap();
        assert_isomorphic!(d, gold);
    }

    #[test]
    fn test_demotion() {
        let mut target = SCLTarget::new();
        target.read_liberty(
            r#"
            library(small) {
                cell(ff3) {
                    area : 3;
                    ff(IQ, IQN) {
                        next_state : "D" ;
                        clocked_on : "CLK" ;
                        clear : "!RST_N" ;
                    }
                    pin(CLK) {
                        capacitance : 1.0;
                        direction : input;
                    }
                    pin(RST_N) {
                        capacitance : 1.0;
                        direction : input;
                    }
                    pin(D) {
                        capacitance : 1.0;
                        direction : input;
                    }
                    pin(Q) {
                        capacitance : 1.0;
                        direction : output;
                        function : "IQ" ;
                    }
                    pin(QN) {
                        capacitance : 1.0;
                        direction : output;
                        function : "IQN" ;
                    }
                }
            }"#
            .to_string(),
        );
        let target_finalized = Arc::new(target);
        let mut d = Design::with_target(Some(target_finalized.clone()));
        let a = d.add_input1("a");
        let clk = d.add_input1("clk");
        let ff = FlipFlop::new(Value::from(a), clk);
        d.add_output("q1", d.add_dff(ff).unwrap_net());
        d.apply();

        map(&d, target_finalized.clone());
        d.apply();

        let mut gold = parse(Some(target_finalized), {
            r#"
        %0:1 = input "a"
        %1:1 = input "clk"
        ; drives "q1"+0
        %4:_ = target "ff3" {
          input "CLK" = %1
          input "RST_N" = 1
          input "D" = %0
          %4:1 = output "Q"
          %5:1 = output "QN"
        }
        %3:0 = output "q1" %4
        "#
        })
        .unwrap();
        assert_isomorphic!(d, gold);
    }
}

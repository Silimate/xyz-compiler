// rewriting in the spirit of:
// https://people.eecs.berkeley.edu/~alanmi/publications/2006/dac06_rwr.pdf

use crate::npn::{npn_semiclass, npn_semiclass_allrepr, Truth6, NPN};
use prjunnamed_netlist::{Cell, CellRef, ControlNet, Design, Net};
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};

pub const CUT_MAXIMUM: usize = 5;
pub const NO_NODE: Net = Net::UNDEF;

#[derive(Debug)]
struct Structure {
    ninputs: usize,
    nodes: Vec<(i32, i32)>,
    hit: usize,
}

struct Index {
    structures: Vec<Structure>,
    classes: HashMap<(usize, Truth6), Vec<(NPN, usize)>>,
}

fn create_index(structures: Vec<Structure>) -> Index {
    let mut classes = HashMap::new();

    for (idx, struct_) in structures.iter().enumerate() {
        let ninputs = struct_.ninputs;
        let mask = mask6(ninputs);
        let mut states: Vec<Truth6> = vec![0];
        for i in 0..(ninputs) {
            states.push(COFACTOR_MASKS[i] & mask);
        }
        for (a, b) in struct_.nodes.iter() {
            states.push(
                (if *a < 0 {
                    states[-*a as usize] ^ mask
                } else {
                    states[*a as usize]
                }) & (if *b < 0 {
                    states[-*b as usize] ^ mask
                } else {
                    states[*b as usize]
                }),
            );
        }
        let function = *states.last().unwrap();
        // IMPROVE ME: prune out structurally equivalent graphs
        npn_semiclass_allrepr(function, ninputs, &mut |map: &NPN| {
            classes
                .entry((ninputs, map.apply(function)))
                .or_insert(Vec::new())
                .push((map.inv(), idx));
        });
    }
    Index {
        structures,
        classes,
    }
}

#[derive(Clone)]
struct Cut {
    leaves: [Net; CUT_MAXIMUM],
    function: Truth6,
}

fn cut_from_slice(slice: &[Net]) -> [Net; CUT_MAXIMUM] {
    let mut ret = [NO_NODE; CUT_MAXIMUM];
    for (idx, net) in slice.iter().copied().enumerate() {
        ret[idx] = net;
    }
    ret
}

fn cut_slice(array: &[Net; CUT_MAXIMUM]) -> &[Net] {
    // see if the cut ends early
    for i in 0..CUT_MAXIMUM {
        if array[i] == NO_NODE {
            return &array[..i];
        }
    }
    // otherwise return all of the array
    &array[..]
}

fn mask6(size: usize) -> Truth6 {
    if size == 6 {
        0xffffffffffffffff
    } else {
        (1 << (1 << size)) - 1
    }
}

fn recode6(t1: Truth6, vars1: &[Net], vars2: &[Net]) -> Truth6 {
    let mut t2: Truth6 = 0;

    for i2 in 0..(1 << vars2.len()) {
        let mut i1 = 0;
        let mut var2i = 0;
        for (var1i, var1) in vars1.iter().copied().enumerate() {
            while var2i < vars2.len() && vars2[var2i] < var1 {
                var2i += 1;
            }

            if var2i == vars2.len() {
                break;
            }

            if i2 & 1 << var2i != 0 {
                i1 |= 1 << var1i;
            }
        }
        if t1 & 1 << i1 != 0 {
            t2 |= 1 << i2;
        }
    }

    t2
}

const COFACTOR_MASKS: [Truth6; 6] = [
    0xaaaaaaaaaaaaaaaa,
    0xcccccccccccccccc,
    0xf0f0f0f0f0f0f0f0,
    0xff00ff00ff00ff00,
    0xffff0000ffff0000,
    0xffffffff00000000,
];

fn check_support6(t: Truth6, var_idx: usize) -> bool {
    let (t_shift, _) = t.overflowing_shl(1 << var_idx);
    ((t_shift ^ t) & COFACTOR_MASKS[var_idx]) != 0
}

struct NwUnderRewrite<'a> {
    structure_index: &'a mut Index,
    design: &'a Design,
    use_counts: RefCell<HashMap<Net, i32>>,
    structural_index: HashMap<(ControlNet, ControlNet), Net>,
    visited: HashSet<Net>,
    cuts: HashMap<Net, Vec<Cut>>,
}

fn cut_union(
    target: &mut [Net],
    cut_len: &mut usize,
    max_cut: usize,
    cut1: &[Net],
    cut2: &[Net],
) -> bool {
    *cut_len = 0;
    let mut j: usize = 0;
    for idx1 in cut1.iter().copied() {
        while j < cut2.len() && cut2[j] < idx1 {
            if *cut_len == max_cut {
                return false;
            }
            target[*cut_len] = cut2[j];
            *cut_len += 1;
            j += 1;
        }

        if j < cut2.len() && cut2[j] == idx1 {
            j += 1;
        }

        if *cut_len == max_cut {
            return false;
        }

        target[*cut_len] = idx1;
        *cut_len += 1;
    }

    while j < cut2.len() {
        if *cut_len == max_cut {
            return false;
        }
        target[*cut_len] = cut2[j];
        *cut_len += 1;
        j += 1;
    }

    return true;
}

enum NodeFunction {
    And,
    Xor,
}

impl<'a> NwUnderRewrite<'a> {
    fn init(structure_index: &'a mut Index, design: &'a Design) -> Self {
        let mut use_counts: HashMap<Net, i32> = HashMap::new();
        let mut structural_index: HashMap<(ControlNet, ControlNet), Net> = HashMap::new();

        for cell in design.iter_cells() {
            match &*cell.get() {
                Cell::Not(_) => {
                    // nots are transparent
                }
                Cell::Aig(arg1, arg2) => {
                    let mut in1 = decode_cnet(design, *arg1);
                    let mut in2 = decode_cnet(design, *arg2);
                    *use_counts.entry(in1.net()).or_insert(0) += 1;
                    *use_counts.entry(in2.net()).or_insert(0) += 1;
                    if in1 > in2 {
                        (in1, in2) = (in2, in1);
                    }
                    let key = (in1, in2);
                    let out = cell.output().unwrap_net();
                    if !structural_index.contains_key(&key) {
                        structural_index.insert(key, out);
                    }
                }
                _ => cell.visit(|net| {
                    let cnet = decode_cnet(design, ControlNet::Pos(net));
                    *use_counts.entry(cnet.net()).or_insert(0) += 1;
                }),
            }
        }

        NwUnderRewrite {
            structure_index,
            design,
            use_counts: use_counts.into(),
            structural_index,
            cuts: HashMap::new(),
            visited: HashSet::new(),
        }
    }

    fn unref(&self, head: Net, leaves: &[Net]) -> usize {
        let Ok((cell, _idx)) = self.design.find_cell(head) else {
            return 0;
        };

        let in1;
        let in2;
        match &*cell.get() {
            Cell::Aig(arg1, arg2) => {
                in1 = decode_cnet(self.design, *arg1);
                in2 = decode_cnet(self.design, *arg2);
            }
            Cell::Xor(arg1, arg2) if arg1.len() == 1 => {
                in1 = decode_cnet(self.design, ControlNet::Pos(arg1.unwrap_net()));
                in2 = decode_cnet(self.design, ControlNet::Pos(arg2.unwrap_net()));
            }
            _ => {
                return 0;
            }
        }
        let mut nnodes = 1;
        for fanin in &[in1.net(), in2.net()] {
            if {
                let mut use_counts = self.use_counts.borrow_mut();
                let nrefs = use_counts.get_mut(fanin).unwrap();
                *nrefs -= 1;
                assert!(*nrefs >= 0);
                *nrefs == 0
            } {
                if !leaves.binary_search(fanin).is_ok() {
                    nnodes += self.unref(*fanin, leaves);
                }
            }
        }
        nnodes
    }

    fn ref_(&self, head: Net, leaves: &[Net]) {
        let Ok((cell, _idx)) = self.design.find_cell(head) else {
            return;
        };

        let in1;
        let in2;
        match &*cell.get() {
            Cell::Aig(arg1, arg2) => {
                in1 = decode_cnet(self.design, *arg1);
                in2 = decode_cnet(self.design, *arg2);
            }
            Cell::Xor(arg1, arg2) if arg1.len() == 1 => {
                in1 = decode_cnet(self.design, ControlNet::Pos(arg1.unwrap_net()));
                in2 = decode_cnet(self.design, ControlNet::Pos(arg2.unwrap_net()));
            }
            _ => {
                return;
            }
        }

        for fanin in &[in1.net(), in2.net()] {
            if {
                let mut use_counts = self.use_counts.borrow_mut();
                let nrefs = use_counts.get_mut(fanin).unwrap();
                *nrefs += 1;
                assert!(*nrefs >= 1);
                *nrefs == 1
            } {
                if !leaves.binary_search(fanin).is_ok() {
                    self.ref_(*fanin, leaves);
                }
            }
        }
    }

    fn collect_source(&self, head: Net, stopping_set: &[Net]) -> Vec<CellRef<'_>> {
        let mut queue: Vec<Net> = vec![head];
        let mut seen: BTreeSet<Net> = BTreeSet::new();
        seen.insert(head);
        let mut ret: Vec<CellRef> = Vec::new();

        let mut idx: usize = 0;
        while idx < queue.len() {
            let net = queue[idx];
            let cell = self.design.find_cell(net).ok().unwrap().0;
            ret.push(cell);
            let in1;
            let in2;
            match &*cell.get() {
                Cell::Aig(arg1, arg2) => {
                    in1 = decode_cnet(self.design, *arg1);
                    in2 = decode_cnet(self.design, *arg2);
                }
                Cell::Xor(arg1, arg2) if arg1.len() == 1 => {
                    in1 = decode_cnet(self.design, ControlNet::Pos(arg1.unwrap_net()));
                    in2 = decode_cnet(self.design, ControlNet::Pos(arg2.unwrap_net()));
                }
                _ => {
                    unreachable!();
                }
            }

            for inet in &[in1.net(), in2.net()] {
                if inet.is_const()
                    || stopping_set.binary_search(&inet).is_ok()
                    || seen.contains(&inet)
                {
                    continue;
                }
                queue.push(*inet);
                seen.insert(*inet);
            }
            idx += 1;
        }
        ret
    }

    fn rewrite_inner(&mut self, head: Net, a: ControlNet, b: ControlNet, function: NodeFunction) {
        let mut out_cuts: Vec<Cut> = vec![];
        let empty: Vec<Cut> = vec![];
        let cuts_a = self.cuts.get(&a.net()).unwrap_or(&empty);
        let cuts_b = self.cuts.get(&b.net()).unwrap_or(&empty);

        struct Candidate {
            delta: i32,
            cut_len: usize,
            cut: [Net; CUT_MAXIMUM],
            structural_cut: [Net; CUT_MAXIMUM],
            struct_idx: usize,
            map: NPN,
        }
        let mut best: Option<Candidate> = None;

        for ia in (-1)..(cuts_a.len() as i32) {
            'next_cut: for ib in (-1)..(cuts_b.len() as i32) {
                let cut1 = if ia == -1 {
                    &Cut {
                        leaves: cut_from_slice(&[a.net()]),
                        function: 2,
                    }
                } else {
                    cuts_a.get(ia as usize).unwrap()
                };
                let cut2 = if ib == -1 {
                    &Cut {
                        leaves: cut_from_slice(&[b.net()]),
                        function: 2,
                    }
                } else {
                    cuts_b.get(ib as usize).unwrap()
                };

                let mut scratch = [NO_NODE; CUT_MAXIMUM];
                let mut cut_len = 0;
                if !cut_union(
                    &mut scratch[..],
                    &mut cut_len,
                    CUT_MAXIMUM,
                    cut_slice(&cut1.leaves),
                    cut_slice(&cut2.leaves),
                ) {
                    continue;
                }

                let structural_cut = &scratch[..cut_len];
                let mut f1 = recode6(cut1.function, cut_slice(&cut1.leaves), structural_cut);
                if a.is_negative() {
                    f1 ^= mask6(structural_cut.len());
                };
                let mut f2 = recode6(cut2.function, cut_slice(&cut2.leaves), structural_cut);
                if b.is_negative() {
                    f2 ^= mask6(structural_cut.len());
                }

                let mut f = match function {
                    NodeFunction::And => f1 & f2,
                    NodeFunction::Xor => f1 ^ f2,
                };

                if structural_cut.len() <= 4 {
                    out_cuts.push(Cut {
                        leaves: cut_from_slice(structural_cut),
                        function: f,
                    });
                }

                // Remove variables which are not in the support of the function
                let mut cut_backing = [NO_NODE; CUT_MAXIMUM];
                let mut cut_len = 0;
                for idx in 0..structural_cut.len() {
                    if check_support6(f, idx) {
                        if cut_len == CUT_MAXIMUM {
                            continue 'next_cut;
                        }
                        cut_backing[cut_len] = structural_cut[idx];
                        cut_len += 1;
                    }
                }
                let cut = &cut_backing[..cut_len];
                if cut_len < structural_cut.len() {
                    f = recode6(f, structural_cut, cut);
                }

                if cut_len > 4 {
                    continue 'next_cut;
                }

                let current_weight = self.unref(head, structural_cut);

                let map = npn_semiclass(f, cut_len);
                if let Some(structs) = self.structure_index.classes.get(&(cut_len, map.apply(f))) {
                    for (map2, struct_idx) in structs {
                        let mut repl_weight = 0;
                        // mapping from structure inputs to `cut`
                        let struct_to_cut = (map2 * &map).inv();
                        let struct_ = &self.structure_index.structures[*struct_idx];
                        let mut nets: Vec<Option<ControlNet>> = (0..cut_len)
                            .map(|idx| {
                                Some(ControlNet::from_net_invert(
                                    cut[struct_to_cut.p[idx as usize] as usize],
                                    struct_to_cut.ic[idx as usize],
                                ))
                            })
                            .collect();
                        for (a, b) in struct_.nodes.iter() {
                            let anet = if *a < 0 {
                                nets[-*a as usize - 1].map(|cn| !cn)
                            } else {
                                nets[*a as usize - 1]
                            };
                            let bnet = if *b < 0 {
                                nets[-*b as usize - 1].map(|cn| !cn)
                            } else {
                                nets[*b as usize - 1]
                            };
                            let (Some(anet2), Some(bnet2)) = (anet, bnet) else {
                                repl_weight += 1;
                                nets.push(None);
                                continue;
                            };
                            let (in1, in2) = if anet2 > bnet2 {
                                (bnet2, anet2)
                            } else {
                                (anet2, bnet2)
                            };
                            let Some(out) = self.structural_index.get(&(in1, in2)) else {
                                repl_weight += 1;
                                nets.push(None);
                                continue;
                            };
                            let use_counts = self.use_counts.borrow();
                            if use_counts.get(&out).copied().unwrap_or(0) == 0 || head == *out {
                                repl_weight += 1;
                                nets.push(None);
                                continue;
                            }
                            nets.push(Some(ControlNet::Pos(*out)));
                        }

                        let delta = current_weight as i32 - repl_weight as i32;
                        if delta > 0
                            && (best.is_none()
                                || ((delta > best.as_ref().unwrap().delta)
                                    || (delta == best.as_ref().unwrap().delta)
                                        && (cut_len < best.as_ref().unwrap().cut_len)))
                        {
                            best = Some(Candidate {
                                delta,
                                cut_len,
                                cut: cut_backing,
                                structural_cut: scratch,
                                struct_idx: *struct_idx,
                                map: struct_to_cut,
                            });
                        }
                    }
                }
                self.ref_(head, structural_cut);
            }
        }

        if !best.is_none() {
            let candidate = best.unwrap();
            let source_cells: Vec<CellRef> =
                self.collect_source(head, cut_slice(&candidate.structural_cut));
            let _guard = self.design.use_metadata_from(source_cells.as_slice());

            self.unref(head, &candidate.structural_cut);
            let struct_ = &mut self.structure_index.structures[candidate.struct_idx];
            struct_.hit += candidate.delta as usize;
            let mut nets: Vec<ControlNet> = (0..struct_.ninputs)
                .map(|idx| {
                    ControlNet::from_net_invert(
                        candidate.cut[candidate.map.p[idx as usize] as usize],
                        candidate.map.ic[idx as usize],
                    )
                })
                .collect();

            for (a, b) in struct_.nodes.iter() {
                let anet = if *a < 0 {
                    !nets[-*a as usize - 1]
                } else {
                    nets[*a as usize - 1]
                };
                let bnet = if *b < 0 {
                    !nets[-*b as usize - 1]
                } else {
                    nets[*b as usize - 1]
                };
                let (in1, in2) = if anet > bnet {
                    (bnet, anet)
                } else {
                    (anet, bnet)
                };
                let mut use_counts = self.use_counts.borrow_mut();
                let node_net: Net;
                if let Some(out) = self.structural_index.get(&(in1, in2)) {
                    if use_counts.get(&out).copied().unwrap_or(0) != 0 {
                        node_net = *out;
                    } else {
                        *use_counts.entry(in1.net()).or_insert(0) += 1;
                        *use_counts.entry(in2.net()).or_insert(0) += 1;
                        node_net = self.design.add_aig(in1, in2);
                    }
                } else {
                    *use_counts.entry(in1.net()).or_insert(0) += 1;
                    *use_counts.entry(in2.net()).or_insert(0) += 1;
                    node_net = self.design.add_aig(in1, in2);
                }
                self.structural_index.insert((in1, in2), node_net);
                nets.push(ControlNet::Pos(node_net));
            }

            let mut new_head = *nets.last().unwrap();
            {
                let mut use_counts = self.use_counts.borrow_mut();
                *use_counts.entry(new_head.net()).or_insert(0) += 1;
            }
            self.structural_index.remove(&(a, b));

            if candidate.map.oc {
                new_head = !new_head;
            }
            assert!(head != new_head.net());
            let new_net = new_head.into_pos(&self.design);
            self.design.replace_net(head, new_net);

            // clean up references if any cut leaves are now fully unused
            for node in candidate.structural_cut {
                if {
                    let use_counts = self.use_counts.borrow();
                    use_counts.get(&node).copied().unwrap_or(0) == 0
                } {
                    self.unref(node, &[]);
                }
            }
        } else {
            self.cuts.insert(head, out_cuts);
        }
    }

    fn rewrite(&mut self, head: Net) {
        if self.visited.contains(&head) {
            return;
        }
        let Ok((cell, _idx)) = self.design.find_cell(head) else {
            self.visited.insert(head);
            return;
        };

        let mut in1;
        let mut in2;
        let node_function: NodeFunction;
        match &*cell.get() {
            Cell::Aig(arg1, arg2) => {
                in1 = decode_cnet(self.design, *arg1);
                in2 = decode_cnet(self.design, *arg2);
                node_function = NodeFunction::And;
            }
            Cell::Xor(arg1, arg2) if arg1.len() == 1 => {
                in1 = decode_cnet(self.design, ControlNet::Pos(arg1.unwrap_net()));
                in2 = decode_cnet(self.design, ControlNet::Pos(arg2.unwrap_net()));
                node_function = NodeFunction::Xor;
            }
            _ => {
                self.visited.insert(head);
                return;
            }
        }
        if in1 > in2 {
            (in1, in2) = (in2, in1);
        }
        self.rewrite(in1.net());
        self.rewrite(in2.net());
        self.rewrite_inner(head, in1, in2, node_function);
        self.visited.insert(head);
    }

    fn pass(&mut self) {
        for cell in self.design.iter_cells() {
            match &*cell.get() {
                Cell::Aig(_, _) | Cell::Xor(_, _) => {
                    if {
                        let use_counts = self.use_counts.borrow_mut();
                        use_counts
                            .get(&cell.output().unwrap_net())
                            .copied()
                            .unwrap_or(0)
                            != 0
                    } {
                        self.rewrite(cell.output().unwrap_net())
                    }
                }
                _ => {}
            }
        }
    }
}

fn decode_cnet(design: &Design, mut cnet: ControlNet) -> ControlNet {
    loop {
        let Ok((cell, idx)) = design.find_cell(cnet.net()) else {
            return cnet;
        };

        match &*cell.get() {
            Cell::Not(arg) => cnet = ControlNet::from_net_invert(arg[idx], !cnet.is_negative()),
            _ => return cnet,
        }
    }
}

pub fn rewrite(design: &Design) {
    let mut index = create_index(vec![
        Structure {
            ninputs: 1,
            nodes: vec![],
            hit: 0,
        },
        Structure {
            ninputs: 2,
            nodes: vec![(1, 2)],
            hit: 0,
        },
        Structure {
            ninputs: 3,
            nodes: vec![(1, 2), (3, 4)],
            hit: 0,
        },
        Structure {
            ninputs: 3,
            nodes: vec![(1, 2), (3, -4)],
            hit: 0,
        },
        Structure {
            ninputs: 4,
            nodes: vec![(-1, -2), (1, 2), (6, 3), (-5, -7), (-8, 4)],
            hit: 0,
        },
        Structure {
            ninputs: 4,
            nodes: vec![(1, -2), (3, 2), (-6, 4), (-5, 7)],
            hit: 0,
        },
        Structure {
            ninputs: 4,
            nodes: vec![(1, 2), (-5, 3), (-6, 4)],
            hit: 0,
        },
        Structure {
            ninputs: 4,
            nodes: vec![(1, 2), (-5, 3), (4, 2), (-6, -7)],
            hit: 0,
        },
        Structure {
            ninputs: 4,
            nodes: vec![(1, 2), (-5, 3), (6, 4)],
            hit: 0,
        },
        Structure {
            ninputs: 4,
            nodes: vec![(1, 2), (3, 4), (-5, -6)],
            hit: 0,
        },
        Structure {
            ninputs: 4,
            nodes: vec![(1, 2), (3, 4), (-5, 6), (-1, -2), (-7, -8)],
            hit: 0,
        },
        Structure {
            ninputs: 4,
            nodes: vec![(1, 2), (3, 4), (-5, 6)],
            hit: 0,
        },
        Structure {
            ninputs: 4,
            nodes: vec![(1, 2), (3, 4), (5, 6)],
            hit: 0,
        },
        Structure {
            ninputs: 4,
            nodes: vec![(1, 2), (5, 3), (-6, 4)],
            hit: 0,
        },
        Structure {
            ninputs: 4,
            nodes: vec![(1, 2), (5, 3), (4, -2), (-6, -7)],
            hit: 0,
        },
        Structure {
            ninputs: 4,
            nodes: vec![(1, 2), (5, 3), (6, 4)],
            hit: 0,
        },
        Structure {
            ninputs: 4,
            nodes: vec![(1, 2), (-5, 3), (5, 4), (-6, -7)],
            hit: 0,
        },
    ]);

    let mut nur = NwUnderRewrite::init(&mut index, design);
    nur.pass();
}

#[cfg(test)]
mod test {
    use crate::dar::rewrite;
    use prjunnamed_netlist::ControlNet::{Neg, Pos};
    use prjunnamed_netlist::{assert_isomorphic, parse, Design};

    #[test]
    fn test_simple() {
        let mut d = Design::with_target(None);
        let i0 = d.add_input1("i0");
        let i1 = d.add_input1("i1");
        let pn = d.add_aig(Pos(i0), Neg(i1));
        let np = d.add_aig(Neg(i0), Pos(i1));
        let nn = d.add_aig(Neg(i0), Neg(i1));
        let y = d.add_aig(Neg(pn), Pos(d.add_aig(Neg(np), Neg(nn))));
        d.add_output("y0", y);
        d.apply();
        rewrite(&d);
    }

    #[test]
    fn test_simple2() {
        let mut d = Design::with_target(None);
        let i0 = d.add_input1("i0");
        let i1 = d.add_input1("i1");
        let i2 = d.add_input1("i2");
        let i01 = d.add_aig(Pos(i0), Pos(i1));
        let i12 = d.add_aig(Pos(i1), Pos(i2));
        let i02 = d.add_aig(Pos(i0), Pos(i2));
        let y0 = d.add_not1(d.add_aig(Neg(i01), Pos(d.add_aig(Neg(i12), Neg(i02)))));
        d.add_output("y0", y0);
        d.apply();
        rewrite(&d);
    }

    #[test]
    fn test_unmap_xor() {
        let mut d = Design::with_target(None);
        let i0 = d.add_input1("i0");
        let i1 = d.add_input1("i1");
        let i01 = d.add_aig(Pos(i0), Pos(i1));
        d.add_output("y0", d.add_xor(i01, i1));
        d.apply();
        rewrite(&d);
        d.apply();
        let mut gold = parse(None, {
            r#"
			%0:1 = input "i0"
			%1:1 = input "i1"
			%2:1 = aig %1 !%0
			%3:0 = output "y0" %2
        "#
        })
        .unwrap();
        assert_isomorphic!(d, gold);
    }

    #[test]
    fn test_unmap_xor2() {
        let mut d = Design::with_target(None);
        let i0 = d.add_input1("i0");
        let i1 = d.add_input1("i1");
        d.add_output(
            "y0",
            d.add_xor(d.add_aig(Pos(i0), Neg(i1)), d.add_aig(Pos(i0), Pos(i1))),
        );
        d.apply();
        rewrite(&d);
        d.apply();
        let mut gold = parse(None, {
            r#"
			%0:1 = input "i0"
			%1:1 = input "i1"
			%2:0 = output "y0" %0
        "#
        })
        .unwrap();
        assert_isomorphic!(d, gold);
    }
}

// Tolerate non-snake-case variable naming for the sake of the `C` variable
#![allow(non_snake_case)]

// combinational mapper
use crate::npn::{npn_semiclass, npn_semiclass_allrepr, Truth6, NPN};
use crate::sm;
use crate::target::{LibraryCell, LibraryPort, SCLTarget};
use bumpalo::Bump;
use prjunnamed_netlist::{
    Cell, CellRef, ControlNet, Design, MetaItem, MetaItemRef, Net, TargetCell, Value,
};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::slice;
use std::sync::Arc;
use MetaItem::{IndexedScope, NamedScope};

struct MapTarget<'a> {
    cell: &'a LibraryCell,
    via: NPN,
    output: usize,
}

struct TargetIndex<'a> {
    classes: HashMap<(usize, Truth6), Vec<MapTarget<'a>>>,
    inverter: &'a LibraryCell,
    tie_lo: (&'a LibraryCell, usize),
    tie_hi: (&'a LibraryCell, usize),
    sm_index: sm::TargetIndex<'a>,
}

impl<'a> TargetIndex<'a> {
    fn create(library: &'a SCLTarget) -> Self {
        let mut ret = HashMap::new();
        let mut inverter: Option<&'a LibraryCell> = None;
        let mut tie_lo: Option<(&'a LibraryCell, usize)> = None;
        let mut tie_hi: Option<(&'a LibraryCell, usize)> = None;
        let mut tie_hilo: Option<(&'a LibraryCell, usize, usize)> = None;
        for (_name, cell) in library.cells.iter() {
            let cm_functions = &cell
                .output_ports
                .iter()
                .map(|port| port.cm_function)
                .collect::<Vec<_>>()[..];
            if let [Some(function)] = cm_functions {
                let ninputs = cell.prototype.inputs.len();
                npn_semiclass_allrepr(*function, ninputs, &mut |map: &NPN| {
                    let target = MapTarget {
                        cell,
                        via: map.inv(),
                        output: 0,
                    };
                    ret.entry((ninputs, map.apply(*function)))
                        .or_insert(Vec::new())
                        .push(target);
                });

                if cell.prototype.inputs.len() == 1 && *function == 1 {
                    if inverter.is_none() || cell.area < inverter.unwrap().area {
                        inverter = Some(cell);
                    }
                }
            }

            if cell.prototype.inputs.is_empty() {
                match cm_functions {
                    [Some(0), Some(1)] => {
                        if tie_hilo.is_none() || cell.area < tie_hilo.unwrap().0.area {
                            tie_hilo = Some((cell, 1, 0));
                        }
                    }
                    [Some(1), Some(0)] => {
                        if tie_hilo.is_none() || cell.area < tie_hilo.unwrap().0.area {
                            tie_hilo = Some((cell, 0, 1));
                        }
                    }
                    [Some(1)] => {
                        if tie_hi.is_none() || cell.area < tie_hi.unwrap().0.area {
                            tie_hi = Some((cell, 0));
                        }
                    }
                    [Some(0)] => {
                        if tie_lo.is_none() || cell.area < tie_lo.unwrap().0.area {
                            tie_lo = Some((cell, 0));
                        }
                    }
                    _ => {}
                }
            }
        }

        if tie_hilo.is_some()
            && (tie_lo.is_none()
                || tie_hi.is_none()
                || tie_lo.unwrap().0.area + tie_hi.unwrap().0.area > tie_hilo.unwrap().0.area)
        {
            let (cell, hi_pin, lo_pin) = tie_hilo.unwrap();
            tie_hi = Some((cell, hi_pin));
            tie_lo = Some((cell, lo_pin));
            ret.entry((0, 1)).or_insert(Vec::new()).push(MapTarget {
                cell,
                via: NPN::identity(0).with_co(),
                output: lo_pin,
            });
            ret.entry((0, 1)).or_insert(Vec::new()).push(MapTarget {
                cell,
                via: NPN::identity(0),
                output: hi_pin,
            });
        }

        for (_semiclass, targets) in ret.iter_mut() {
            targets.sort_by(|a, b| {
                (a.via.c_fingerprint(), a.cell.area)
                    .partial_cmp(&(b.via.c_fingerprint(), b.cell.area))
                    .unwrap()
            });
            targets.dedup_by_key(|t| t.via.c_fingerprint());
        }

        Self {
            classes: ret,
            inverter: inverter.unwrap(),
            tie_hi: tie_hi.unwrap(),
            tie_lo: tie_lo.unwrap(),
            sm_index: sm::TargetIndex::create(library),
        }
    }
}

type FrontierSlot = u32;
type Area = f32;

pub const CUT_MAXIMUM: usize = 6;
pub const NO_NODE: Net = Net::UNDEF;

#[derive(Clone, Debug)]
struct Match {
    semiclass: Truth6,
    npn: NPN,
    cut: [Net; CUT_MAXIMUM],
    eliminated_support: Option<Box<Vec<Net>>>,
}

impl Default for Match {
    fn default() -> Self {
        Self {
            semiclass: 0,
            npn: NPN::empty(),
            cut: [NO_NODE; CUT_MAXIMUM],
            eliminated_support: None,
        }
    }
}

#[derive(Default, Copy, Clone)]
struct NodePolarity<'a> {
    map_fouts: i32,
    flow_fouts: f32,
    farea: f32,
    selected: Option<(&'a Match, &'a MapTarget<'a>)>,
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum NodeRole {
    Pi(bool), // Pi: argument signifies complement is free
    Guts,
}
use crate::cm::NodeRole::{Guts, Pi};

struct Node<'a> {
    role: NodeRole,
    fanouts: u32,
    matches: &'a [Match],
    fid: FrontierSlot,
    pol: [NodePolarity<'a>; 2],
    visited: bool,
}

impl<'a> Node<'a> {
    fn pi() -> Self {
        Self {
            role: Pi(false),
            fanouts: 0,
            matches: &[],
            fid: 0,
            pol: [NodePolarity::default(); 2],
            visited: false,
        }
    }

    fn pi_with_free_complement() -> Self {
        Self {
            role: Pi(true),
            fanouts: 0,
            matches: &[],
            fid: 0,
            pol: [NodePolarity::default(); 2],
            visited: false,
        }
    }
    fn guts() -> Self {
        Self {
            role: Guts,
            fanouts: 0,
            matches: &[],
            fid: 0,
            pol: [NodePolarity::default(); 2],
            visited: false,
        }
    }
}

pub struct Mapping<'a> {
    design: &'a Design,
    alloc: &'a Bump,
    nodes: HashMap<Net, Node<'a>>,
    pos: Vec<ControlNet>,
    negated_po_fixups: Vec<(Net, Net)>,
    inverter_area: Area,
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

// TODO: invertor of invertor?
fn remove_invertor(sm_index: &sm::TargetIndex, design: &Design, net: Net) -> ControlNet {
    if let Ok((cell, idx)) = design.find_cell(net) {
        match &*cell.get() {
            Cell::Not(arg) => return ControlNet::Neg(arg[idx]),
            Cell::Target(target) => {
                if let Some(indexed_flop) = sm_index.per_cell.get(&target.kind) {
                    if Some(idx) == indexed_flop.pins.data_negated_out {
                        let data_out = indexed_flop.pins.data_out.unwrap();
                        return ControlNet::Neg(cell.output()[data_out]);
                    }
                }
            }
            _ => {}
        }
    }
    ControlNet::Pos(net)
}

enum NodeFunction {
    And,
    Nand,
    Xor,
    Mux,
}

fn decode_node(
    sm_index: &sm::TargetIndex,
    design: &Design,
    net: Net,
) -> (NodeFunction, Vec<ControlNet>) {
    let Ok((cell, idx)) = design.find_cell(net) else {
        panic!("cell not found");
    };
    match &*cell.get() {
        Cell::And(arg1, arg2) => {
            let net1 = remove_invertor(sm_index, design, arg1[idx]);
            let net2 = remove_invertor(sm_index, design, arg2[idx]);
            return (NodeFunction::And, vec![net1, net2]);
        }
        Cell::Aig(arg1, arg2) => {
            let mut net1 = remove_invertor(sm_index, design, arg1.net());
            if arg1.is_negative() {
                net1 = !net1;
            }
            let mut net2 = remove_invertor(sm_index, design, arg2.net());
            if arg2.is_negative() {
                net2 = !net2;
            }
            return (NodeFunction::And, vec![net1, net2]);
        }
        Cell::Or(arg1, arg2) => {
            let net1 = remove_invertor(sm_index, design, arg1[idx]);
            let net2 = remove_invertor(sm_index, design, arg2[idx]);
            return (NodeFunction::Nand, vec![!net1, !net2]);
        }
        Cell::Xor(arg1, arg2) => {
            let net1 = remove_invertor(sm_index, design, arg1[idx]);
            let net2 = remove_invertor(sm_index, design, arg2[idx]);
            return (NodeFunction::Xor, vec![net1, net2]);
        }
        Cell::Mux(s, arg1, arg2) => {
            let net1 = remove_invertor(sm_index, design, *s);
            let net2 = remove_invertor(sm_index, design, arg1[idx]);
            let net3 = remove_invertor(sm_index, design, arg2[idx]);
            return (NodeFunction::Mux, vec![net1, net2, net3]);
        }
        _ => {
            panic!("unsupported cell: {}", design.display_cell(cell));
        }
    }
}

fn scope_level(item: MetaItemRef) -> usize {
    if !item.is_none() {
        scope_level(item.scope_parent().unwrap()) + 1
    } else {
        0
    }
}

fn common_scope<'b>(mut scope1: MetaItemRef<'b>, mut scope2: MetaItemRef<'b>) -> MetaItemRef<'b> {
    let mut level1 = scope_level(scope1);
    let mut level2 = scope_level(scope2);
    while level1 > level2 {
        scope1 = scope1.scope_parent().unwrap();
        level1 -= 1;
    }
    while level2 > level1 {
        scope2 = scope2.scope_parent().unwrap();
        level2 -= 1;
    }
    while scope1 != scope2 {
        println!("{:?} XXX {:?}", scope1, scope2);
        scope1 = scope1.scope_parent().unwrap();
        scope2 = scope2.scope_parent().unwrap();
    }
    scope1
}

impl<'a> Mapping<'a> {
    fn init(design: &'a Design, bump_alloc: &'a Bump, target_index: &'a TargetIndex<'a>) -> Self {
        let mut nodes: HashMap<Net, Node<'a>> = HashMap::new();
        let mut po_nets: BTreeSet<ControlNet> = BTreeSet::new();
        let mut negated_po_fixups: BTreeSet<(Net, Net)> = BTreeSet::new();
        let sm_index = &target_index.sm_index;

        for cell in design.iter_cells_topo() {
            match &*cell.get() {
                Cell::Input(_, _) => {
                    for net in cell.output() {
                        nodes.insert(net, Node::pi());
                    }
                }
                // TODO: make mapping of name TFI optional
                Cell::Output(_, _) | Cell::Name(_, _) => {
                    cell.visit(|net| {
                        let cnet = remove_invertor(&sm_index, design, net);
                        po_nets.insert(cnet);
                        if let ControlNet::Neg(from) = cnet {
                            negated_po_fixups.insert((from, net));
                        }
                    });
                }
                Cell::And(_, _)
                | Cell::Aig(_, _)
                | Cell::Or(_, _)
                | Cell::Mux(_, _, _)
                | Cell::Xor(_, _) => {
                    for net in cell.output() {
                        nodes.insert(net, Node::guts());
                    }
                }
                Cell::Not(_) => {}
                Cell::Target(target_cell) => {
                    // all inputs become POs
                    for net in target_cell.inputs.iter() {
                        let cnet = remove_invertor(&sm_index, design, net);
                        po_nets.insert(cnet);
                        if let ControlNet::Neg(from) = cnet {
                            negated_po_fixups.insert((from, net));
                        }
                    }

                    // outputs become PIs
                    if let Some(indexed_flop) = sm_index.per_cell.get(&target_cell.kind) {
                        for (idx, net) in cell.output().iter().enumerate() {
                            if Some(idx) == indexed_flop.pins.data_negated_out {
                                // negated data out is never introduced as a PI,
                                // all uses of it should have been converted to
                                // a complement of the ordinary data out by
                                // the remove_invertor helper
                            } else if Some(idx) == indexed_flop.pins.data_out
                                && indexed_flop.pins.data_negated_out.is_some()
                            {
                                nodes.insert(net, Node::pi_with_free_complement());
                            } else {
                                nodes.insert(net, Node::pi());
                            }
                        }
                    } else {
                        for net in cell.output() {
                            nodes.insert(net, Node::pi());
                        }
                    }
                }
                Cell::Debug(_, _) => {}
                _ => {
                    // all inputs become POs
                    cell.visit(|net| {
                        let cnet = remove_invertor(&sm_index, design, net);
                        po_nets.insert(cnet);
                        if let ControlNet::Neg(from) = cnet {
                            negated_po_fixups.insert((from, net));
                        }
                    });

                    // outputs become PIs
                    for net in cell.output() {
                        nodes.insert(net, Node::pi());
                    }
                }
            }
        }

        let mut mapping = Mapping {
            design: design,
            nodes: nodes,
            alloc: bump_alloc,
            pos: po_nets.iter().copied().collect::<Vec<ControlNet>>(),
            negated_po_fixups: negated_po_fixups
                .iter()
                .copied()
                .collect::<Vec<(Net, Net)>>(),
            inverter_area: target_index.inverter.area,
        };

        mapping.prepare_matches(target_index, 64, 16, 4);
        mapping.index_nodes(target_index);
        mapping
    }

    fn index_nodes(&mut self, target_index: &TargetIndex) -> () {
        // TODO: rewrite once we move away from HashMap<Net, ...>
        let mut nrefs: HashMap<Net, u32> = HashMap::new();
        for (net, node) in self.nodes.iter_mut() {
            if let Guts = node.role {
                let (_, args) = decode_node(&target_index.sm_index, self.design, *net);
                for arg in args.iter() {
                    *nrefs.entry(arg.net()).or_insert(0) += 1;
                }
            }
            node.fanouts = 0;
        }
        for po in self.pos.iter() {
            // skew in favor of pos
            *nrefs.entry(po.net()).or_insert(0) += 100;
        }
        for (net, node) in self.nodes.iter_mut() {
            node.fanouts = nrefs.get(net).copied().unwrap_or(0);
        }
    }

    fn frontier(&mut self, target_index: &TargetIndex) -> usize {
        let mut size: usize = 1;
        let mut free_indices: Vec<FrontierSlot> = vec![];

        for node in self.nodes.values_mut() {
            node.visited = false;
            node.fid = 0;
        }

        for cell in self.design.iter_cells_topo().rev() {
            for net in cell.output().iter() {
                let Some(node) = self.nodes.get_mut(&net) else {
                    continue;
                };
                let fanins = match node.role {
                    Pi(_) => vec![],
                    Guts => {
                        let (_, args) = decode_node(&target_index.sm_index, self.design, net);
                        args
                    }
                };

                let node_fid = node.fid;
                node.visited = true;

                for fanin in fanins {
                    let Some(fanin_node) = self.nodes.get_mut(&fanin.net()) else {
                        continue;
                    };
                    assert!(!fanin_node.visited);
                    if fanin_node.fid == 0 {
                        fanin_node.fid = free_indices.pop().unwrap_or_else(|| {
                            size += 1;
                            (size - 1) as FrontierSlot
                        });
                    }
                }

                if node_fid != 0 {
                    free_indices.push(node_fid);
                }
            }
        }

        size
    }

    fn prepare_matches(
        &mut self,
        target_index: &TargetIndex,
        npriority_cuts: usize,
        nmatches_max: usize,
        max_cut: usize,
    ) {
        // TODO: error messages
        assert!(max_cut >= 3 && max_cut <= CUT_MAXIMUM);
        assert!((0..512).contains(&nmatches_max));
        assert!((0..65536).contains(&npriority_cuts));

        #[derive(Clone)]
        struct PriorityCut {
            cut: [Net; CUT_MAXIMUM],
            function: Truth6,
            eliminated_support: Option<Box<Vec<Net>>>,
        }

        #[derive(Clone)]
        struct NodeCache {
            ps_len: usize,
            mark: Net,
        }
        struct ImmediateFanin<'a> {
            net: Net,
            complement: bool,
            cut_ptr: i32,
            pcuts: &'a [PriorityCut],
        }

        let mut nsaturated = 0;

        fn combine_cuts(
            fanins: &mut Vec<ImmediateFanin>,
            max_cut: usize,
            mut F: impl FnMut(&[Net], &[Truth6], &mut Vec<Net>) -> bool,
        ) -> bool {
            let mut first = true;
            let mut eliminated: Vec<Net> = vec![];

            'next_choice: loop {
                if !first {
                    for (idx, fanin) in fanins.iter_mut().enumerate().rev() {
                        if fanin.cut_ptr < (fanin.pcuts.len() as i32) - 1 {
                            fanin.cut_ptr += 1;
                            break;
                        } else {
                            if idx == 0 {
                                break 'next_choice;
                            }
                            fanin.cut_ptr = -1;
                            continue;
                        }
                    }
                } else {
                    first = false;
                }

                let mut function = [0 as Truth6; CUT_MAXIMUM];
                // TODO: there's an edge case where we exhaust this due to constant inputs
                let mut scratch_backing = [NO_NODE; CUT_MAXIMUM * CUT_MAXIMUM];
                let mut scratch = &mut scratch_backing[..];
                let mut cut: &[Net] = &[];
                for fanin in fanins.iter() {
                    let pointee_cut = if fanin.cut_ptr == -1 {
                        match fanin.net {
                            Net::ONE | Net::ZERO | Net::UNDEF => &[] as &[Net],
                            _ => slice::from_ref(&fanin.net),
                        }
                    } else {
                        cut_slice(&fanin.pcuts[fanin.cut_ptr as usize].cut)
                    };

                    if cut.is_empty() {
                        cut = pointee_cut;
                    } else {
                        let mut cut_len = 0;
                        if !cut_union(
                            &mut scratch[..CUT_MAXIMUM],
                            &mut cut_len,
                            max_cut,
                            cut,
                            pointee_cut,
                        ) {
                            continue 'next_choice;
                        } else {
                            (cut, scratch) = scratch.split_at_mut(cut_len);
                        }
                    }
                }

                eliminated.clear();
                for (idx, fanin) in fanins.iter().enumerate() {
                    let (pointee_cut, mut cut_function) = if fanin.cut_ptr == -1 {
                        match fanin.net {
                            Net::ONE => (&[] as &[Net], 1),
                            Net::ZERO | Net::UNDEF => (&[] as &[Net], 0),
                            _ => (slice::from_ref(&fanin.net), 2),
                        }
                    } else {
                        let pcut = &fanin.pcuts[fanin.cut_ptr as usize];

                        if let Some(elim_nodes) = &pcut.eliminated_support {
                            eliminated.extend(elim_nodes.iter());
                        }

                        (cut_slice(&pcut.cut), pcut.function)
                    };

                    if fanin.complement {
                        cut_function ^= mask6(pointee_cut.len());
                    }

                    function[idx] = recode6(cut_function, pointee_cut, cut);
                }

                if F(cut, &function[..fanins.len()], &mut eliminated) {
                    return true;
                }
            }
            return false;
        }

        let mut pcuts_all: Vec<PriorityCut> = Vec::new();
        let mut cache_all: Vec<NodeCache> = Vec::new();
        let frontier_size = self.frontier(target_index);
        pcuts_all.resize(
            npriority_cuts * frontier_size,
            PriorityCut {
                cut: [NO_NODE; CUT_MAXIMUM],
                function: 0,
                eliminated_support: None,
            },
        );
        cache_all.resize(
            frontier_size,
            NodeCache {
                ps_len: 0,
                mark: NO_NODE,
            },
        );

        let mut matches_buffer: &mut [Match] = self
            .alloc
            .alloc(std::array::from_fn::<_, 2048, _>(|_| Match::default()));

        for cell in self.design.iter_cells_topo() {
            for net in cell.output().iter() {
                let Some(node) = self.nodes.get(&net) else {
                    continue;
                };

                match node.role {
                    Pi(_) => {
                        let cache = &mut cache_all[node.fid as usize];
                        cache.ps_len = 0;
                        cache.mark = net;
                        continue;
                    }
                    Guts => {
                        // fallthrough
                    }
                }

                if matches_buffer.len() < nmatches_max {
                    matches_buffer = self
                        .alloc
                        .alloc(std::array::from_fn::<_, 2048, _>(|_| Match::default()));
                }

                let (pcuts_below, pcuts_all2) =
                    (&mut pcuts_all[..]).split_at_mut((node.fid as usize) * npriority_cuts);
                let (pcuts, pcuts_above) = pcuts_all2.split_at_mut(npriority_cuts);

                // TODO: revisit inverter handling once we allow mapping subset of design
                let (node_function, args) = decode_node(&target_index.sm_index, self.design, net);
                let mut fanins: Vec<ImmediateFanin> = args
                    .iter()
                    .map(|arg| {
                        if arg.net().is_const() {
                            return ImmediateFanin {
                                net: arg.canonicalize().net(),
                                complement: false,
                                cut_ptr: -1,
                                pcuts: &[] as &[PriorityCut],
                            };
                        }
                        let Some(fanin_node) = self.nodes.get(&arg.net()) else {
                            unreachable!();
                        };
                        let fanin_cache = &cache_all[fanin_node.fid as usize];
                        assert!(fanin_cache.mark == arg.net());
                        assert!(fanin_node.fid != node.fid);
                        ImmediateFanin {
                            net: arg.net(),
                            complement: arg.is_negative(),
                            cut_ptr: -1,
                            pcuts: if fanin_node.fid < node.fid {
                                let base = (fanin_node.fid as usize) * npriority_cuts;
                                &pcuts_below[base..base + (fanin_cache.ps_len as usize)]
                            } else {
                                let base =
                                    (fanin_node.fid - node.fid - 1) as usize * npriority_cuts;
                                &pcuts_above[base..base + (fanin_cache.ps_len as usize)]
                            },
                        }
                    })
                    .collect();

                let mut seen_cuts: HashSet<Vec<Net>> = HashSet::new();
                let mut ps_slot = 0;
                let mut match_slot = 0;
                combine_cuts(
                    &mut fanins,
                    max_cut,
                    &mut |structural_cut: &[Net],
                          fanin_functions: &[Truth6],
                          eliminated: &mut Vec<Net>| {
                        let mut cut_function = {
                            let f = fanin_functions;
                            match node_function {
                                NodeFunction::And => f[0] & f[1],
                                NodeFunction::Nand => !(f[0] & f[1]) & mask6(structural_cut.len()),
                                NodeFunction::Xor => f[0] ^ f[1],
                                NodeFunction::Mux => (!f[0] & f[1]) | (f[0] & f[2]),
                            }
                        };

                        // Remove variables which are not in the support of the function
                        let mut cut_backing = [NO_NODE; 6];
                        let mut cut_len = 0;
                        for idx in 0..structural_cut.len() {
                            if check_support6(cut_function, idx) {
                                cut_backing[cut_len] = structural_cut[idx];
                                cut_len += 1;
                            } else {
                                eliminated.push(structural_cut[idx]);
                            }
                        }
                        let cut = &cut_backing[..cut_len];
                        if cut_len < structural_cut.len() {
                            cut_function = recode6(cut_function, structural_cut, cut);
                        }

                        if seen_cuts.contains(cut) {
                            return false;
                        }
                        seen_cuts.insert(cut.to_vec());

                        let npn = npn_semiclass(cut_function, cut.len());
                        let semiclass = npn.apply(cut_function);

                        if let Some(_) = target_index.classes.get(&(cut.len(), semiclass)) {
                            if match_slot < nmatches_max {
                                matches_buffer[match_slot] = Match {
                                    semiclass: semiclass,
                                    npn: npn,
                                    cut: cut_from_slice(cut),
                                    eliminated_support: if !eliminated.is_empty() {
                                        Some(Box::new((*eliminated.clone()).to_vec()))
                                    } else {
                                        None
                                    },
                                };
                                match_slot += 1;
                            }
                        }

                        if ps_slot == npriority_cuts {
                            nsaturated += 1;
                            return false;
                        }

                        pcuts[ps_slot].cut = cut_from_slice(cut);
                        pcuts[ps_slot].function = cut_function;
                        pcuts[ps_slot].eliminated_support = if !eliminated.is_empty() {
                            Some(Box::new((*eliminated.clone()).to_vec()))
                        } else {
                            None
                        };
                        ps_slot += 1;
                        return false;
                    },
                );

                cache_all[node.fid as usize] = NodeCache {
                    mark: net,
                    ps_len: ps_slot,
                };
                (self.nodes.get_mut(&net).unwrap().matches, matches_buffer) =
                    matches_buffer.split_at_mut(match_slot);
            }
        }
    }

    fn ref_cut(&mut self, net: Net, C: bool) -> Area {
        assert!(net != NO_NODE);

        let (cut, local_map) = {
            let node = self.nodes.get(&net).unwrap();
            if let Pi(_) = node.role {
                return 0.0;
            }

            let pol = &node.pol[C as usize];
            let (match_, target) = pol.selected.unwrap();
            (match_.cut, &target.via * &match_.npn)
        };

        let mut sum: Area = 0.0;
        for (n, cut_net) in cut_slice(&cut).iter().copied().enumerate() {
            let cut_node = self.nodes.get_mut(&cut_net).unwrap();
            let cut_nodeC = local_map.ic[n];
            assert!(cut_net != net);
            let cut_pol = &mut cut_node.pol[cut_nodeC as usize];
            cut_pol.map_fouts += 1;
            let mut descend = false;

            if cut_pol.map_fouts == 1 {
                match cut_node.role {
                    Pi(false) if cut_nodeC => {
                        sum += self.inverter_area;
                    }
                    Pi(_) => {}
                    Guts => {
                        sum += cut_pol.selected.unwrap().1.cell.area;
                        descend = true;
                    }
                }
            }

            assert!(cut_pol.map_fouts >= 1);

            if descend {
                sum += self.ref_cut(cut_net, cut_nodeC);
            }
        }
        sum
    }

    fn deref_cut(&mut self, net: Net, C: bool) {
        assert!(net != NO_NODE);

        let (cut, local_map) = {
            let node = self.nodes.get(&net).unwrap();
            if let Pi(_) = node.role {
                return;
            }

            let pol = &node.pol[C as usize];
            let (match_, target) = pol.selected.unwrap();
            (match_.cut, &target.via * &match_.npn)
        };

        for (n, cut_net) in cut_slice(&cut).iter().copied().enumerate() {
            let cut_node = self.nodes.get_mut(&cut_net).unwrap();
            let cut_nodeC = local_map.ic[n];
            assert!(cut_net != net);
            let cut_pol = &mut cut_node.pol[cut_nodeC as usize];
            cut_pol.map_fouts -= 1;
            let mut descend = false;

            if cut_pol.map_fouts == 0 {
                if !matches!(cut_node.role, Pi(_)) {
                    descend = true;
                }
            }

            assert!(cut_pol.map_fouts >= 0);
            if descend {
                self.deref_cut(cut_net, cut_nodeC);
            }
        }
    }

    fn deref_node(&mut self, net: Net, C: bool) {
        if net.is_const() {
            return;
        }

        let node = self.nodes.get_mut(&net).unwrap();
        let pol = &mut node.pol[C as usize];
        assert!(pol.map_fouts >= 1);
        pol.map_fouts -= 1;
        if pol.map_fouts == 0 {
            match node.role {
                Guts => {
                    self.deref_cut(net, C);
                }
                _ => {}
            }
        }
    }

    fn ref_node(&mut self, net: Net, C: bool) -> Area {
        if net.is_const() {
            return 0.0;
        }

        let node = self.nodes.get_mut(&net).unwrap();
        let pol = &mut node.pol[C as usize];
        pol.map_fouts += 1;
        if pol.map_fouts == 1 {
            match node.role {
                Guts => {
                    let (_match, target) = pol.selected.unwrap();
                    self.ref_cut(net, C) + target.cell.area
                }
                Pi(false) if C => self.inverter_area,
                Pi(_) => 0.0,
            }
        } else {
            0.0
        }
    }

    fn walk_selection(&mut self, initial: bool) -> Area {
        if !initial {
            for idx in 0..self.pos.len() {
                let po = self.pos[idx];
                self.deref_node(po.net(), po.is_negative());
            }
        }

        for node in self.nodes.values() {
            assert_eq!(node.pol[0].map_fouts, 0);
            assert_eq!(node.pol[1].map_fouts, 0);
        }

        let mut area = 0.0;
        for idx in 0..self.pos.len() {
            let po = self.pos[idx];
            area += self.ref_node(po.net(), po.is_negative());
        }
        area
    }

    fn area_flow_round(&mut self, target_index: &'a TargetIndex, refs_blend: f32) {
        for node in self.nodes.values_mut() {
            for pol in node.pol.iter_mut() {
                let blended_nrefs =
                    refs_blend * (node.fanouts as f32) + (1.0 - refs_blend) * pol.map_fouts as f32;
                pol.flow_fouts = if blended_nrefs >= 1.0 {
                    blended_nrefs
                } else {
                    1.0
                };
            }
        }

        for cell in self.design.iter_cells_topo() {
            for net in cell.output().iter() {
                match self.nodes.get(&net).map(|node| node.role) {
                    None => {
                        continue;
                    }
                    Some(Pi(complement_free)) => {
                        let node = self.nodes.get_mut(&net).unwrap();
                        node.pol[0].farea = 0.0;
                        node.pol[1].farea = if complement_free {
                            0.0
                        } else {
                            self.inverter_area / node.pol[1].flow_fouts
                        };
                        continue;
                    }
                    Some(Guts) => {
                        // fallthrough
                    }
                }

                for C in 0..2 {
                    let matches = self.nodes[&net].matches;
                    let mut best_area: Area = f32::MAX;
                    let mut best: Option<(&'a Match, &'a MapTarget<'a>)> = None;
                    for match_ in matches {
                        let semiclass = match_.semiclass;
                        let match_npn = match_.npn.clone();
                        'next_target: for target in target_index
                            .classes
                            .get(&(match_.npn.ninputs(), semiclass))
                            .unwrap()
                        {
                            let local_map = &target.via * &match_npn;
                            if local_map.oc != (C != 0) {
                                // this doesn't come out to the polarity we are interested in,
                                // skip
                                continue;
                            }

                            let mut area: Area = target.cell.area;
                            for (j, cut_net) in cut_slice(&match_.cut).iter().copied().enumerate() {
                                let cut_node = &self.nodes[&cut_net];
                                let pol = &cut_node.pol[local_map.ic[j] as usize];
                                if !matches!(cut_node.role, Pi(_)) && pol.selected.is_none() {
                                    // this node lacks a selected match, it is not availabe
                                    continue 'next_target;
                                }
                                area += pol.farea;
                            }

                            if area > 1.0e20 {
                                area = 1.0e20;
                            }

                            if area < best_area {
                                best_area = area;
                                best = Some((match_, target));
                            }
                        }
                    }

                    let pol = &self.nodes[&net].pol[C];
                    if pol.map_fouts != 0
                        && !(std::ptr::eq(pol.selected.unwrap().0, best.unwrap().0)
                            && std::ptr::eq(pol.selected.unwrap().1, best.unwrap().1))
                    {
                        self.deref_cut(net, C != 0);
                        self.nodes.get_mut(&net).unwrap().pol[C].selected = best;
                        self.ref_cut(net, C != 0);
                    } else {
                        self.nodes.get_mut(&net).unwrap().pol[C].selected = best;
                    }

                    let pol = &mut self.nodes.get_mut(&net).unwrap().pol[C];
                    pol.farea = best_area / pol.flow_fouts;
                }
            }
        }
    }

    fn exact_round(&mut self, target_index: &'a TargetIndex) {
        for cell in self.design.iter_cells_topo() {
            for net in cell.output().iter() {
                if !matches!(self.nodes.get(&net).map(|node| node.role), Some(Guts)) {
                    continue;
                }

                for C in 0..2 {
                    let matches = self.nodes[&net].matches;
                    if self.nodes.get(&net).unwrap().pol[C].map_fouts != 0 {
                        self.deref_cut(net, C != 0);
                    }
                    let mut best_area: Area = f32::MAX;
                    let mut best: Option<(&'a Match, &'a MapTarget<'a>)> = None;
                    for match_ in matches {
                        let semiclass = match_.semiclass;
                        let match_npn = match_.npn.clone();
                        'next_target: for target in target_index
                            .classes
                            .get(&(match_.npn.ninputs(), semiclass))
                            .unwrap()
                        {
                            let local_map = &target.via * &match_npn;
                            if local_map.oc != (C != 0) {
                                // this doesn't come out to the polarity we are interested in,
                                // skip
                                continue;
                            }

                            for (j, cut_net) in cut_slice(&match_.cut).iter().copied().enumerate() {
                                let cut_node = &self.nodes[&cut_net];
                                let pol = &cut_node.pol[local_map.ic[j] as usize];
                                if !matches!(cut_node.role, Pi(_)) && pol.selected.is_none() {
                                    // this node lacks a selected match, it is not availabe
                                    continue 'next_target;
                                }
                            }

                            self.nodes.get_mut(&net).unwrap().pol[C].selected =
                                Some((match_, target));
                            let mut area = target.cell.area + self.ref_cut(net, C != 0);

                            if area > 1.0e20 {
                                area = 1.0e20;
                            }

                            if area < best_area {
                                best_area = area;
                                best = Some((match_, target));
                            }
                            self.deref_cut(net, C != 0);
                        }
                    }

                    self.nodes.get_mut(&net).unwrap().pol[C].selected = best;
                    if self.nodes.get(&net).unwrap().pol[C].map_fouts != 0 {
                        self.ref_cut(net, C != 0);
                    }
                }
            }
        }
    }

    fn collect_source(
        &self,
        index: &'a TargetIndex,
        head: Net,
        pol: &NodePolarity,
    ) -> Vec<CellRef<'_>> {
        let mut stopping_set = cut_slice(&pol.selected.unwrap().0.cut);

        let mut stopping_set2: Vec<Net>;
        if let Some(eliminated) = &pol.selected.unwrap().0.eliminated_support {
            stopping_set2 = vec![];
            stopping_set2.extend(cut_slice(&pol.selected.unwrap().0.cut));
            stopping_set2.extend(eliminated.iter());
            stopping_set2.sort();
            stopping_set2.dedup();
            stopping_set = &stopping_set2[..];
        }

        let mut queue: Vec<Net> = vec![head];
        let mut seen: BTreeSet<Net> = BTreeSet::new();
        seen.insert(head);
        let mut ret: Vec<CellRef> = Vec::new();

        let mut idx: usize = 0;
        while idx < queue.len() {
            let net = queue[idx];
            ret.push(self.design.find_cell(net).ok().unwrap().0);
            let (_, fanins) = decode_node(&index.sm_index, self.design, net);

            for inet in fanins.iter().map(|cn| cn.net()) {
                if inet.is_const()
                    || stopping_set.binary_search(&inet).is_ok()
                    || seen.contains(&inet)
                {
                    continue;
                }
                queue.push(inet);
                seen.insert(inet);
            }
            idx += 1;
        }
        ret
    }

    fn reintegrate(&mut self, index: &'a TargetIndex) {
        let mut complements: HashMap<Net, Net> = HashMap::new();

        let top_scope = self
            .design
            .iter_cells()
            .map(|cell| {
                cell.metadata()
                    .iter()
                    .filter_map(|item_ref| match item_ref.get() {
                        NamedScope { .. } | IndexedScope { .. } => Some(item_ref),
                        MetaItem::Ident { scope, .. } => Some(scope),
                        _ => None,
                    })
            })
            .flatten()
            .reduce(common_scope)
            .unwrap_or(MetaItemRef::new_none(self.design));

        {
            let out_lo: Net;
            let out_hi: Net;
            if std::ptr::eq(index.tie_lo.0, index.tie_hi.0) {
                let tie_lo = TargetCell::new(&index.tie_lo.0.name, &index.tie_lo.0.prototype);
                let out = self.design.add_target(tie_lo);
                out_lo = out[index.tie_lo.1];
                out_hi = out[index.tie_hi.1];
            } else {
                let tie_lo = TargetCell::new(&index.tie_lo.0.name, &index.tie_lo.0.prototype);
                out_lo = self.design.add_target(tie_lo)[index.tie_lo.1];
                let tie_hi = TargetCell::new(&index.tie_hi.0.name, &index.tie_hi.0.prototype);
                out_hi = self.design.add_target(tie_hi)[index.tie_hi.1];
            }

            // TODO: replace_value on constants trips up feature detection
            // on flip flops; it interferes e.g. with `reset.is_always(false)`
            /*
            self.design.replace_value(Net::ZERO, out_lo);
            self.design.replace_value(Net::UNDEF, out_lo);
            self.design.replace_value(Net::ONE, out_hi);
            */
            complements.insert(Net::ZERO, Net::ONE);
            complements.insert(Net::UNDEF, Net::ONE);
            complements.insert(Net::ONE, Net::ZERO);
        }

        for cell in self.design.iter_cells_topo() {
            for net in cell.output().iter() {
                let Some(node) = self.nodes.get(&net) else {
                    continue;
                };

                for C in 0..2 {
                    let pol = &node.pol[C];
                    if pol.map_fouts == 0 {
                        continue;
                    }

                    if let Pi(complement_free) = node.role {
                        if C != 0 && !complement_free {
                            let _guard = if net.is_const() {
                                self.design.use_metadata(top_scope)
                            } else {
                                self.design.use_metadata_from(&[self
                                    .design
                                    .find_cell(net)
                                    .ok()
                                    .unwrap()
                                    .0])
                            };

                            let inverter = index.inverter;
                            let mut target_cell =
                                TargetCell::new(&inverter.name, &inverter.prototype);
                            target_cell.inputs = Value::from(net);
                            complements.insert(net, self.design.add_target(target_cell)[0]);
                        } else if C != 0 && complement_free {
                            let (cell, idx) = self.design.find_cell(net).ok().unwrap();
                            let Cell::Target(target) = &*cell.get() else {
                                panic!("PI w/ free complement not driven by target cell");
                            };
                            let Some(indexed_flop) = index.sm_index.per_cell.get(&target.kind)
                            else {
                                panic!("cannot find driver in sm index for PI w/ free complement");
                            };
                            let pins = &indexed_flop.pins;
                            assert_eq!(Some(idx), pins.data_out);
                            let qn = cell.output()[pins.data_negated_out.unwrap()];
                            complements.insert(net, qn);
                        }
                        continue;
                    }

                    let source_cells: Vec<CellRef> = self.collect_source(index, net, pol);
                    let _guard = self.design.use_metadata_from(source_cells.as_slice());

                    let (match_, target) = pol.selected.unwrap();
                    let local_map = (&target.via * &match_.npn).inv();
                    let mut target_cell =
                        TargetCell::new(&target.cell.name, &target.cell.prototype);
                    target_cell.inputs = (0..target.cell.prototype.input_len)
                        .map(|idx| {
                            let mut inet = match_.cut[local_map.p[idx] as usize];
                            if local_map.ic[idx] {
                                inet = *complements.get(&inet).unwrap();
                            }
                            inet
                        })
                        .collect::<Value>();
                    let onet = self.design.add_target(target_cell)[target.output];

                    if C != 0 {
                        complements.insert(net, onet);
                    } else {
                        self.design.replace_value(net, onet);
                    }
                }
            }
        }

        for (from, to) in self.negated_po_fixups.iter() {
            self.design
                .replace_value(*to, *complements.get(&from).unwrap());
        }
    }
}

pub fn map(design: &Design, target: Arc<SCLTarget>) {
    let index = TargetIndex::create(target.as_ref());

    let bump_alloc = Bump::new();
    let mut mapping = Mapping::init(design, &bump_alloc, &index);

    mapping.area_flow_round(&index, 1.0);
    println!("af A={:8.2}", mapping.walk_selection(true));
    mapping.area_flow_round(&index, 0.5);
    println!("af A={:8.2}", mapping.walk_selection(false));
    mapping.area_flow_round(&index, 0.2);
    println!("af A={:8.2}", mapping.walk_selection(false));
    mapping.area_flow_round(&index, 0.0);
    println!("af A={:8.2}", mapping.walk_selection(false));
    mapping.exact_round(&index);
    println!("ea A={:8.2}", mapping.walk_selection(false));
    mapping.exact_round(&index);
    println!("ea A={:8.2}", mapping.walk_selection(false));

    mapping.reintegrate(&index);
}

#[cfg(test)]
mod test {
    use crate::cm::map;
    use crate::target::SCLTarget;
    use prjunnamed_netlist::{assert_isomorphic, parse, Design, Net};
    use std::sync::Arc;

    fn prep_test_library1() -> Arc<SCLTarget> {
        let mut target = SCLTarget::new();
        target.read_liberty(
            r#"
			library(small) {
				cell(and) {
					area : 3;
					pin(A) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(B) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(Y) {
                        capacitance : 1.0;
						direction : output;
						function : "A&B";
					}
				}
				cell(or) {
					area : 3;
					pin(A) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(B) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(Y) {
                        capacitance : 1.0;
						direction : output;
						function : "A|B";
					}
				}
				cell(nand) {
					area : 3;
					pin(A) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(B) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(Y) {
                        capacitance : 1.0;
						direction : output;
						function : "!(A&B)";
					}
				}
				cell(xor) {
					area : 3;
					pin(A) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(B) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(Y) {
                        capacitance : 1.0;
						direction : output;
						function : "(A&!B)|(!A&B)";
					}
				}
				cell(nor) {
					area : 3;
					pin(A) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(B) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(Y) {
                        capacitance : 1.0;
						direction : output;
						function : "!(A|B)";
					}
				}
				cell(andnot) {
					area : 3;
					pin(A) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(B) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(Y) {
                        capacitance : 1.0;
						direction : output;
						function : "A&!B";
					}
				}
				cell(ornot) {
					area : 3;
					pin(A) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(B) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(Y) {
                        capacitance : 1.0;
						direction : output;
						function : "A|!B";
					}
				}
				cell(not) {
					area : 1;
					pin(A) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(Y) {
                        capacitance : 1.0;
						direction : output;
						function : "!A";
					}
				}
				cell(mux) {
					area: 7;
					pin(A) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(B) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(S) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(Y) {
                        capacitance : 1.0;
						direction : output;
						function : "(A&!S)|(B&S)";
					}
				}
				cell(ao2) {
					area: 4;
					pin(A) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(B) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(C) {
                        capacitance : 1.0;
						direction : input;
					}
					pin(Y) {
                        capacitance : 1.0;
						direction : output;
						function : "A&(B|C)";
					}
				}
				cell(tie) {
					area: 1;
					pin(LO) {
                        capacitance : 1.0;
						direction : output;
						function : "0";
					}
					pin(HI) {
                        capacitance : 1.0;
						direction : output;
						function : "1";
					}
				}
				cell(ff) {
					area: 3;
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
						function : "IQ";
					}
					pin(QN) {
                        capacitance : 1.0;
						direction : output;
						function : "IQN";
					}
					ff (IQ, IQN) {
						next_state : "D";
                        clocked_on : "CLK";
					}
				}
			}
		"#
            .to_string(),
        );
        Arc::new(target)
    }

    #[test]
    fn test_simple_map() {
        let target = prep_test_library1();
        let mut d = Design::with_target(Some(target.clone()));
        let a = d.add_input1("a");
        let b = d.add_input1("b");
        let s = d.add_input1("s");
        let y = d.add_mux(s, a, b);
        d.add_output("y", y);
        d.apply();
        map(&d, target.clone());
        d.apply();
        let mut gold = parse(Some(target), {
            r#"
		%0:1 = input "a"
		%1:1 = input "b"
		%2:1 = input "s"
		%3:1 = mux %2 %0 %1
		%7:1 = target "mux" {
		  input "A" = %0
		  input "B" = %1
		  input "S" = %2
		}
		%4:0 = output "y" %7
        "#
        })
        .unwrap();
        assert_isomorphic!(d, gold);
    }

    #[test]
    fn test_const_po_edge_cases() {
        let target = prep_test_library1();
        let mut d = Design::with_target(Some(target.clone()));
        d.add_output("y0", Net::ZERO);
        d.add_output("y1", Net::ONE);
        let not1 = d.add_not1(Net::ZERO);
        let not2 = d.add_not1(Net::ZERO);
        let not3 = d.add_not1(Net::ONE);
        d.add_output("y2", not1);
        d.add_output("y3", not1);
        d.add_output("y4", not2);
        d.add_output("y5", not3);
        d.apply();
        map(&d, target.clone());
        d.apply();
        let mut gold = parse(Some(target), {
            r#"
		%0:0 = output "y0" 0
        %1:0 = output "y1" 1
        %4:0 = output "y2" 1
        %5:0 = output "y3" 1
        %6:0 = output "y4" 1
        %7:0 = output "y5" 0
        "#
        })
        .unwrap();
        assert_isomorphic!(d, gold);
    }

    #[test]
    fn test_po_edge_cases() {
        let target = prep_test_library1();
        let mut d = Design::with_target(Some(target.clone()));
        let i0 = d.add_input1("i0");
        d.add_output("y0", i0);
        let not1 = d.add_not1(i0);
        let not2 = d.add_not1(i0);
        d.add_output("y2", not1);
        d.add_output("y3", not1);
        d.add_output("y4", not2);
        d.apply();
        map(&d, target.clone());
        d.apply();
        let mut gold = parse(Some(target), {
            r#"
		%0:1 = input "i0"
		%8:1 = target "not" {
		  input "A" = %0
		}
		%1:0 = output "y0" %0
		%3:0 = output "y2" %8
		%4:0 = output "y3" %8
		%5:0 = output "y4" %8
        "#
        })
        .unwrap();
        assert_isomorphic!(d, gold);
    }

    #[test]
    fn test_degenerate_mapping() {
        let target = prep_test_library1();
        let mut d = Design::with_target(Some(target.clone()));
        let a = d.add_input1("a");
        let y = d.add_xor(a, d.add_not1(a));
        d.add_output("y", y);
        d.apply();
        map(&d, target.clone());
        d.apply();
        let mut gold = parse(Some(target), {
            r#"
		%0:1 = input "a"
		%4:_ = target "tie" {
		  %4:1 = output "LO"
		  %5:1 = output "HI"
		}
		%3:0 = output "y" %5
        "#
        })
        .unwrap();
        assert_isomorphic!(d, gold);
    }

    #[test]
    fn test_using_flop_qn_output() {
        let target = prep_test_library1();
        let mut d = parse(Some(target.clone()), {
            r#"
        %0:1 = input "a"
        %1:1 = input "b"
        %2:1 = input "clk"
        %3:_ = target "ff" {
          input "CLK" = %2
          input "D" = %0
          %3:1 = output "Q"
          %4:1 = output "QN"
        }
        %5:1 = xor %1 %3
        %6:1 = not %5
        %7:0 = output "y" %6+0
        "#
        })
        .unwrap();
        map(&d, target.clone());
        d.compact();
        let mut gold = parse(Some(target), {
            r#"
		%0:1 = input "a"
		%1:1 = input "b"
		%2:1 = input "clk"
		%3:_ = target "ff" {
		  input "D" = %0
		  input "CLK" = %2
		  %3:1 = output "Q"
		  %4:1 = output "QN"
		}
		; drives "y"+0
		%6:1 = target "xor" {
		  input "A" = %4
		  input "B" = %1
		}
		%5:0 = output "y" %6
        "#
        })
        .unwrap();
        assert_isomorphic!(d, gold);
    }
}

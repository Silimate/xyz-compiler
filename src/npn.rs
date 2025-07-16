use std::ops::Mul;

pub type Truth6 = u64;

fn mask6(size: usize) -> Truth6 {
    if size == 6 {
        0xffffffffffffffff
    } else {
        (1 << (1 << size)) - 1
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct NPN {
    pub oc: bool,
    pub ic: [bool; 6],
    pub p: [i8; 6],
}

impl NPN {
    pub fn identity(ninputs: usize) -> Self {
        let mut p: [i8; 6] = [-1; 6];
        for i in 0..ninputs {
            p[i] = i as i8;
        }
        Self {
            oc: false,
            ic: [false; 6],
            p: p,
        }
    }

    pub fn empty() -> Self {
        Self {
            oc: false,
            ic: [false; 6],
            p: [-1; 6],
        }
    }

    pub fn ninputs(&self) -> usize {
        for i in 0..6 {
            if self.p[i] == -1 {
                return i;
            }
        }
        6
    }

    pub fn inv(&self) -> Self {
        let mut ret = Self {
            oc: self.oc,
            ic: [false; 6],
            p: [-1; 6],
        };
        for i in 0..self.ninputs() {
            ret.p[self.p[i] as usize] = i as i8;
            ret.ic[self.p[i] as usize] = self.ic[i];
        }
        ret
    }

    pub fn apply(&self, m: Truth6) -> Truth6 {
        let mut ret: Truth6 = 0;
        for idx1 in 0..(1 << self.ninputs()) {
            if m & 1 << idx1 == 0 {
                continue;
            }
            let mut idx2 = 0;
            for j in 0..self.ninputs() {
                if (idx1 & 1 << j != 0) ^ self.ic[j] {
                    idx2 |= 1 << self.p[j];
                }
            }
            ret |= 1 << idx2;
        }
        if self.oc {
            ret ^= mask6(self.ninputs());
        }
        return ret;
    }

    pub fn c_fingerprint(&self) -> [bool; 7] {
        [
            self.oc, self.ic[0], self.ic[1], self.ic[2], self.ic[3], self.ic[4], self.ic[5],
        ]
    }

    pub fn is_identity(&self) -> bool {
        if self.oc {
            return false;
        }
        for i in 0..self.ninputs() {
            if self.p[i] != i as i8 || self.ic[i] {
                return false;
            }
        }
        return true;
    }

    pub fn with_co(mut self) -> Self {
        self.oc ^= true;
        self
    }

    pub fn with_ci(mut self, i: usize) -> Self {
        self.ic[i] ^= true;
        self
    }
}

impl Mul for &NPN {
    type Output = NPN;

    fn mul(self, other: Self) -> NPN {
        assert_eq!(self.ninputs(), other.ninputs());
        let mut ret = NPN::empty();
        ret.oc = self.oc ^ other.oc;
        for i in 0..self.ninputs() {
            ret.ic[i] = self.ic[other.p[i] as usize] ^ other.ic[i];
            ret.p[i] = self.p[other.p[i] as usize];
        }
        ret
    }
}

const COFACTOR_MASKS: [Truth6; 6] = [
    0xaaaaaaaaaaaaaaaa,
    0xcccccccccccccccc,
    0xf0f0f0f0f0f0f0f0,
    0xff00ff00ff00ff00,
    0xffff0000ffff0000,
    0xffffffff00000000,
];

pub fn npn_semiclass(mut m: Truth6, ninputs: usize) -> NPN {
    if ninputs == 0 {
        return NPN::empty();
    }

    let mut npn = NPN::empty();
    let nbits = 1 << ninputs;
    let mask: Truth6 = mask6(ninputs);
    m &= mask;
    if m.count_ones() > nbits / 2 {
        npn.oc = true;
        m ^= mask;
    }

    let mut compls = [false; 6];
    let mut popcount = [0; 6];
    let mut order = [0; 6];

    for i in 0..ninputs {
        let mut nfactor = (m & !COFACTOR_MASKS[i]).count_ones();
        let mut pfactor = (m & COFACTOR_MASKS[i]).count_ones();

        if nfactor > pfactor {
            (pfactor, nfactor) = (nfactor, pfactor);
            let _ = pfactor;
            compls[i] = true;
        }

        let mut j = 0;
        while j < i {
            if nfactor < popcount[j] {
                break;
            }
            j += 1;
        }

        for k in (j..i).rev() {
            popcount[k + 1] = popcount[k];
            order[k + 1] = order[k];
        }
        popcount[j] = nfactor;
        order[j] = i;
    }

    for i in 0..ninputs {
        npn.ic[i] = compls[i];
        npn.p[order[i]] = i as i8;
    }

    npn
}

fn next_permutation(order: &mut [usize]) -> bool {
    let mut i = order.len() - 1;
    while i > 0 && order[i - 1] >= order[i] {
        i -= 1;
    }

    if i > 0 {
        let mut j = order.len() - 1;
        while order[j] <= order[i - 1] {
            j -= 1;
        }
        order.swap(i - 1, j);
    }

    order[i..].reverse();
    i > 0
}

pub fn npn_semiclass_allrepr<F>(mut m: Truth6, ninputs: usize, f: &mut F)
where
    F: FnMut(&NPN) -> (),
{
    if ninputs == 0 {
        f(&NPN::empty());
        return;
    }

    let mut oc_ambiguous = false;
    let mut npn = NPN::empty();
    let nbits = 1 << ninputs;
    let mask: Truth6 = mask6(ninputs);
    m &= mask;
    if m.count_ones() > nbits / 2 {
        npn.oc = true;
        m ^= mask;
    } else if m.count_ones() == nbits / 2 {
        oc_ambiguous = true;
    }

    loop {
        let mut compls = [false; 6];
        let mut popcount = [0; 6];
        let mut order = [0; 6];
        let mut ic_unambigous_mask = nbits - 1;

        for i in 0..ninputs {
            let mut nfactor = (m & !COFACTOR_MASKS[i]).count_ones();
            let mut pfactor = (m & COFACTOR_MASKS[i]).count_ones();

            if nfactor > pfactor {
                (pfactor, nfactor) = (nfactor, pfactor);
                let _ = pfactor;
                compls[i] = true;
            } else if nfactor == pfactor {
                ic_unambigous_mask &= !(1 << i);
            }

            let mut j = 0;
            while j < i {
                if nfactor < popcount[j] {
                    break;
                }
                j += 1;
            }

            for k in (j..i).rev() {
                popcount[k + 1] = popcount[k];
                order[k + 1] = order[k];
            }
            popcount[j] = nfactor;
            order[j] = i;
        }

        let tied = {
            let mut tied: [usize; 6] = [0; 6];
            let mut i = 0;
            while i < ninputs - 1 {
                let mark = i;
                while i < ninputs - 1 && popcount[i] == popcount[i + 1] {
                    tied[mark] += 1;
                    i += 1;
                }
                i += 1;
            }
            tied
        };

        let mut k = 0;
        while k < nbits {
            'permute: loop {
                for i in 0..ninputs {
                    npn.ic[i] = compls[i] ^ (k & 1 << i != 0);
                    npn.p[order[i]] = i as i8;
                }

                f(&npn);

                for j in (0..(ninputs - 1)).rev() {
                    if tied[j as usize] != 0 {
                        if next_permutation(&mut order[j..(j + tied[j] + 1)]) {
                            continue 'permute;
                        }
                    }
                }
                break;
            }

            k = ((k | ic_unambigous_mask) + 1) & !ic_unambigous_mask;
        }

        if oc_ambiguous {
            // we need to go around one more time with complemented output
            m ^= mask;
            npn.oc ^= true;
            oc_ambiguous = false;
        } else {
            break;
        }
    }
}

#[cfg(test)]
mod test {
    use crate::npn::{npn_semiclass, npn_semiclass_allrepr, Truth6, NPN};
    use std::collections::HashSet;

    #[test]
    fn test_classify_all_K4() {
        let K = 4;
        let mut unique = 0;
        let mut seen: HashSet<Truth6> = HashSet::new();
        for f in 0..(1 << (1 << K)) {
            let npn = npn_semiclass(f, K);
            let sc = npn.apply(f);
            assert_eq!(npn.inv().apply(sc), f);
            assert!((&npn.inv() * &npn).is_identity());
            assert!((&npn * &npn.inv()).is_identity());

            if seen.contains(&sc) {
                continue;
            }

            unique += 1;

            let mut found = false;
            npn_semiclass_allrepr(f, K, &mut |npn_any| {
                let sc_any = npn_any.apply(f);
                seen.insert(sc_any);
                if sc == sc_any {
                    found = true;
                }
            });
            assert!(found);
        }
        assert_eq!(unique, 222);
    }

    #[test]
    fn test_K1_expected() {
        assert_eq!(npn_semiclass(2, 1), NPN::identity(1));
        assert_eq!(npn_semiclass(1, 1), NPN::identity(1).with_ci(0));
    }
}

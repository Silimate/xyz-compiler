use crate::npn::Truth6;
use prjunnamed_netlist::{
    Cell, Const, Design, Target, TargetCell, TargetImportError, TargetPrototype, Value,
};
use regex::Regex;
use std::collections::{BTreeMap, HashMap};
use std::iter::Peekable;

struct Scanner<I>
where
    I: Iterator<Item = char>,
{
    iter: Peekable<I>,
    staging: String,
    lineno: usize,
    colno: usize,
    keyword_re: Regex,
    inject: Option<Token>,
}

#[derive(Debug, PartialEq)]
enum Token {
    Punctuation(char),
    Float(f64),
    Id(String),
    String(String),
    None,
    EOF,
    Invalid,
}

impl<I: Iterator<Item = char>> Scanner<I> {
    fn wrap(iter: I) -> Self {
        Self {
            iter: iter.peekable(),
            staging: String::new(),
            lineno: 1,
            colno: 1,
            keyword_re: Regex::new("[a-zA-Z]([a-zA-Z0-9_])*").unwrap(),
            inject: None,
        }
    }

    fn peek_char(&mut self) -> Option<char> {
        self.iter.peek().copied()
    }

    fn peek_is_digit(&mut self) -> bool {
        let Some(ch) = self.peek_char() else {
            return false;
        };
        matches!(ch, '0'..='9')
    }

    fn error(&self, arg: &str) -> ! {
        panic!("syntax error at {}:{}: {}", self.lineno, self.colno, arg)
    }

    fn error_eof(&self) -> ! {
        self.error("unexpected EOF")
    }

    fn next_char(&mut self) -> Option<char> {
        let ret = self.iter.next();
        if let Some(ch) = ret {
            self.staging.push(ch);
            match ch {
                '\n' => {
                    self.lineno += 1;
                    self.colno = 1;
                }
                _ => {
                    self.colno += 1;
                }
            }
        }
        ret
    }

    fn expect_char(&mut self, expect: char) {
        let Some(found) = self.next_char() else {
            self.error_eof();
        };
        if found != expect {
            self.error(&format!("expected character {expect}; found {found}"));
        }
    }

    fn flush(&mut self) -> String {
        std::mem::replace(&mut self.staging, String::new())
    }

    fn push_back(&mut self, tok: Token) {
        assert!(self.inject == None);
        self.inject = Some(tok);
    }

    fn next(&mut self) -> Token {
        if let Some(_) = &self.inject {
            return std::mem::replace(&mut self.inject, None).unwrap();
        };
        'outer: loop {
            self.flush();
            let Some(ch) = self.next_char() else {
                return Token::EOF;
            };
            match ch {
                '/' => {
                    self.expect_char('*');
                    while let Some(ch) = self.next_char() {
                        if ch == '*' && self.peek_char() == Some('/') {
                            self.next_char();
                            continue 'outer;
                        }
                    }
                    self.error_eof()
                }
                ' ' | '\r' | '\n' | '\t' => {}
                '\\' if matches!(self.peek_char(), Some('\r') | Some('\n')) => {}
                '"' => {
                    let mut content = String::new();
                    while let Some(ch) = self.next_char() {
                        match ch {
                            '\\' => {
                                let Some(ch) = self.next_char() else {
                                    self.error_eof();
                                };
                                match ch {
                                    '\r' | '\n' => {
                                        if ch == '\r' && self.peek_char() == Some('\n') {
                                            self.next_char();
                                        }
                                    }
                                    _ => {
                                        content.push(ch);
                                    }
                                }
                            }
                            '"' => return Token::String(content),
                            '\n' => self.error("unterminated string"),
                            ch => content.push(ch),
                        }
                    }
                    self.error_eof()
                }
                'a'..='z' | 'A'..='Z' | '0'..='9' | '_' => {
                    while let Some(ch) = self.peek_char() {
                        if matches!(
                            ch,
                            ',' | ':'
                                | ';'
                                | '|'
                                | '('
                                | ')'
                                | '{'
                                | '}'
                                | '*'
                                | '&'
                                | '\''
                                | '='
                                | ' '
                                | '\t'
                                | '\r'
                                | '\n'
                        ) {
                            break;
                        }
                        self.next_char();
                    }
                    let text = self.flush();
                    if self.keyword_re.is_match(&text) {
                        return Token::Id(text);
                    } else {
                        return Token::String(text);
                    }
                }
                ',' | ':' | ';' | '|' | '(' | ')' | '{' | '}' | '*' | '&' | '\'' | '=' => {
                    return Token::Punctuation(ch)
                }
                _ => {
                    self.error(&format!("unexpected character '{ch}'"));
                }
            }
        }
    }

    fn consume(&mut self, expected: Token) -> bool {
        let read = self.next();
        if read == expected {
            return true;
        } else {
            self.push_back(read);
            return false;
        }
    }

    fn expect(&mut self, expected: Token) {
        let read = self.next();
        if read != expected {
            self.error(&format!("expected {:?} found {:?}", expected, read))
        }
    }
}

type AttrValue = String;

#[derive(Debug, Clone)]
pub enum Statement {
    SimpleAttr(String, AttrValue),
    ComplexAttr(String, Vec<AttrValue>),
    Group(String, Vec<AttrValue>, Vec<Statement>),
    Variable(String, f64),
}

impl Statement {
    fn key(&self) -> &str {
        match self {
            Statement::SimpleAttr(key @ _, _) => key,
            Statement::ComplexAttr(key @ _, _) => key,
            Statement::Group(key @ _, _, _) => key,
            Statement::Variable(key @ _, _) => key,
        }
    }

    fn sort(&mut self) {
        match self {
            Statement::Group(_, _, stmts @ _) => {
                for stmt in &mut *stmts {
                    stmt.sort();
                }
                stmts.sort_by(|a, b| a.key().cmp(b.key()));
            }
            _ => {}
        }
    }

    pub fn lookup(&self, key: &str) -> Option<&Statement> {
        match self.lookup_all(key) {
            [el @ _, ..] => Some(el),
            [] => None,
        }
    }

    fn lookup_all(&self, key: &str) -> &[Statement] {
        match self {
            Statement::Group(_, _, ref stmts @ _) => {
                let i = stmts.partition_point(|a| a.key() < key);
                let j = stmts.partition_point(|a| a.key() <= key);
                &stmts[i..j]
            }
            _ => &[],
        }
    }

    fn find_attribute(&self, key: &str) -> Option<&str> {
        match self.lookup(key) {
            Some(Statement::SimpleAttr(_, ref s @ _)) => Some(s),
            _ => None,
        }
    }

    fn arguments(&self) -> &Vec<AttrValue> {
        match self {
            Statement::ComplexAttr(_, attrs, ..) => attrs,
            Statement::Group(_, attrs, ..) => attrs,
            _ => {
                unreachable!();
            }
        }
    }
}

fn skip_semicolon<I>(scanner: &mut Scanner<I>)
where
    I: Iterator<Item = char>,
{
    scanner.consume(Token::Punctuation(';'));
}

fn parse_attr_value<I>(scanner: &mut Scanner<I>) -> AttrValue
where
    I: Iterator<Item = char>,
{
    match scanner.next() {
        Token::Id(text @ _) | Token::String(text @ _) => return text,
        t @ _ => scanner.error(&format!("bad attr value {:?}", t)),
    }
}

fn parse_statement<I>(scanner: &mut Scanner<I>) -> Statement
where
    I: Iterator<Item = char>,
{
    let Token::Id(label) = scanner.next() else {
        scanner.error("expected identifier")
    };

    match scanner.next() {
        Token::Punctuation(':') => {
            let value = parse_attr_value(scanner);
            skip_semicolon(scanner);
            return Statement::SimpleAttr(label, value);
        }
        Token::Punctuation('=') => {
            // FIXME: should allow non-identifier strings on LHS
            let Token::Float(value) = scanner.next() else {
                scanner.error("expected floating-point number");
            };
            skip_semicolon(scanner);
            return Statement::Variable(label, value);
        }
        Token::Punctuation('(') => {
            // fall through
        }
        t @ _ => {
            scanner.error(&format!("unexpected token {:?}", t));
        }
    }

    let mut expecting_value = false;
    let mut attrs: Vec<AttrValue> = Vec::new();
    loop {
        match scanner.next() {
            Token::EOF => {
                scanner.error_eof();
            }
            Token::Punctuation(',') if !expecting_value => {
                expecting_value = true;
                continue;
            }
            Token::Punctuation(')') => break,
            t @ (Token::String(_) | Token::Id(_)) => {
                scanner.push_back(t);
                attrs.push(parse_attr_value(scanner));
            }
            t @ _ => {
                scanner.error(&format!("unexpected token {:?}", t));
            }
        }
        expecting_value = false;
    }

    if scanner.consume(Token::Punctuation('{')) {
        let mut statements: Vec<Statement> = Vec::new();
        while !scanner.consume(Token::Punctuation('}')) {
            statements.push(parse_statement(scanner));
        }
        skip_semicolon(scanner);
        return Statement::Group(label, attrs, statements);
    } else {
        skip_semicolon(scanner);
        return Statement::ComplexAttr(label, attrs);
    }
}

fn parse_pin1<I>(chars: &mut Peekable<I>) -> Option<(bool, String)>
where
    I: Iterator<Item = char>,
{
    while matches!(chars.peek(), Some(' ') | Some('\t')) {
        chars.next();
    }
    assert!(chars.peek() != None);

    let mut negated: bool;
    let mut pin: String;

    match chars.peek().unwrap() {
        'a'..='z' | 'A'..='Z' => {
            pin = String::new();
            negated = false;
            while !matches!(
                chars.peek(),
                None | Some('\t' | '(' | ')' | '\'' | '!' | '^' | '*' | '&' | ' ' | '+' | '|')
            ) {
                pin.push(chars.next().unwrap());
            }
        }
        '!' => {
            chars.next();
            (negated, pin) = parse_pin1(chars)?;
            negated = !negated;
        }
        _ => return None,
    }

    while matches!(chars.peek(), Some(' ' | '\t')) {
        chars.next();
    }

    if chars.peek() == Some(&'\'') {
        chars.next();
        negated = !negated;
    }

    while matches!(chars.peek(), Some(' ' | '\t')) {
        chars.next();
    }

    return Some((negated, pin));
}

pub fn parse_pin(expr: &str) -> Option<(bool, String)> {
    let mut chars = expr.chars().peekable();
    let ret = parse_pin1(&mut chars)?;
    if chars.peek() != None {
        return None;
    }
    Some(ret)
}

// Thanks to Ravenslofty for reference code
fn parse_expr<I>(
    chars: &mut Peekable<I>,
    input_pins: &[LibraryPort],
    priority: usize,
) -> Option<Truth6>
where
    I: Iterator<Item = char>,
{
    while matches!(chars.peek(), Some(' ') | Some('\t')) {
        chars.next();
    }
    assert!(chars.peek() != None);

    let mask: Truth6 = (1 as Truth6)
        .checked_shl(1 << input_pins.len())
        .unwrap_or(0)
        .overflowing_sub(1)
        .0;
    let mut lhs: Truth6;

    match chars.peek().unwrap() {
        'a'..='z' | 'A'..='Z' => {
            let mut pin = String::new();
            while !matches!(
                chars.peek(),
                None | Some('\t' | '(' | ')' | '\'' | '!' | '^' | '*' | '&' | ' ' | '+' | '|')
            ) {
                pin.push(chars.next().unwrap());
            }

            let pin_idx = input_pins.iter().position(|p| p.name == pin)?;

            lhs = 0;
            for i in 0..(1 << input_pins.len()) {
                if i & 1 << pin_idx != 0 {
                    lhs |= 1 << i;
                }
            }
        }
        '0' => {
            lhs = 0;
        }
        '1' => {
            lhs = mask;
        }
        '(' => {
            chars.next();
            lhs = parse_expr(chars, input_pins, 0)?;
            assert_eq!(chars.next(), Some(')'));
        }
        '!' => {
            chars.next();
            lhs = !parse_expr(chars, input_pins, 7)? & mask;
        }
        ch @ _ => {
            panic!("unrecognized {}", ch);
        }
    }

    loop {
        while matches!(chars.peek(), Some(' ' | '\t')) {
            chars.next();
        }

        if chars.peek() == None {
            break;
        }

        match chars.peek().unwrap() {
            '\'' => {
                if priority > 7 {
                    break;
                }

                chars.next();
                lhs = !lhs & mask;
                continue;
            }
            '^' => {
                if priority > 5 {
                    break;
                }

                chars.next();
                lhs ^= parse_expr(chars, input_pins, 6)?;
                continue;
            }
            '&' | '*' => {
                // TODO: support whitespace as infix AND
                if priority > 3 {
                    break;
                }

                chars.next();
                lhs &= parse_expr(chars, input_pins, 4)?;
                continue;
            }

            '+' | '|' => {
                if priority > 1 {
                    break;
                }

                chars.next();
                lhs |= parse_expr(chars, input_pins, 2)?;
                continue;
            }

            _ => {
                break;
            }
        }
    }

    return Some(lhs);
}

#[derive(Clone, Copy, Eq, Hash, PartialEq, Debug)]
pub enum RiseFall {
    Rise,
    Fall,
}

impl RiseFall {
    pub fn text(self) -> &'static str {
        match self {
            RiseFall::Rise => "rise",
            RiseFall::Fall => "fall",
        }
    }
}

#[derive(Debug)]
enum TableModelVariable {
    TotalOutputNetCapacitance,
    InputTransitionTime,
    RelatedPinTransition,
    ConstrainedPinTransition,
    Unknown(String),
}

#[derive(Debug)]
struct TableModelAxis {
    variable: TableModelVariable,
    indices: Vec<f32>,
}

#[derive(Debug)]
struct TableModel {
    axes: Vec<TableModelAxis>,
    values: Vec<f32>,
}

#[derive(Debug)]
enum ArcDelayModel {
    Table(TableModel),
}

#[derive(Debug, Clone)]
enum ArcType {
    Combinational(RiseFall, RiseFall),
    ThreeStateEnable(RiseFall, RiseFall),
    ThreeStateDisable(RiseFall, RiseFall),
    Launch(RiseFall, RiseFall),
    HoldCheck(RiseFall, RiseFall),
    SetupCheck(RiseFall, RiseFall),
    NonSeqHoldCheck(RiseFall, RiseFall),
    NonSeqSetupCheck(RiseFall, RiseFall),
    Preset(RiseFall),
    Clear(RiseFall),
    RecoveryCheck(RiseFall, RiseFall),
    RemovalCheck(RiseFall, RiseFall),
    MinPulseWidth(RiseFall),
    Unknown(String),
}

impl ArcType {
    fn to_rf(&self) -> RiseFall {
        use ArcType::*;
        match self {
            Combinational(_, rf)
            | ThreeStateEnable(_, rf)
            | ThreeStateDisable(_, rf)
            | Launch(_, rf)
            | HoldCheck(_, rf)
            | SetupCheck(_, rf)
            | NonSeqHoldCheck(_, rf)
            | NonSeqSetupCheck(_, rf)
            | Preset(rf)
            | Clear(rf)
            | RecoveryCheck(_, rf)
            | RemovalCheck(_, rf)
            | MinPulseWidth(rf) => *rf,
            Unknown(_) => {
                unreachable!()
            }
        }
    }
}

#[derive(PartialOrd, Ord, Eq, Hash, PartialEq, Debug, Clone, Copy)]
pub enum LibraryPortIndex {
    Input(usize),
    Output(usize),
    Io(usize),
    Other(usize),
    None,
}

#[derive(Debug)]
pub struct Arc {
    arc_type: ArcType,
    from_pin: LibraryPortIndex,
    to_pin: LibraryPortIndex,
    delay_model: Option<ArcDelayModel>,
    constraint_model: Option<ArcDelayModel>,
    slew_model: Option<ArcDelayModel>,
}

#[derive(Debug)]
pub enum LibraryPortDirection {
    Input,
    Output,
    InOut,
    Internal,
}

#[derive(Debug)]
pub struct LibraryPort {
    name: String,
    direction: LibraryPortDirection,
    index: LibraryPortIndex,
    cap: f32,
    rise_cap: f32,
    fall_cap: f32,
    pub cm_function: Option<Truth6>,
}

#[derive(Debug)]
pub struct LibraryCell {
    pub name: String,
    pub ast: Statement,
    pub prototype: TargetPrototype,
    pub output_pin_asts: Vec<Statement>,
    pub input_ports: Vec<LibraryPort>,
    pub output_ports: Vec<LibraryPort>,
    pub io_ports: Vec<LibraryPort>,
    pub other_ports: Vec<LibraryPort>,
    pub ports_by_name: HashMap<String, LibraryPortIndex>,
    pub arc_sets: BTreeMap<(LibraryPortIndex, LibraryPortIndex), Vec<Arc>>,
    pub area: f32,
}

#[derive(Debug)]
pub struct SCLTarget {
    pub cells: BTreeMap<String, LibraryCell>,
}

fn parse_arc_type(statement: &Statement) -> Vec<ArcType> {
    use ArcType::Combinational as Comb;
    use ArcType::*;
    use RiseFall::{Fall, Rise};

    let sense = statement.find_attribute("timing_sense");

    match statement.find_attribute("timing_type") {
        Some("combinational") | None => {
            return match sense {
                Some("positive_unate") => vec![Comb(Rise, Rise), Comb(Fall, Fall)],
                Some("negative_unate") => vec![Comb(Rise, Fall), Comb(Fall, Rise)],
                Some("non_unate") => vec![
                    Comb(Rise, Rise),
                    Comb(Rise, Fall),
                    Comb(Fall, Rise),
                    Comb(Fall, Fall),
                ],
                _ => {
                    panic!("bad timing_sense value {:?}", sense)
                }
            }
        }
        Some("combinational_rise") => {
            return match sense {
                Some("positive_unate") => vec![Comb(Rise, Rise)],
                Some("negative_unate") => vec![Comb(Fall, Rise)],
                Some("non_unate") => vec![Comb(Rise, Rise), Comb(Fall, Rise)],
                _ => {
                    panic!("unknown timing_sense value {:?}", sense)
                }
            }
        }
        Some("combinational_fall") => {
            return match sense {
                Some("positive_unate") => vec![Comb(Fall, Fall)],
                Some("negative_unate") => vec![Comb(Rise, Fall)],
                Some("non_unate") => vec![Comb(Rise, Fall), Comb(Fall, Fall)],
                _ => {
                    panic!("unknown timing_sense value {:?}", sense)
                }
            }
        }
        Some("setup_rising") => vec![SetupCheck(Rise, Rise), SetupCheck(Rise, Fall)],
        Some("setup_falling") => vec![SetupCheck(Fall, Rise), SetupCheck(Fall, Fall)],
        Some("hold_rising") => vec![HoldCheck(Rise, Rise), HoldCheck(Rise, Fall)],
        Some("hold_falling") => vec![HoldCheck(Fall, Rise), HoldCheck(Fall, Fall)],
        Some("non_seq_setup_rising") => {
            vec![NonSeqSetupCheck(Rise, Rise), NonSeqSetupCheck(Rise, Fall)]
        }
        Some("non_seq_setup_falling") => {
            vec![NonSeqSetupCheck(Fall, Rise), NonSeqSetupCheck(Fall, Fall)]
        }
        Some("non_seq_hold_rising") => {
            vec![NonSeqHoldCheck(Rise, Rise), NonSeqHoldCheck(Rise, Fall)]
        }
        Some("non_seq_hold_falling") => {
            vec![NonSeqHoldCheck(Fall, Rise), NonSeqHoldCheck(Fall, Fall)]
        }
        Some("rising_edge") => vec![Launch(Rise, Rise), Launch(Rise, Fall)],
        Some("falling_edge") => vec![Launch(Fall, Rise), Launch(Fall, Fall)],
        Some("clear") => vec![Clear(Rise), Clear(Fall)],
        Some("preset") => vec![Preset(Rise), Preset(Fall)],
        Some("clear") => vec![Clear(Rise), Clear(Fall)],
        Some("recovery_rising") => vec![RecoveryCheck(Rise, Rise), RecoveryCheck(Rise, Fall)],
        Some("recovery_falling") => vec![RecoveryCheck(Fall, Rise), RecoveryCheck(Fall, Fall)],
        Some("removal_rising") => vec![RemovalCheck(Rise, Rise), RemovalCheck(Rise, Fall)],
        Some("removal_falling") => vec![RemovalCheck(Fall, Rise), RemovalCheck(Fall, Fall)],
        Some("three_state_enable") | None => {
            return match sense {
                Some("positive_unate") => {
                    vec![ThreeStateEnable(Rise, Rise), ThreeStateEnable(Rise, Fall)]
                }
                Some("negative_unate") => {
                    vec![ThreeStateEnable(Fall, Rise), ThreeStateEnable(Fall, Fall)]
                }
                Some("non_unate") => vec![
                    ThreeStateEnable(Rise, Rise),
                    ThreeStateEnable(Rise, Fall),
                    ThreeStateEnable(Fall, Rise),
                    ThreeStateEnable(Fall, Fall),
                ],
                _ => {
                    panic!("bad timing_sense value {:?}", sense)
                }
            }
        }
        Some("three_state_disable") | None => {
            return match sense {
                Some("positive_unate") => {
                    vec![ThreeStateDisable(Rise, Rise), ThreeStateDisable(Rise, Fall)]
                }
                Some("negative_unate") => {
                    vec![ThreeStateDisable(Fall, Rise), ThreeStateDisable(Fall, Fall)]
                }
                Some("non_unate") => vec![
                    ThreeStateDisable(Rise, Rise),
                    ThreeStateDisable(Rise, Fall),
                    ThreeStateDisable(Fall, Rise),
                    ThreeStateDisable(Fall, Fall),
                ],
                _ => {
                    panic!("bad timing_sense value {:?}", sense)
                }
            }
        }
        Some("three_state_enable_rise") => {
            return match sense {
                Some("positive_unate") => vec![ThreeStateEnable(Rise, Rise)],
                Some("negative_unate") => vec![ThreeStateEnable(Fall, Rise)],
                Some("non_unate") => {
                    vec![ThreeStateEnable(Rise, Rise), ThreeStateEnable(Fall, Rise)]
                }
                _ => {
                    panic!("unknown timing_sense value {:?}", sense)
                }
            }
        }
        Some("three_state_enable_fall") => {
            return match sense {
                Some("positive_unate") => vec![ThreeStateEnable(Rise, Fall)],
                Some("negative_unate") => vec![ThreeStateEnable(Fall, Fall)],
                Some("non_unate") => {
                    vec![ThreeStateEnable(Rise, Fall), ThreeStateEnable(Fall, Fall)]
                }
                _ => {
                    panic!("unknown timing_sense value {:?}", sense)
                }
            }
        }
        Some("three_state_disable_rise") => {
            return match sense {
                Some("positive_unate") => vec![ThreeStateDisable(Rise, Rise)],
                Some("negative_unate") => vec![ThreeStateDisable(Fall, Rise)],
                Some("non_unate") => {
                    vec![ThreeStateDisable(Rise, Rise), ThreeStateDisable(Fall, Rise)]
                }
                _ => {
                    panic!("unknown timing_sense value {:?}", sense)
                }
            }
        }
        Some("three_state_disable_fall") => {
            return match sense {
                Some("positive_unate") => vec![ThreeStateDisable(Rise, Fall)],
                Some("negative_unate") => vec![ThreeStateDisable(Fall, Fall)],
                Some("non_unate") => {
                    vec![ThreeStateDisable(Rise, Fall), ThreeStateDisable(Fall, Fall)]
                }
                _ => {
                    panic!("unknown timing_sense value {:?}", sense)
                }
            }
        }
        Some("min_pulse_width") => vec![MinPulseWidth(Fall), MinPulseWidth(Rise)],
        Some(kind) => return vec![ArcType::Unknown(kind.to_owned())],
    }
}

fn parse_table_template(template_name: &str, library: &Statement) -> Vec<TableModelAxis> {
    if template_name == "scalar" {
        return vec![];
    }

    // FIXME: linear search
    for template in library.lookup_all(&"lu_table_template") {
        if template.arguments()[..] == [template_name] {
            let mut axes = vec![];
            let mut var_idx = 1;
            loop {
                let Some(variable_text) = template.find_attribute(&format!("variable_{}", var_idx))
                else {
                    break;
                };

                use TableModelVariable::*;
                let variable = match variable_text {
                    "total_output_net_capacitance" => TotalOutputNetCapacitance,
                    "input_transition_time" | "input_net_transition" => InputTransitionTime,
                    "related_pin_transition" => RelatedPinTransition,
                    "constrained_pin_transition" => ConstrainedPinTransition,
                    text @ _ => {
                        panic!("unsupported variable {}", text);
                    }
                };

                let Some(indices) =
                    template
                        .lookup(&format!("index_{}", var_idx))
                        .and_then(|stmt| {
                            let [ref text] = stmt.arguments()[..] else {
                                return None;
                            };
                            text.split(",")
                                .map(|v| v.trim().parse::<f32>())
                                .collect::<Result<Vec<_>, _>>()
                                .ok()
                        })
                else {
                    panic!("no index: {:?}", template);
                };

                axes.push(TableModelAxis { variable, indices });
                var_idx += 1;
            }
            return axes;
        }
    }

    panic!("table template {} not found", template_name);
}

fn parse_delay_model(statement: &Statement, library: &Statement) -> ArcDelayModel {
    let [ref template_name] = statement.arguments()[..] else {
        panic!("malformed pin");
    };

    let axes = parse_table_template(template_name, library);

    let Some(values) = statement.lookup(&"values").and_then(|stmt| {
        stmt.arguments()
            .iter()
            .map(|text| text.split(",").map(|v| v.trim().parse::<f32>()))
            .flatten()
            .collect::<Result<Vec<_>, _>>()
            .ok()
    }) else {
        panic!("cannot parse values");
    };

    ArcDelayModel::Table(TableModel { axes, values })
}

impl SCLTarget {
    pub fn new() -> Self {
        SCLTarget {
            cells: BTreeMap::new(),
        }
    }

    pub fn read_port(&mut self, statement: &Statement) -> LibraryPort {
        use LibraryPortDirection::*;

        let [ref name] = statement.arguments()[..] else {
            panic!("malformed pin");
        };

        let direction = match statement.find_attribute("direction") {
            Some("input") => Input,
            Some("output") => Output,
            Some("inout") => InOut,
            Some("internal") => Internal,
            v @ _ => {
                panic!("unknown direction {:?}", v)
            }
        };

        // TODO: silent failures
        let cap = statement
            .find_attribute("capacitance")
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.0);
        let rise_cap = statement
            .find_attribute("rise_capacitance")
            .map(|v| v.parse::<f32>().ok().unwrap())
            .unwrap_or(cap);
        let fall_cap = statement
            .find_attribute("fall_capacitance")
            .map(|v| v.parse::<f32>().ok().unwrap())
            .unwrap_or(cap);

        LibraryPort {
            name: name.clone(),
            direction,
            index: LibraryPortIndex::None,
            cap,
            rise_cap,
            fall_cap,
            cm_function: None,
        }
    }

    pub fn read_cell(&mut self, top_statement: &Statement, statement: &Statement) {
        let Statement::Group(_, ref attrs, _) = statement else {
            unreachable!();
        };

        let [ref name] = attrs[..] else {
            panic!("bad cell (1)");
        };
        let mut proto;
        // TODO
        if statement.lookup("ff").is_none() {
            proto = TargetPrototype::new_pure();
        } else {
            proto = TargetPrototype::new_has_state();
        }

        let mut input_ports: Vec<LibraryPort> = vec![];
        let mut output_ports: Vec<LibraryPort> = vec![];
        let mut io_ports: Vec<LibraryPort> = vec![];
        let mut other_ports: Vec<LibraryPort> = vec![];
        let mut ports_by_name: HashMap<String, LibraryPortIndex> = HashMap::new();
        let mut arc_sets: BTreeMap<_, Vec<Arc>> = BTreeMap::new();
        let mut output_pin_asts: Vec<Statement> = vec![];

        for pin_statement in statement.lookup_all(&"pin") {
            let mut port = self.read_port(pin_statement);

            match port.direction {
                Input => {
                    proto = proto.add_input(port.name.clone(), Const::undef(1));
                }
                Output => {
                    proto = proto.add_output(port.name.clone(), 1);
                    output_pin_asts.push(pin_statement.clone());
                }
                InOut => {
                    proto = proto.add_io(port.name.clone(), 1);
                }
                _ => {}
            }

            let target_vector;
            use LibraryPortDirection::*;
            (port.index, target_vector) = match port.direction {
                Input => (LibraryPortIndex::Input(input_ports.len()), &mut input_ports),
                Output => (
                    LibraryPortIndex::Output(output_ports.len()),
                    &mut output_ports,
                ),
                InOut => (LibraryPortIndex::Io(io_ports.len()), &mut io_ports),
                _ => (LibraryPortIndex::Other(other_ports.len()), &mut other_ports),
            };

            ports_by_name.insert(port.name.clone(), port.index);
            target_vector.push(port);
        }

        // pass two
        for pin_stmt in statement.lookup_all(&"pin") {
            let [ref pin_name] = pin_stmt.arguments()[..] else {
                panic!("malformed pin");
            };

            let pin_index = *ports_by_name.get(pin_name).unwrap();
            for timing_stmt in pin_stmt.lookup_all(&"timing") {
                let arc_types = parse_arc_type(timing_stmt);
                if let [ArcType::Unknown(_)] = arc_types[..] {
                    eprintln!(
                        "ignoring unknown arc type {:?} to pin {} on cell {}",
                        arc_types[0], pin_name, name
                    );
                    continue;
                }

                let Some(related_pin) = timing_stmt.find_attribute("related_pin") else {
                    panic!("missing related_pin");
                };

                let from_pin = *ports_by_name.get(related_pin).unwrap();

                for arc_type in arc_types.iter() {
                    let to_rf_text = arc_type.to_rf().text();

                    let delay_model = timing_stmt
                        .lookup(&format!("cell_{}", to_rf_text))
                        .map(|stmt| parse_delay_model(stmt, top_statement));
                    let constraint_model = timing_stmt
                        .lookup(&format!("{}_constraint", to_rf_text))
                        .map(|stmt| parse_delay_model(stmt, top_statement));
                    let slew_model = timing_stmt
                        .lookup(&format!("{}_transition", to_rf_text))
                        .map(|stmt| parse_delay_model(stmt, top_statement));

                    arc_sets
                        .entry((from_pin, pin_index))
                        .or_default()
                        .push(Arc {
                            arc_type: arc_type.clone(),
                            from_pin,
                            to_pin: pin_index,
                            delay_model,
                            constraint_model,
                            slew_model,
                        });
                }
            }

            if let LibraryPortIndex::Output(out_no) = pin_index {
                // use Lut or some other type here, no reason for LUT-6 limit
                output_ports[out_no].cm_function =
                    pin_stmt.find_attribute("function").and_then(|text| {
                        let mut iter = text.chars().peekable();
                        parse_expr(&mut iter, &input_ports[..], 0)
                    })
            }
        }

        // FIXME: check for conflict
        self.cells.insert(
            name.clone(),
            LibraryCell {
                name: name.clone(),
                ast: statement.clone(),
                prototype: proto,
                output_pin_asts,
                input_ports,
                output_ports,
                io_ports,
                other_ports,
                ports_by_name,
                arc_sets,
                area: statement
                    .find_attribute("area")
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap(),
            },
        );
    }

    pub fn read_liberty(&mut self, text: String) {
        let mut top_group = {
            let mut scanner = Scanner::wrap(text.chars());
            parse_statement(&mut scanner)
        };
        top_group.sort();

        let Statement::Group(ref text, ref attrs, ref top_statements) = top_group else {
            panic!("bad library (1)");
        };
        if text != "library" {
            panic!("bad library (2)");
        };
        let [ref lib_name] = attrs[..] else {
            panic!("bad library (3)");
        };
        println!("reading {}", lib_name);
        for statement in top_statements {
            if let Statement::Group(ref name, _, _) = statement {
                if name == "cell" {
                    self.read_cell(&top_group, &statement)
                }
            }
        }
    }
}

impl Target for SCLTarget {
    fn name(&self) -> &str {
        "liberty"
    }

    fn options(&self) -> BTreeMap<String, String> {
        BTreeMap::new()
    }

    fn prototype(&self, name: &str) -> Option<&TargetPrototype> {
        Some(&self.cells.get(name)?.prototype)
    }

    fn validate(&self, _design: &Design, _cell: &TargetCell) {}

    fn import(&self, design: &mut Design) -> Result<(), TargetImportError> {
        for cell_ref in design.iter_cells() {
            let Cell::Other(instance) = &*cell_ref.get() else {
                continue;
            };
            if let Some(prototype) = self.prototype(&instance.kind) {
                cell_ref.unalive();
                let (target_cell, value) = prototype
                    .instance_to_target_cell(design, &instance, cell_ref.output())
                    .map_err(|cause| TargetImportError::new(cell_ref, cause))?;
                design.replace_value(value, design.add_target(target_cell));
            }
        }
        design.compact();
        Ok(())
    }

    fn export(&self, design: &mut Design) {
        // copied from prjunnamed/siliconblue/src/lib.rs
        for cell_ref in design.iter_cells() {
            let Cell::Target(target_cell) = &*cell_ref.get() else {
                continue;
            };
            let _guard = design.use_metadata_from(&[cell_ref]);
            let prototype = design.target_prototype(target_cell);
            let mut instance = prototype.target_cell_to_instance(target_cell);

            // perform output reconstruction surgery on the instance
            let mut new_output_map = vec![];
            let mut index = 0;
            instance.outputs.retain(|_name, range| {
                let orig_range = range.clone();
                let instance_range = index..index + range.len();
                *range = instance_range.clone();
                index = range.end;
                new_output_map.push((orig_range, instance_range));
                true
            });
            let instance_output = design.add_other(instance);
            let mut new_output = Value::undef(cell_ref.output_len());
            for (orig_range, instance_range) in new_output_map {
                new_output[orig_range].copy_from_slice(&instance_output[instance_range]);
            }
            design.replace_value(cell_ref.output(), new_output);
            cell_ref.unalive();
        }
        design.compact();
    }

    fn synthesize(&self, _design: &mut Design) -> Result<(), ()> {
        Ok(())
    }
}

use crate::npn::Truth6;
use prjunnamed_netlist::{
    Cell, Const, Design, Target, TargetCell, TargetImportError, TargetPrototype, Value,
};
use regex::Regex;
use std::collections::BTreeMap;
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

#[derive(Debug)]
pub struct StandardCell {
    pub name: String,
    pub ast: Statement,
    pub prototype: TargetPrototype,
    pub output_pin_asts: Vec<Statement>,
    pub input_pin_asts: Vec<Statement>,
    pub cm_function: Vec<Option<Truth6>>,
    pub area: f32,
}

#[derive(Debug)]
pub struct SCLTarget {
    pub cells: BTreeMap<String, StandardCell>,
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
fn parse_expr<I>(chars: &mut Peekable<I>, pins: &Vec<&str>, priority: usize) -> Option<Truth6>
where
    I: Iterator<Item = char>,
{
    while matches!(chars.peek(), Some(' ') | Some('\t')) {
        chars.next();
    }
    assert!(chars.peek() != None);

    let mask: Truth6 = (1 as Truth6)
        .checked_shl(1 << pins.len())
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

            let pin_idx = pins.iter().position(|&p| p == pin)?;

            lhs = 0;
            for i in 0..(1 << pins.len()) {
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
            lhs = parse_expr(chars, pins, 0)?;
            assert_eq!(chars.next(), Some(')'));
        }
        '!' => {
            chars.next();
            lhs = !parse_expr(chars, pins, 7)? & mask;
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
                lhs ^= parse_expr(chars, pins, 6)?;
                continue;
            }
            '&' | '*' => {
                // TODO: support whitespace as infix AND
                if priority > 3 {
                    break;
                }

                chars.next();
                lhs &= parse_expr(chars, pins, 4)?;
                continue;
            }

            '+' | '|' => {
                if priority > 1 {
                    break;
                }

                chars.next();
                lhs |= parse_expr(chars, pins, 2)?;
                continue;
            }

            _ => {
                break;
            }
        }
    }

    return Some(lhs);
}

impl SCLTarget {
    pub fn new() -> Self {
        SCLTarget {
            cells: BTreeMap::new(),
        }
    }

    pub fn read_liberty(&mut self, text: String) {
        let mut top_group = {
            let mut scanner = Scanner::wrap(text.chars());
            parse_statement(&mut scanner)
        };
        top_group.sort();

        let Statement::Group(text, attrs, top_statements) = top_group else {
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
            if let Statement::Group(ref name, ref attrs, _) = statement {
                if name == "cell" {
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

                    let mut input_names: Vec<&str> = Vec::new();
                    let mut input_pin_asts: Vec<Statement> = Vec::new();
                    let mut output_pin_asts: Vec<Statement> = Vec::new();

                    for pin_stmt in statement.lookup_all(&"pin") {
                        let Statement::Group(_name, pin_attrs, _) = pin_stmt else {
                            panic!("bad pin (1)");
                        };
                        if pin_attrs.len() != 1 {
                            panic!("bad pin (2)");
                        }
                        let [ref name] = pin_attrs[..] else {
                            panic!("bad pin (3)");
                        };

                        match pin_stmt.find_attribute("direction") {
                            Some("input") => {
                                input_names.push(&name);
                                input_pin_asts.push(pin_stmt.clone());
                                proto = proto.add_input(name, Const::undef(1));
                            }
                            Some("output") => {
                                output_pin_asts.push(pin_stmt.clone());
                                proto = proto.add_output(name, 1);
                            }
                            Some("inout") => {}
                            Some("internal") => {}
                            Some(d @ _) => {
                                panic!("unknown direction {}", d)
                            }
                            _ => {
                                panic!("no direction")
                            }
                        }
                    }

                    // FIXME: check for conflict
                    self.cells.insert(
                        name.clone(),
                        StandardCell {
                            name: name.clone(),
                            ast: statement.clone(),
                            prototype: proto,
                            // use Lut or some other type here, no reason for LUT-6 limit
                            cm_function: output_pin_asts
                                .iter()
                                .map(|pin| {
                                    pin.find_attribute("function").and_then(|text| {
                                        let mut iter = text.chars().peekable();
                                        parse_expr(&mut iter, &input_names, 0)
                                    })
                                })
                                .collect::<Vec<Option<Truth6>>>(),
                            area: statement
                                .find_attribute("area")
                                .and_then(|v| v.parse::<f32>().ok())
                                .unwrap(),
                            input_pin_asts: input_pin_asts,
                            output_pin_asts: output_pin_asts,
                        },
                    );
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

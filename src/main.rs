use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::str::FromStr;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut last_result = 0.0;
    let mut result_idx = 0;

    let globals = Globals::new();

    let config = rustyline::config::Builder::new()
        .edit_mode(rustyline::config::EditMode::Vi)
        .build();

    let mut editor = rustyline::Editor::<LineHelper>::with_config(config);

    let mut context = Context::new();
    let helper = LineHelper::new(globals.clone());
    editor.set_helper(Some(helper));
    editor.bind_sequence(
        rustyline::KeyEvent::new('\t', rustyline::Modifiers::NONE),
        rustyline::Cmd::CompleteHint,
    );

    loop {
        let line = if let Ok(line) = editor.readline("> ") {
            line
        } else {
            break;
        };

        if line.trim() == "quit()" {
            break;
        }

        editor.add_history_entry(line.trim());

        let expression: Expression = line.parse()?;

        let result = expression.evaluate(&mut context, &globals, last_result);

        let global_name = format!("@{}", result_idx);
        editor.bind_sequence(
            rustyline::KeyEvent::new('\t', rustyline::Modifiers::SHIFT),
            rustyline::Cmd::Insert(1, format!("{} ", global_name)),
        );

        globals.insert(global_name, Global::Constant(result));
        last_result = result;

        println!("@{} = {}", result_idx, DisplayFloat(result, Some(&context)));
        result_idx += 1;
    }

    Ok(())
}

macro_rules! create_global_function(
    ($globals:ident, $name:ident, $arg_count:literal, $body:expr) => {
        pub fn $name(args: &[f64]) -> f64 {
            if args.len() < $arg_count {
                f64::NAN
            } else {
                let f: fn(&[f64]) -> f64 = $body;
                f(args)
            }
        }

        $globals.insert(stringify!($name).to_string(), Global::Function($arg_count, $name));
    };
    ($globals:ident, $name:ident, $body:expr) => {
        pub fn $name(_args: &[f64]) -> f64 {
            $body
        }

        $globals.insert(stringify!($name).to_string(), Global::Function(0, $name));
    };
);

#[derive(Clone)]
struct Globals {
    map: Rc<RefCell<HashMap<String, Global>>>,
}

impl Globals {
    fn new() -> Self {
        let mut globals = HashMap::new();

        create_global_function!(globals, sin, 1, |args| args[0].sin());
        create_global_function!(globals, cos, 1, |args| args[0].cos());
        create_global_function!(globals, tan, 1, |args| args[0].tan());

        create_global_function!(globals, asin, 1, |args| args[0].asin());
        create_global_function!(globals, acos, 1, |args| args[0].acos());
        create_global_function!(globals, atan, 1, |args| args[0].atan());

        create_global_function!(globals, sinh, 1, |args| args[0].sinh());
        create_global_function!(globals, cosh, 1, |args| args[0].cosh());
        create_global_function!(globals, tanh, 1, |args| args[0].tanh());

        create_global_function!(globals, asinh, 1, |args| args[0].asinh());
        create_global_function!(globals, acosh, 1, |args| args[0].acosh());
        create_global_function!(globals, atanh, 1, |args| args[0].atanh());

        create_global_function!(globals, sqrt, 1, |args| args[0].sqrt());
        create_global_function!(globals, ln, 1, |args| args[0].ln());
        create_global_function!(globals, floor, 1, |args| args[0].floor());
        create_global_function!(globals, ceil, 1, |args| args[0].ceil());
        create_global_function!(globals, round, 1, |args| args[0].round());
        create_global_function!(globals, abs, 1, |args| args[0].abs());
        create_global_function!(globals, log, 2, |args| args[0].log(args[1]));
        create_global_function!(globals, min, 2, |args| args[0].min(args[1]));
        create_global_function!(globals, max, 2, |args| args[0].max(args[1]));
        create_global_function!(globals, rand, rand::random());
        create_global_function!(globals, quit, 0.0);

        create_global_function!(globals, f_to_c, 1, |args| (args[0] - 32.0) / 1.8);
        create_global_function!(globals, c_to_f, 1, |args| args[0] * 1.8 + 32.0);
        create_global_function!(globals, lb_to_kg, 1, |args| args[0] * 0.45359237);
        create_global_function!(globals, kg_to_lb, 1, |args| args[0] / 0.45359237);
        create_global_function!(globals, mi_to_km, 1, |args| args[0] * 1.609344);
        create_global_function!(globals, km_to_mi, 1, |args| args[0] / 1.609344);
        create_global_function!(globals, ft_to_m, 1, |args| args[0] * 0.3048);
        create_global_function!(globals, m_to_ft, 1, |args| args[0] / 0.3048);

        globals.insert("PI".to_string(), Global::Constant(std::f64::consts::PI));
        globals.insert("TAU".to_string(), Global::Constant(std::f64::consts::TAU));
        globals.insert("E".to_string(), Global::Constant(std::f64::consts::E));
        globals.insert("NAN".to_string(), Global::Constant(std::f64::NAN));
        globals.insert("INF".to_string(), Global::Constant(std::f64::INFINITY));

        let map = Rc::new(RefCell::new(globals));

        Self { map }
    }

    pub fn insert<S: Into<String>>(&self, name: S, value: Global) {
        self.map.borrow_mut().insert(name.into(), value);
    }

    pub fn borrow<'a>(&'a self) -> impl std::ops::Deref<Target = HashMap<String, Global>> + 'a {
        self.map.borrow()
    }

    pub fn get<S: AsRef<str>>(&self, name: S) -> Option<Global> {
        self.map.borrow().get(name.as_ref()).copied()
    }
}

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
enum Mode {
    Hex,
    Binary,
    Integer,
    Float,
}

#[derive(Debug, Clone)]
struct Context {
    mode: Mode,
}

impl Context {
    fn new() -> Self {
        Self { mode: Mode::Float }
    }

    fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
    }

    fn get_mode(&self) -> Mode {
        self.mode
    }
}

struct LineHelper {
    globals: Globals,
    highlighter: rustyline::highlight::MatchingBracketHighlighter,
}

impl LineHelper {
    fn new(globals: Globals) -> Self {
        let highlighter = rustyline::highlight::MatchingBracketHighlighter::new();

        Self {
            globals,
            highlighter,
        }
    }
}

impl rustyline::Helper for LineHelper {}

impl rustyline::highlight::Highlighter for LineHelper {
    fn highlight<'l>(&self, line: &'l str, pos: usize) -> std::borrow::Cow<'l, str> {
        self.highlighter.highlight(line, pos)
    }

    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        default: bool,
    ) -> std::borrow::Cow<'b, str> {
        self.highlighter.highlight_prompt(prompt, default)
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> std::borrow::Cow<'h, str> {
        format!("{}\x1b[1;90m{}\x1b[0m", &hint[0..1], &hint[1..]).into()
    }

    fn highlight_candidate<'c>(
        &self,
        candidate: &'c str,
        completion: rustyline::CompletionType,
    ) -> std::borrow::Cow<'c, str> {
        self.highlighter.highlight_candidate(candidate, completion)
    }

    fn highlight_char(&self, line: &str, pos: usize) -> bool {
        self.highlighter.highlight_char(line, pos)
    }
}

impl rustyline::completion::Completer for LineHelper {
    type Candidate = String;
}

impl rustyline::hint::Hinter for LineHelper {
    type Hint = GlobalHint;

    fn hint(&self, line: &str, _pos: usize, _ctx: &rustyline::Context<'_>) -> Option<Self::Hint> {
        let words = line.split(|c: char| {
            c != '_' && c != '$' && c != '#' && c != '@' && !c.is_alphabetic() && !c.is_digit(10)
        });
        let final_word = words.last();
        if let Some(final_word) = final_word {
            if final_word.len() == 0 {
                return None;
            }

            for (name, value) in self.globals.borrow().iter() {
                if name.starts_with(final_word) {
                    return Some(GlobalHint::new(final_word.len(), name.to_string(), &value));
                }
            }
        }

        None
    }
}

#[derive(Debug, Copy, Clone)]
struct DisplayFloat<'a>(f64, Option<&'a Context>);

impl<'a> std::fmt::Display for DisplayFloat<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.1.map(|ctx| ctx.get_mode()).unwrap_or(Mode::Float) {
            Mode::Hex => {
                let v = self.0 as i128;
                write!(f, "${:X}", v)
            }
            Mode::Binary => {
                let v = self.0 as i128;
                write!(f, "b{:b}", v)
            }
            Mode::Integer => {
                let v = self.0 as i128;
                write!(f, "{}", v)
            }
            Mode::Float => {
                if self.0 > 10e20
                    || (self.0 < 10e-20 && self.0 > 0.0)
                    || (self.0 > -10e-20 && self.0 < 0.0)
                    || self.0 < -10e20
                {
                    write!(f, "{:e}", self.0)
                } else {
                    write!(f, "{}", self.0)
                }
            }
        }
    }
}

struct GlobalHint {
    display: String,
    complete: String,
}

impl GlobalHint {
    fn new(offset: usize, name: String, global: &Global) -> Self {
        let display = match global {
            Global::Function(0, _) => format!("{}()", name),
            Global::Function(1, _) => format!("{}( x )", name),
            Global::Function(2, _) => format!("{}( x, y )", name),
            Global::Function(3, _) => format!("{}( x, y, z )", name),
            Global::Function(_, _) => format!("{}( ... )", name),
            Global::Constant(v) => format!("{} = {}", name, DisplayFloat(*v, None)),
        };

        let complete = match global {
            Global::Function(0, _) => format!("{}()", name),
            Global::Function(_, _) => format!("{}(", name),
            Global::Constant(_) => format!("{}", name),
        };

        let display = display[offset..].to_string();
        let complete = complete[offset..].to_string();

        Self { display, complete }
    }
}

impl rustyline::hint::Hint for GlobalHint {
    fn display(&self) -> &str {
        self.display.as_str()
    }

    fn completion(&self) -> Option<&str> {
        Some(self.complete.as_str())
    }
}

impl rustyline::validate::Validator for LineHelper {}

#[derive(Copy, Clone)]
enum Global {
    Function(usize, fn(&[f64]) -> f64),
    Constant(f64),
}

#[derive(Debug, Clone)]
struct Expression {
    tokens: Vec<Token>,
}

impl Expression {
    fn evaluate(self, context: &mut Context, globals: &Globals, last_result: f64) -> f64 {
        let tokens = self.tokens;
        let mut output = Vec::with_capacity(tokens.len());
        output.push(Token::Number(last_result));
        let mut operators = Vec::with_capacity(tokens.len());
        for token in tokens {
            match token {
                Token::Number(n) => output.push(Token::Number(n)),
                Token::Operator(Operator::OpenParen) => {
                    let prev = output.pop();
                    match prev {
                        Some(Token::Identifier(ident)) => {
                            operators.push(Operator::Function(ident));
                        }
                        Some(other) => {
                            operators.push(Operator::OpenParen);
                            output.push(other);
                        }
                        None => {}
                    }
                }
                Token::Operator(Operator::CloseParen) => {
                    while let Some(op) = operators.pop() {
                        match op {
                            Operator::OpenParen => {
                                break;
                            }
                            Operator::Function(ident) => {
                                output.push(Token::Operator(Operator::Function(ident)));
                                break;
                            }
                            op => output.push(Token::Operator(op)),
                        }
                    }
                }
                Token::Operator(Operator::SetMode(mode)) => context.set_mode(mode),
                Token::Operator(op) => {
                    while let Some(next_op) = operators.last() {
                        match next_op {
                            Operator::OpenParen | Operator::CloseParen => {
                                break;
                            }
                            next_op if next_op.precedence() >= op.precedence() => {
                                let next_op = operators.pop().unwrap();
                                output.push(Token::Operator(next_op));
                            }
                            _ => {
                                break;
                            }
                        }
                    }
                    operators.push(op)
                }
                Token::Identifier(ident) => {
                    output.push(Token::Identifier(ident));
                }
                Token::Unknown(_) => {}
            }
        }

        while let Some(op) = operators.pop() {
            output.push(Token::Operator(op));
        }

        let mut eval_stack = Vec::with_capacity(output.len());
        for token in output {
            match token {
                Token::Number(n) => eval_stack.push(n),
                Token::Operator(Operator::Add) => {
                    let rhs = eval_stack.pop().unwrap_or(0.0);
                    let lhs = eval_stack.pop().unwrap_or(0.0);
                    eval_stack.push(lhs + rhs);
                }
                Token::Operator(Operator::Sub) => {
                    let rhs = eval_stack.pop().unwrap_or(0.0);
                    let lhs = eval_stack.pop().unwrap_or(0.0);
                    eval_stack.push(lhs - rhs);
                }
                Token::Operator(Operator::Mul) => {
                    let rhs = eval_stack.pop().unwrap_or(0.0);
                    let lhs = eval_stack.pop().unwrap_or(0.0);
                    eval_stack.push(lhs * rhs);
                }
                Token::Operator(Operator::Div) => {
                    let rhs = eval_stack.pop().unwrap_or(0.0);
                    let lhs = eval_stack.pop().unwrap_or(0.0);
                    if context.get_mode() == Mode::Float {
                        eval_stack.push(lhs / rhs);
                    } else {
                        let lhs = lhs as i128;
                        let rhs = rhs as i128;
                        eval_stack.push((lhs / rhs) as f64);
                    }
                }
                Token::Operator(Operator::Pow) => {
                    let rhs = eval_stack.pop().unwrap_or(0.0);
                    let lhs = eval_stack.pop().unwrap_or(0.0);
                    eval_stack.push(lhs.powf(rhs));
                }
                Token::Operator(Operator::Mod) => {
                    let rhs = eval_stack.pop().unwrap_or(0.0);
                    let lhs = eval_stack.pop().unwrap_or(0.0);
                    eval_stack.push(lhs % rhs);
                }
                Token::Operator(Operator::Neg) => {
                    let val = eval_stack.pop().unwrap_or(0.0);
                    eval_stack.push(-val);
                }
                Token::Operator(Operator::BitAnd) => {
                    let rhs = eval_stack.pop().unwrap_or(0.0) as i128;
                    let lhs = eval_stack.pop().unwrap_or(0.0) as i128;
                    eval_stack.push((lhs & rhs) as f64);
                }
                Token::Operator(Operator::BitXor) => {
                    let rhs = eval_stack.pop().unwrap_or(0.0) as i128;
                    let lhs = eval_stack.pop().unwrap_or(0.0) as i128;
                    eval_stack.push((lhs ^ rhs) as f64);
                }
                Token::Operator(Operator::BitOr) => {
                    let rhs = eval_stack.pop().unwrap_or(0.0) as i128;
                    let lhs = eval_stack.pop().unwrap_or(0.0) as i128;
                    eval_stack.push((lhs | rhs) as f64);
                }
                Token::Operator(Operator::Comma) => (),
                Token::Operator(Operator::Function(ident)) => {
                    if let Some(Global::Function(arg_count, func)) = globals.get(ident.as_str()) {
                        let mut args = Vec::with_capacity(arg_count);
                        for _ in 0..arg_count {
                            args.push(eval_stack.pop().unwrap_or(0.0));
                        }

                        args.reverse();

                        let result = func(&args);
                        eval_stack.push(result);
                    }
                }
                Token::Identifier(ident) => {
                    if let Some(Global::Constant(value)) = globals.get(ident.as_str()) {
                        eval_stack.push(value);
                    } else {
                        eval_stack.push(std::f64::NAN);
                    }
                }
                Token::Unknown(_) => panic!("unknown token"),
                _ => unreachable!("invalid rpn"),
            }
        }

        eval_stack.pop().unwrap_or(0.0)
    }
}

impl FromStr for Expression {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut tokens = Vec::new();
        let mut chars = s.chars().fuse().peekable();
        let mut neg = true;

        loop {
            let next = chars.peek();
            if let Some(next) = next {
                let token = match next {
                    '(' => {
                        let _ = chars.next();
                        Some(Token::Operator(Operator::OpenParen))
                    }
                    ')' => {
                        let _ = chars.next();
                        Some(Token::Operator(Operator::CloseParen))
                    }
                    '+' => {
                        let _ = chars.next();
                        Some(Token::Operator(Operator::Add))
                    }
                    '-' if neg => {
                        let _ = chars.next();
                        Some(Token::Operator(Operator::Neg))
                    }
                    '-' => {
                        let _ = chars.next();
                        Some(Token::Operator(Operator::Sub))
                    }
                    '*' => {
                        let _ = chars.next();
                        Some(Token::Operator(Operator::Mul))
                    }
                    '/' => {
                        let _ = chars.next();
                        Some(Token::Operator(Operator::Div))
                    }
                    '%' => {
                        let _ = chars.next();
                        Some(Token::Operator(Operator::Mod))
                    }
                    '^' => {
                        let _ = chars.next();
                        Some(Token::Operator(Operator::Pow))
                    }
                    ',' => {
                        let _ = chars.next();
                        Some(Token::Operator(Operator::Comma))
                    }
                    '&' => {
                        let _ = chars.next();
                        Some(Token::Operator(Operator::BitAnd))
                    }
                    '~' => {
                        let _ = chars.next();
                        Some(Token::Operator(Operator::BitXor))
                    }
                    '|' => {
                        let _ = chars.next();
                        Some(Token::Operator(Operator::BitOr))
                    }
                    '#' => {
                        let _ = chars.next();
                        let flag = chars.next();
                        if let Some(flag) = flag {
                            match flag {
                                'i' => Some(Token::Operator(Operator::SetMode(Mode::Integer))),
                                'h' => Some(Token::Operator(Operator::SetMode(Mode::Hex))),
                                'b' => Some(Token::Operator(Operator::SetMode(Mode::Binary))),
                                'f' => Some(Token::Operator(Operator::SetMode(Mode::Float))),
                                c => Some(Token::Unknown(c)),
                            }
                        } else {
                            None
                        }
                    }
                    &d if d.is_token_number() => {
                        let mut result: f64 = 0.0;
                        let mut scale = 0;
                        let mut fractional = false;

                        while chars
                            .peek()
                            .map(|c| c.is_token_number() && (*c != '.' || !fractional))
                            .unwrap_or(false)
                        {
                            match chars.next() {
                                Some('.') if !fractional => {
                                    fractional = true;
                                }
                                Some('.') if fractional => unreachable!(),
                                Some(d) if d.is_digit(10) => {
                                    let digit = (d as u32 - '0' as u32) as f64;
                                    result = result.mul_add(10.0, digit);
                                    if fractional {
                                        scale += 1;
                                    }
                                }
                                _ => unreachable!(),
                            }
                        }

                        result /= (10.0f64).powi(scale);

                        Some(Token::Number(result))
                    }
                    c if c.is_token_identifier() => {
                        let mut s = String::new();

                        while chars
                            .peek()
                            .map(|c| c.is_token_identifier())
                            .unwrap_or(false)
                        {
                            if let Some(c) = chars.next() {
                                s.push(c);
                            }
                        }

                        Some(Token::Identifier(s))
                    }
                    w if w.is_token_whitespace() => {
                        let _ = chars.next();
                        None
                    }
                    c if *c == '@' => {
                        let mut s = String::from("@");

                        let _ = chars.next();

                        while chars.peek().map(|c| c.is_digit(10)).unwrap_or(false) {
                            if let Some(c) = chars.next() {
                                s.push(c);
                            }
                        }

                        Some(Token::Identifier(s))
                    }
                    c if *c == '$' => {
                        let mut value = 0;

                        let _ = chars.next();

                        while let Some(digit) = chars.peek().and_then(|c| c.to_digit(16)) {
                            let _ = chars.next();
                            value *= 16;
                            value += digit;
                        }

                        Some(Token::Number(value as f64))
                    }
                    c => {
                        let c = *c;
                        let _ = chars.next();
                        Some(Token::Unknown(c))
                    }
                };

                if let Some(token) = token {
                    match token {
                        Token::Number(_) => neg = false,
                        Token::Identifier(_) => neg = false,
                        Token::Operator(Operator::CloseParen) => neg = false,
                        Token::Operator(_) => neg = true,
                        Token::Unknown(_) => (),
                    }
                    tokens.push(token);
                }
            } else {
                break;
            }
        }

        Ok(Expression { tokens })
    }
}

trait CharExt {
    fn is_token_number(&self) -> bool;
    fn is_token_identifier(&self) -> bool;
    fn is_token_whitespace(&self) -> bool;
}

impl CharExt for char {
    fn is_token_number(&self) -> bool {
        self.is_digit(10) || *self == '.'
    }

    fn is_token_identifier(&self) -> bool {
        self.is_alphabetic() || *self == '_'
    }

    fn is_token_whitespace(&self) -> bool {
        self.is_whitespace()
    }
}

#[derive(Debug)]
enum ParseError {}

impl std::error::Error for ParseError {}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parse error")
    }
}

#[derive(Debug, Clone)]
enum Operator {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Neg,
    BitOr,
    BitAnd,
    BitXor,
    OpenParen,
    CloseParen,
    Function(String),
    Comma,
    SetMode(Mode),
}

impl Operator {
    fn precedence(&self) -> i32 {
        match self {
            Operator::Add => 10,
            Operator::Sub => 10,
            Operator::Mul => 100,
            Operator::Div => 100,
            Operator::Mod => 100,
            Operator::Pow => 1000,
            Operator::Neg => 10000,
            Operator::Comma => 1,
            Operator::BitAnd => 5,
            Operator::BitXor => 4,
            Operator::BitOr => 3,
            _ => 0,
        }
    }
}

#[derive(Debug, Clone)]
enum Token {
    Number(f64),
    Identifier(String),
    Operator(Operator),
    Unknown(char),
}

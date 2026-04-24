use std::collections::HashSet;

use regex::Regex;

use super::KeywordError;

const DANGEROUS_KEYWORDS: &[&str] = &[
    "SELECT", "UNION", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "EXEC",
    "EXECUTE", "GRANT", "REVOKE",
];

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Identifier(String),
    Placeholder,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Like,
    Regexp,
    Between,
    In,
    And,
    Or,
    Not,
    Is,
    Null,
    LParen,
    RParen,
    Comma,
    Eof,
}

fn quick_safety_check(condition: &str) -> Result<(), KeywordError> {
    let upper = condition.to_uppercase();

    if condition.contains("--") || condition.contains("/*") || condition.contains("*/") {
        return Err(KeywordError::SqlCommentsNotAllowed);
    }

    if condition.contains(';') {
        return Err(KeywordError::SqlSemicolonNotAllowed);
    }

    for keyword in DANGEROUS_KEYWORDS {
        let pattern = Regex::new(&format!(r"\b{}\b", keyword))?;
        if pattern.is_match(&upper) {
            return Err(KeywordError::SqlKeywordNotAllowed((*keyword).to_string()));
        }
    }

    Ok(())
}

fn tokenize(input: &str) -> Result<Vec<Token>, KeywordError> {
    let chars: Vec<char> = input.chars().collect();
    let mut tokens = Vec::new();
    let mut position = 0;

    while position < chars.len() {
        if chars[position].is_whitespace() {
            position += 1;
            continue;
        }

        match chars[position] {
            '?' => {
                tokens.push(Token::Placeholder);
                position += 1;
                continue;
            }
            '(' => {
                tokens.push(Token::LParen);
                position += 1;
                continue;
            }
            ')' => {
                tokens.push(Token::RParen);
                position += 1;
                continue;
            }
            ',' => {
                tokens.push(Token::Comma);
                position += 1;
                continue;
            }
            '=' => {
                tokens.push(Token::Eq);
                position += 1;
                continue;
            }
            _ => {}
        }

        if position + 1 < chars.len() {
            let two_chars: String = chars[position..position + 2].iter().collect();
            match two_chars.as_str() {
                "!=" | "<>" => {
                    tokens.push(Token::Ne);
                    position += 2;
                    continue;
                }
                "<=" => {
                    tokens.push(Token::Le);
                    position += 2;
                    continue;
                }
                ">=" => {
                    tokens.push(Token::Ge);
                    position += 2;
                    continue;
                }
                _ => {}
            }
        }

        match chars[position] {
            '<' => {
                tokens.push(Token::Lt);
                position += 1;
                continue;
            }
            '>' => {
                tokens.push(Token::Gt);
                position += 1;
                continue;
            }
            _ => {}
        }

        if chars[position].is_ascii_alphabetic() || chars[position] == '_' {
            let start = position;
            while position < chars.len()
                && (chars[position].is_ascii_alphanumeric() || chars[position] == '_')
            {
                position += 1;
            }

            let word: String = chars[start..position].iter().collect();
            let upper = word.to_uppercase();
            let token = match upper.as_str() {
                "AND" => Token::And,
                "OR" => Token::Or,
                "NOT" => Token::Not,
                "IS" => Token::Is,
                "NULL" => Token::Null,
                "LIKE" => Token::Like,
                "REGEXP" => Token::Regexp,
                "BETWEEN" => Token::Between,
                "IN" => Token::In,
                _ => Token::Identifier(word),
            };
            tokens.push(token);
            continue;
        }

        if chars[position] == '"' {
            position += 1;
            let start = position;
            while position < chars.len() && chars[position] != '"' {
                position += 1;
            }

            if position >= chars.len() {
                return Err(KeywordError::UnterminatedQuotedIdentifier);
            }

            let word: String = chars[start..position].iter().collect();
            tokens.push(Token::Identifier(word));
            position += 1;
            continue;
        }

        return Err(KeywordError::UnexpectedCharacter(chars[position]));
    }

    tokens.push(Token::Eof);
    Ok(tokens)
}

struct ConditionValidator<'a> {
    tokens: &'a [Token],
    position: usize,
    valid_columns: &'a HashSet<String>,
}

impl<'a> ConditionValidator<'a> {
    fn new(tokens: &'a [Token], valid_columns: &'a HashSet<String>) -> Self {
        Self {
            tokens,
            position: 0,
            valid_columns,
        }
    }

    fn current(&self) -> &Token {
        self.tokens.get(self.position).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) {
        if self.position < self.tokens.len() {
            self.position += 1;
        }
    }

    fn expect(&mut self, expected: &Token) -> Result<(), KeywordError> {
        if self.current() == expected {
            self.advance();
            Ok(())
        } else {
            Err(KeywordError::ConditionParseError {
                expected: token_label(expected),
                found: format!("{:?}", self.current()),
            })
        }
    }

    fn validate(&mut self) -> Result<(), KeywordError> {
        self.parse_expr()?;
        if *self.current() != Token::Eof {
            return Err(KeywordError::UnexpectedTrailingToken(format!(
                "{:?}",
                self.current()
            )));
        }
        Ok(())
    }

    fn parse_expr(&mut self) -> Result<(), KeywordError> {
        self.parse_and_expr()?;
        while *self.current() == Token::Or {
            self.advance();
            self.parse_and_expr()?;
        }
        Ok(())
    }

    fn parse_and_expr(&mut self) -> Result<(), KeywordError> {
        self.parse_unary_expr()?;
        while *self.current() == Token::And {
            self.advance();
            self.parse_unary_expr()?;
        }
        Ok(())
    }

    fn parse_unary_expr(&mut self) -> Result<(), KeywordError> {
        if *self.current() == Token::Not {
            self.advance();
        }
        self.parse_primary_expr()
    }

    fn parse_primary_expr(&mut self) -> Result<(), KeywordError> {
        if *self.current() == Token::LParen {
            self.advance();
            self.parse_expr()?;
            self.expect(&Token::RParen)?;
            return Ok(());
        }

        let column_name = match self.current().clone() {
            Token::Identifier(name) => name,
            other => {
                return Err(KeywordError::ConditionParseError {
                    expected: "column name",
                    found: format!("{:?}", other),
                })
            }
        };

        let column_name_lower = column_name.to_lowercase();
        let valid = self
            .valid_columns
            .iter()
            .any(|column| column.to_lowercase() == column_name_lower);
        if !valid {
            return Err(KeywordError::UnknownColumn(column_name));
        }
        self.advance();

        match self.current() {
            Token::Is => {
                self.advance();
                if *self.current() == Token::Not {
                    self.advance();
                }
                self.expect(&Token::Null)?;
            }
            Token::Not => {
                self.advance();
                match self.current() {
                    Token::Between => {
                        self.advance();
                        self.expect(&Token::Placeholder)?;
                        self.expect(&Token::And)?;
                        self.expect(&Token::Placeholder)?;
                    }
                    Token::In => {
                        self.advance();
                        self.parse_in_list()?;
                    }
                    Token::Like => {
                        self.advance();
                        self.expect(&Token::Placeholder)?;
                    }
                    Token::Regexp => {
                        self.advance();
                        self.expect(&Token::Placeholder)?;
                    }
                    other => {
                        return Err(KeywordError::ConditionParseError {
                            expected: "BETWEEN, IN, LIKE, or REGEXP after NOT",
                            found: format!("{:?}", other),
                        })
                    }
                }
            }
            Token::Between => {
                self.advance();
                self.expect(&Token::Placeholder)?;
                self.expect(&Token::And)?;
                self.expect(&Token::Placeholder)?;
            }
            Token::In => {
                self.advance();
                self.parse_in_list()?;
            }
            Token::Like => {
                self.advance();
                self.expect(&Token::Placeholder)?;
            }
            Token::Regexp => {
                self.advance();
                self.expect(&Token::Placeholder)?;
            }
            Token::Eq | Token::Ne | Token::Lt | Token::Le | Token::Gt | Token::Ge => {
                self.advance();
                self.expect(&Token::Placeholder)?;
            }
            other => {
                return Err(KeywordError::ConditionParseError {
                    expected: "operator after column name",
                    found: format!("{:?}", other),
                })
            }
        }

        Ok(())
    }

    fn parse_in_list(&mut self) -> Result<(), KeywordError> {
        self.expect(&Token::LParen)?;
        self.expect(&Token::Placeholder)?;
        while *self.current() == Token::Comma {
            self.advance();
            self.expect(&Token::Placeholder)?;
        }
        self.expect(&Token::RParen)
    }
}

fn token_label(token: &Token) -> &'static str {
    match token {
        Token::Identifier(_) => "identifier",
        Token::Placeholder => "placeholder",
        Token::Eq => "=",
        Token::Ne => "!=",
        Token::Lt => "<",
        Token::Le => "<=",
        Token::Gt => ">",
        Token::Ge => ">=",
        Token::Like => "LIKE",
        Token::Regexp => "REGEXP",
        Token::Between => "BETWEEN",
        Token::In => "IN",
        Token::And => "AND",
        Token::Or => "OR",
        Token::Not => "NOT",
        Token::Is => "IS",
        Token::Null => "NULL",
        Token::LParen => "(",
        Token::RParen => ")",
        Token::Comma => ",",
        Token::Eof => "end of input",
    }
}

fn is_numeric_equality(condition: &str) -> bool {
    Regex::new(r"^(\d+)\s*=\s*(\d+)$")
        .map(|regex| regex.is_match(condition.trim()))
        .unwrap_or(false)
}

pub(super) fn validate_condition(
    condition: &str,
    valid_columns: &HashSet<String>,
) -> Result<(), KeywordError> {
    if is_numeric_equality(condition) {
        return Ok(());
    }

    quick_safety_check(condition)?;
    let tokens = tokenize(condition)?;
    let mut validator = ConditionValidator::new(&tokens, valid_columns);
    validator.validate()
}

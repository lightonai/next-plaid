//! Tests for SQL code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_create_table() {
    let source = r#"CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255)
);
"#;
    let units = parse(source, Language::Sql, "test.sql");

    assert!(!units.is_empty(), "Should extract CREATE TABLE");

    let unit = &units[0];
    let text = build_embedding_text(unit);

    let expected = r#"CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(255),
    email VARCHAR(255)
);"#;
    assert_eq!(text, expected);
}

#[test]
fn test_create_table_with_comment() {
    let source = r#"-- User information table
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(255)
);
"#;
    let units = parse(source, Language::Sql, "test.sql");

    assert!(
        !units.is_empty(),
        "Should extract CREATE TABLE with comment"
    );

    let unit = &units[0];
    let text = build_embedding_text(unit);

    let expected = r#"-- User information table
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(255)
);"#;
    assert_eq!(text, expected);
}

#[test]
fn test_select_query() {
    let source = r#"SELECT id, name, email
FROM users
WHERE active = 1
ORDER BY name;
"#;
    let units = parse(source, Language::Sql, "test.sql");

    assert!(!units.is_empty(), "Should extract SELECT query");

    let unit = &units[0];
    let text = build_embedding_text(unit);
    println!("EMBEDDING TEXT:\n{}", text);
    println!("---END---");
}

#[test]
fn test_create_view() {
    let source = r#"
CREATE VIEW active_users AS
SELECT id, name, email
FROM users
WHERE active = 1;
"#;
    let units = parse(source, Language::Sql, "test.sql");

    assert!(!units.is_empty(), "Should extract CREATE VIEW");
}

#[test]
fn test_stored_procedure_style() {
    let source = r#"
CREATE FUNCTION get_user_count()
RETURNS INT
AS
BEGIN
    RETURN (SELECT COUNT(*) FROM users);
END;
"#;
    let units = parse(source, Language::Sql, "test.sql");

    assert!(!units.is_empty(), "Should extract stored function");
}

#[test]
fn test_insert_statement() {
    let source = r#"
INSERT INTO users (name, email)
VALUES ('John', 'john@example.com');
"#;
    let units = parse(source, Language::Sql, "test.sql");

    assert!(!units.is_empty(), "Should extract INSERT statement");
}

#[test]
fn test_update_statement() {
    let source = r#"
UPDATE users
SET active = 1
WHERE id = 1;
"#;
    let units = parse(source, Language::Sql, "test.sql");

    assert!(!units.is_empty(), "Should extract UPDATE statement");
}

#[test]
fn test_create_index() {
    let source = r#"
CREATE INDEX idx_users_email ON users(email);
CREATE UNIQUE INDEX idx_users_name ON users(name);
"#;
    let units = parse(source, Language::Sql, "test.sql");

    assert!(!units.is_empty(), "Should extract CREATE INDEX");
}

#[test]
fn test_complex_query_with_joins() {
    let source = r#"
SELECT u.name, o.total
FROM users u
INNER JOIN orders o ON u.id = o.user_id
LEFT JOIN products p ON o.product_id = p.id
WHERE o.created_at > '2024-01-01'
GROUP BY u.name
HAVING COUNT(*) > 5
ORDER BY o.total DESC;
"#;
    let units = parse(source, Language::Sql, "test.sql");

    assert!(!units.is_empty(), "Should extract complex query");
}

#[test]
fn test_trigger() {
    let source = r#"
CREATE TRIGGER update_timestamp
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    SET NEW.updated_at = NOW();
END;
"#;
    let units = parse(source, Language::Sql, "test.sql");

    assert!(!units.is_empty(), "Should extract trigger");
}

#[test]
fn test_multiple_statements() {
    let source = r#"
CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT,
    total DECIMAL(10,2)
);

CREATE TABLE products (
    id INT PRIMARY KEY,
    name VARCHAR(255),
    price DECIMAL(10,2)
);
"#;
    let units = parse(source, Language::Sql, "test.sql");

    assert!(!units.is_empty(), "Should extract multiple statements");
}

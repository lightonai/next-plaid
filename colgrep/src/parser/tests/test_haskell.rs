//! Tests for Haskell code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::Language;

#[test]
fn test_basic_function() {
    let source = r#"greet :: String -> String
greet name = "Hello, " ++ name ++ "!"
"#;
    let units = parse(source, Language::Haskell, "test.hs");

    let func = get_unit_by_name(&units, "greet").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: greet
Signature: greet name = "Hello, " ++ name ++ "!"
Code:
greet name = "Hello, " ++ name ++ "!"
File: test test.hs"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_haddock() {
    let source = r#"-- | Calculates the sum of two numbers.
add :: Int -> Int -> Int
add a b = a + b
"#;
    let units = parse(source, Language::Haskell, "test.hs");

    let func = get_unit_by_name(&units, "add").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: add
Signature: add a b = a + b
Code:
add a b = a + b
File: test test.hs"#;
    assert_eq!(text, expected);
}

#[test]
fn test_data_type() {
    let source = r#"data Person = Person { name :: String, age :: Int }
"#;
    let units = parse(source, Language::Haskell, "test.hs");

    assert_eq!(units.len(), 1);
    let unit = &units[0];
    let text = build_embedding_text(unit);
    let expected = r#"Code block: raw_code_1
Signature: data Person = Person { name :: String, age :: Int }
Code:
data Person = Person { name :: String, age :: Int }
File: test test.hs"#;
    assert_eq!(text, expected);
}

#[test]
fn test_type_class() {
    let source = r#"class Eq a => Ord a where
  compare :: a -> a -> Ordering
  (<) :: a -> a -> Bool
  (>) :: a -> a -> Bool
"#;
    let units = parse(source, Language::Haskell, "test.hs");

    assert_eq!(units.len(), 1);
    let unit = &units[0];
    let text = build_embedding_text(unit);
    let expected = r#"Code block: raw_code_1
Signature: class Eq a => Ord a where
Code:
class Eq a => Ord a where
  compare :: a -> a -> Ordering
  (<) :: a -> a -> Bool
  (>) :: a -> a -> Bool
File: test test.hs"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_guards() {
    let source = r#"abs :: Int -> Int
abs x
  | x >= 0    = x
  | otherwise = -x
"#;
    let units = parse(source, Language::Haskell, "test.hs");

    let func = get_unit_by_name(&units, "abs").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: abs
Signature: abs x
Code:
abs x
  | x >= 0    = x
  | otherwise = -x
File: test test.hs"#;
    assert_eq!(text, expected);
}

#[test]
fn test_newtype() {
    let source = r#"newtype UserId = UserId Int
"#;
    let units = parse(source, Language::Haskell, "test.hs");

    let unit = get_unit_by_name(&units, "UserId").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Class: UserId
Signature: newtype UserId = UserId Int
Code:
newtype UserId = UserId Int
File: test test.hs"#;
    assert_eq!(text, expected);
}

#[test]
fn test_function_with_pattern_matching() {
    let source = r#"length :: [a] -> Int
length [] = 0
length (_:xs) = 1 + length xs
"#;
    let units = parse(source, Language::Haskell, "test.hs");

    let func = get_unit_by_name(&units, "length").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: length
Signature: length [] = 0
Code:
length [] = 0
File: test test.hs"#;
    assert_eq!(text, expected);
}

#[test]
fn test_where_clause() {
    let source = r#"quadratic :: Double -> Double -> Double -> Double -> Double
quadratic a b c x = a*x^2 + b*x + c
  where
    square y = y * y
"#;
    let units = parse(source, Language::Haskell, "test.hs");

    let func = get_unit_by_name(&units, "quadratic").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: quadratic
Signature: quadratic a b c x = a*x^2 + b*x + c
Code:
quadratic a b c x = a*x^2 + b*x + c
  where
    square y = y * y
File: test test.hs"#;
    assert_eq!(text, expected);
}

#[test]
fn test_instance_declaration() {
    let source = r#"instance Show Person where
  show (Person name age) = name ++ " (" ++ show age ++ ")"
"#;
    let units = parse(source, Language::Haskell, "test.hs");

    assert_eq!(units.len(), 2);

    let unit0 = &units[0];
    let text0 = build_embedding_text(unit0);
    let expected0 = r#"Function: show
Signature: show (Person name age) = name ++ " (" ++ show age ++ ")"
Code:
  show (Person name age) = name ++ " (" ++ show age ++ ")"
File: test test.hs"#;
    assert_eq!(text0, expected0);

    let unit1 = &units[1];
    let text1 = build_embedding_text(unit1);
    let expected1 = r#"Code block: raw_code_1
Signature: instance Show Person where
Code:
instance Show Person where
File: test test.hs"#;
    assert_eq!(text1, expected1);
}

#[test]
fn test_type_alias() {
    let source = r#"type Name = String
type Age = Int
type Person = (Name, Age)
"#;
    let units = parse(source, Language::Haskell, "test.hs");

    assert_eq!(units.len(), 1);
    let unit = &units[0];
    let text = build_embedding_text(unit);
    let expected = r#"Code block: raw_code_1
Signature: type Name = String
Code:
type Name = String
type Age = Int
type Person = (Name, Age)
File: test test.hs"#;
    assert_eq!(text, expected);
}

#[test]
fn test_higher_order_function() {
    let source = r#"map :: (a -> b) -> [a] -> [b]
map _ [] = []
map f (x:xs) = f x : map f xs
"#;
    let units = parse(source, Language::Haskell, "test.hs");

    let func = get_unit_by_name(&units, "map").unwrap();
    let text = build_embedding_text(func);

    let expected = r#"Function: map
Signature: map _ [] = []
Code:
map _ [] = []
File: test test.hs"#;
    assert_eq!(text, expected);
}

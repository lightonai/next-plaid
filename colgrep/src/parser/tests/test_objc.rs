//! Tests for Objective-C code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const PERSON: &str = r#"#import <Foundation/Foundation.h>
#import "PersonStore.h"
@import UIKit;

static NSString * const kPersonKey = @"person";

/// Something that can greet.
@protocol Greeter <NSObject>
- (NSString *)greet:(NSString *)name;
@optional
+ (instancetype)shared;
@end

/**
 * A person with a name.
 */
@interface Person : NSObject <Greeter>
@property (nonatomic, copy) NSString *name;
- (instancetype)initWithName:(NSString *)name age:(NSInteger)age;
@end

@interface Person ()
@property (nonatomic) NSInteger age;
@end

@implementation Person

/// Designated initializer.
- (instancetype)initWithName:(NSString *)name age:(NSInteger)age {
    self = [super init];
    if (self) {
        _name = [name copy];
        [self setAge:age];
        [PersonStore registerPerson:self withKey:kPersonKey];
    }
    return self;
}

+ (instancetype)shared {
    static Person *shared;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        shared = [[Person alloc] initWithName:@"Ada" age:36];
    });
    return shared;
}

- (NSString *)greet:(NSString *)name {
    return [NSString stringWithFormat:@"Hello %@, I am %@", name, self.name];
}

@end

@implementation Person (Formatting)
- (NSString *)displayName {
    return [self.name uppercaseString];
}
@end

NSString *PersonDescription(Person *person) {
    return [person greet:@"you"];
}
"#;

#[test]
fn test_method_embedding_text() {
    let units = parse(PERSON, Language::ObjectiveC, "Person.m");

    let unit = get_unit_by_name(&units, "initWithName:age:").unwrap();
    assert_eq!(unit.unit_type, UnitType::Method);
    let text = build_embedding_text(unit);
    let expected = r#"Method: initWithName:age:
Signature: - (instancetype)initWithName:(NSString *)name age:(NSInteger)age {
Class: Person
Description: Designated initializer.
Parameters: name, age
Returns: instancetype
Calls: copy, init, registerPerson:withKey:, setAge:
File: person Person.m
Code:
/// Designated initializer.
- (instancetype)initWithName:(NSString *)name age:(NSInteger)age {
    self = [super init];
    if (self) {
        _name = [name copy];
        [self setAge:age];
        [PersonStore registerPerson:self withKey:kPersonKey];
    }
    return self;
}"#;
    assert_eq!(text, expected);
}

#[test]
fn test_interface_implementation_and_protocol() {
    let units = assert_extractor_invariants(PERSON, Language::ObjectiveC, "Person.m");

    let protocol = get_unit_by_name(&units, "Greeter").unwrap();
    assert_eq!(protocol.unit_type, UnitType::Class);
    assert_eq!(
        protocol.docstring.as_deref(),
        Some("Something that can greet.")
    );

    let interface = units
        .iter()
        .find(|u| u.name == "Person" && u.code.contains("@interface Person : NSObject"))
        .unwrap();
    assert_eq!(interface.extends.as_deref(), Some("NSObject"));
    assert_eq!(
        interface.docstring.as_deref(),
        Some("A person with a name.")
    );

    let implementation = units
        .iter()
        .find(|u| u.name == "Person" && u.code.starts_with("@implementation Person"))
        .unwrap();
    assert_eq!(implementation.unit_type, UnitType::Class);

    // Class extension and category are named the way they are written.
    assert!(get_unit_by_name(&units, "Person ()").is_some());
    let category = get_unit_by_name(&units, "Person (Formatting)").unwrap();
    assert_eq!(category.unit_type, UnitType::Class);
    let display = get_unit_by_name(&units, "displayName").unwrap();
    assert_eq!(display.parent_class.as_deref(), Some("Person (Formatting)"));
    assert_eq!(display.return_type.as_deref(), Some("NSString *"));

    // Declarations in @interface / @protocol are not separate units.
    assert_eq!(
        units.iter().filter(|u| u.name == "greet:").count(),
        1,
        "only the @implementation method"
    );
}

#[test]
fn test_class_method_and_blocks() {
    let units = parse(PERSON, Language::ObjectiveC, "Person.m");
    let shared = get_unit_by_name(&units, "shared").unwrap();
    assert_eq!(shared.unit_type, UnitType::Method);
    assert!(shared.signature.starts_with("+ (instancetype)shared"));
    assert!(shared.parameters.is_empty());
    // Message sends inside a block count, nested sends included.
    for call in ["alloc", "initWithName:age:", "dispatch_once"] {
        assert!(shared.calls.contains(&call.to_string()), "{call}");
    }
    assert!(shared.variables.contains(&"onceToken".to_string()));
}

#[test]
fn test_message_send_links_call_graph() {
    let mut units = parse(PERSON, Language::ObjectiveC, "Person.m");
    crate::parser::build_call_graph(&mut units);
    let greet = units
        .iter()
        .find(|u| u.name == "greet:" && u.unit_type == UnitType::Method)
        .unwrap();
    assert!(
        greet.called_by.contains(&"PersonDescription".to_string()),
        "{:?}",
        greet.called_by
    );
}

#[test]
fn test_c_function_constant_and_imports() {
    let units = parse(PERSON, Language::ObjectiveC, "Person.m");

    let function = get_unit_by_name(&units, "PersonDescription").unwrap();
    assert_eq!(function.unit_type, UnitType::Function);
    assert_eq!(function.parameters, vec!["person"]);
    assert_eq!(function.calls, vec!["greet:"]);

    let constant = get_unit_by_name(&units, "kPersonKey").unwrap();
    assert_eq!(constant.unit_type, UnitType::Constant);

    let raw = units
        .iter()
        .find(|u| u.unit_type == UnitType::RawCode && u.line == 1)
        .unwrap();
    assert_eq!(raw.imports, vec!["Foundation", "PersonStore", "UIKit"]);
}

/// `#if` branches inside an @interface/@implementation, lone macros and
/// NS_ENUM would otherwise throw tree-sitter-objc into error recovery and
/// lose the whole class.
#[test]
fn test_preprocessor_and_macros() {
    let source = r#"#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, LoaderState) {
    LoaderStateIdle,
    LoaderStateLoading,
};

@implementation Loader
#if TARGET_OS_IOS
- (void)start {
    [self.view setNeedsLayout];
}
#else
- (void)start {
    [self.view setNeedsDisplay:YES];
}
#endif

- (void)viewDidLoad NS_REQUIRES_SUPER {
    [super viewDidLoad];
}
@end

NS_ASSUME_NONNULL_END
"#;
    let units = assert_extractor_invariants(source, Language::ObjectiveC, "Loader.m");

    let state = get_unit_by_name(&units, "LoaderState").unwrap();
    assert_eq!(state.unit_type, UnitType::Class);
    assert_eq!((state.line, state.end_line), (5, 8));

    let loader = get_unit_by_name(&units, "Loader").unwrap();
    assert_eq!((loader.line, loader.end_line), (10, 24));

    // The first branch is the one parsed; the other stays in the class's code.
    let start = get_unit_by_name(&units, "start").unwrap();
    assert_eq!((start.line, start.end_line), (12, 14));
    assert_eq!(start.calls, vec!["setNeedsLayout"]);

    let did_load = get_unit_by_name(&units, "viewDidLoad").unwrap();
    assert_eq!(did_load.parent_class.as_deref(), Some("Loader"));
    assert_eq!(did_load.calls, vec!["viewDidLoad"]);
}

/// Objective-C++ (`.mm`) is parsed with the Objective-C grammar: the
/// Objective-C parts split as usual, and C++ the grammar tolerates
/// (templates, namespaces, `auto`) stays inside its method. Range-based `for`
/// and reference parameters are beyond it and degrade the file to raw code.
#[test]
fn test_objective_cpp_file() {
    let source = r#"#import "Renderer.h"
#include <vector>

@implementation Renderer
- (void)drawPoints:(NSArray<NSValue *> *)points {
    std::vector<CGPoint> buffer;
    for (NSValue *value in points) {
        auto p = value.CGPointValue;
        buffer.push_back(p);
        [self drawPoint:p];
    }
    AS::MutexLocker l(_lock);
}
- (void)drawPoint:(CGPoint)point {
    CGContextFillRect(_context, CGRectMake(point.x, point.y, 1, 1));
}
@end
"#;
    let units = assert_extractor_invariants(source, Language::ObjectiveC, "Renderer.mm");
    assert!(get_unit_by_name(&units, "Renderer").is_some());
    let draw = get_unit_by_name(&units, "drawPoint:").unwrap();
    assert_eq!(draw.parameters, vec!["point"]);
    assert!(draw.calls.contains(&"CGContextFillRect".to_string()));
}

/// A method header whose trailing comment ends in a multi-byte character
/// (Chinese punctuation, a curly quote, an emoji) used to be sliced inside
/// that character when looking for a trailing macro, and panicked.
#[test]
fn test_trailing_comment_with_multibyte_characters() {
    for source in [
        "- (void)setupUI // 设置界面。\n{\n}\n",
        "- (NSString *)title // the ‘title’\n{\n}\n",
        "- (void)f 😀{\n}\n",
    ] {
        assert_extractor_invariants(source, Language::ObjectiveC, "View.m");
    }
}

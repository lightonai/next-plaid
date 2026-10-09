//! Tests for Perl code extraction.

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const MODULE: &str = r#"package My::Animal;
use strict;
use warnings;
use parent -norequire, 'My::Base';
use List::Util qw(sum max);
use Carp;

our $VERSION = '1.02';
use constant LEGS => 4;

# Create a new animal.
sub new {
    my ($class, %args) = @_;
    my $self = bless { name => $args{name} }, $class;
    $self->init(%args);
    return $self;
}

=head2 speak

Makes the animal speak, once per C<$times>.

=cut

sub speak {
    my $self  = shift;
    my $times = shift;
    for my $i (1 .. $times) {
        print "...\n" if $i > 1;
    }
    Carp::croak("silent") unless $self->{name};
    return sum(1, $times);
}

1;
__END__

=head1 NAME

My::Animal - A talking animal

=head1 METHODS

=head2 legs

Returns the number of legs.
"#;

#[test]
fn test_sub_embedding_text() {
    let units = assert_extractor_invariants(MODULE, Language::Perl, "lib/My/Animal.pm");
    let new = get_unit_by_name(&units, "new").unwrap();
    let expected = r#"Method: new
Signature: sub new {
Class: My::Animal
Description: Create a new animal.
Parameters: %args
Calls: init
Variables: %args
File: lib my animal Animal.pm
Code:
# Create a new animal.
sub new {
    my ($class, %args) = @_;
    my $self = bless { name => $args{name} }, $class;
    $self->init(%args);
    return $self;
}"#;
    assert_eq!(build_embedding_text(new), expected);
}

/// A POD block naming the sub documents it; builtins (`print`, `bless`,
/// `shift`) are not calls, `Carp::croak` is a call to `croak` from `Carp`.
#[test]
fn test_pod_doc_and_calls() {
    let units = parse(MODULE, Language::Perl, "lib/My/Animal.pm");
    let speak = get_unit_by_name(&units, "speak").unwrap();
    assert_eq!(speak.unit_type, UnitType::Method);
    assert_eq!((speak.line, speak.end_line), (19, 33));
    assert_eq!(
        speak.docstring.as_deref(),
        Some("speak Makes the animal speak, once per C<$times>.")
    );
    assert_eq!(speak.parameters, vec!["$times"]);
    assert_eq!(speak.calls, vec!["croak", "sum"]);
    assert!(speak.has_loops && speak.has_branches);
    assert_eq!(speak.imports, vec!["Carp"]);
}

/// A statement-form `package` is a class unit spanning the package up to
/// `__END__`, with its parent and the abstract of its `=head1 NAME` POD.
#[test]
fn test_package_class_unit() {
    let units = parse(MODULE, Language::Perl, "lib/My/Animal.pm");
    let class = get_unit_by_name(&units, "My::Animal").unwrap();
    assert_eq!(class.unit_type, UnitType::Class);
    assert_eq!((class.line, class.end_line), (1, 35));
    assert_eq!(class.extends.as_deref(), Some("My::Base"));
    assert_eq!(
        class.docstring.as_deref(),
        Some("My::Animal - A talking animal")
    );
    assert_eq!(class.imports, vec!["Carp", "List::Util"]);
    // Pragmas are not imports.
    assert!(!class.imports.contains(&"strict".to_string()));
}

#[test]
fn test_use_constant_and_pod_sections() {
    let units = parse(MODULE, Language::Perl, "lib/My/Animal.pm");
    let legs = get_unit_by_name(&units, "LEGS").unwrap();
    assert_eq!(legs.unit_type, UnitType::Constant);
    assert_eq!(legs.line, 9);

    // The reference POD after __END__ is split by heading.
    let name = get_unit_by_name(&units, "NAME").unwrap();
    assert_eq!(name.unit_type, UnitType::Section);
    let section = get_unit_by_name(&units, "legs").unwrap();
    assert_eq!(section.unit_type, UnitType::Section);
    assert_eq!((section.line, section.end_line), (44, 46));
    assert!(section.code.contains("Returns the number of legs."));
}

#[test]
fn test_signatures_and_block_packages() {
    let source = r#"use v5.36;

package Point {
    use parent -norequire, 'Shape';

    sub new ($class, $x = 0, $y = 0) {
        return bless { x => $x, y => $y }, $class;
    }

    sub norm ($self) {
        return sqrt($self->{x}**2 + $self->{y}**2);
    }
}

class Counter 1.0 {
    field $count :param = 0;
    method inc ($by = 1) { $count += $by; return $self }
}

sub helper { Point->new(1, 2)->norm }
"#;
    let units = assert_extractor_invariants(source, Language::Perl, "point.pl");

    let point = get_unit_by_name(&units, "Point").unwrap();
    assert_eq!(point.unit_type, UnitType::Class);
    assert_eq!(point.extends.as_deref(), Some("Shape"));
    let new = get_unit_by_name(&units, "new").unwrap();
    assert_eq!(new.parent_class.as_deref(), Some("Point"));
    assert_eq!(new.parameters, vec!["$x", "$y"]);

    let inc = get_unit_by_name(&units, "inc").unwrap();
    assert_eq!(inc.unit_type, UnitType::Method);
    assert_eq!(inc.parent_class.as_deref(), Some("Counter"));
    assert_eq!(inc.parameters, vec!["$by"]);

    // Outside any package: a plain function, and `Point->new` uses `Point`.
    let helper = get_unit_by_name(&units, "helper").unwrap();
    assert_eq!(helper.unit_type, UnitType::Function);
    assert_eq!(helper.calls, vec!["new", "norm"]);
}

#[test]
fn test_multiple_packages_in_one_file() {
    let source = r#"package First;
sub one { 1 }

package Second;
our @ISA = ('First');
sub two { First::one() + 1 }
"#;
    let units = assert_extractor_invariants(source, Language::Perl, "multi.pm");
    let first = get_unit_by_name(&units, "First").unwrap();
    assert_eq!((first.line, first.end_line), (1, 2));
    let second = get_unit_by_name(&units, "Second").unwrap();
    assert_eq!((second.line, second.end_line), (4, 6));
    assert_eq!(second.extends.as_deref(), Some("First"));
    let two = get_unit_by_name(&units, "two").unwrap();
    assert_eq!(two.parent_class.as_deref(), Some("Second"));
    assert_eq!(two.calls, vec!["one"]);
}

/// Test files: each `subtest` is a unit named by its description; web-app
/// routes (Mojolicious::Lite, Dancer2) are units named by method and path.
#[test]
fn test_subtests_and_routes() {
    let source = r#"use Test::More;
use Mojolicious::Lite;

get '/hello' => sub {
    my $c = shift;
    $c->render(text => 'Hello');
};

subtest 'parses cookies' => sub {
    my $jar = Cookie::Jar->new;
    ok $jar->parse('a=b'), 'parsed';
};

done_testing();
"#;
    let units = assert_extractor_invariants(source, Language::Perl, "t/cookies.t");
    let route = get_unit_by_name(&units, "get /hello").unwrap();
    assert_eq!(route.unit_type, UnitType::Function);
    assert_eq!((route.line, route.end_line), (4, 7));
    assert_eq!(route.calls, vec!["render"]);

    let subtest = get_unit_by_name(&units, "parses cookies").unwrap();
    assert_eq!((subtest.line, subtest.end_line), (9, 12));
    assert_eq!(subtest.calls, vec!["new", "ok", "parse"]);
    assert_eq!(subtest.variables, vec!["$jar"]);
}

/// A leading `=head1 DESCRIPTION` block does not document the sub after it.
#[test]
fn test_unrelated_pod_is_not_a_doc() {
    let source = r#"package Foo;

=head1 DESCRIPTION

Foo does things.

=cut

sub bar { 1 }
"#;
    let units = parse(source, Language::Perl, "Foo.pm");
    let bar = get_unit_by_name(&units, "bar").unwrap();
    assert_eq!(bar.docstring, None);
    assert_eq!(bar.line, 9);
}

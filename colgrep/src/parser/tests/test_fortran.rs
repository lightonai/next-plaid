//! Tests for Fortran code extraction (free and fixed form).

use super::common::*;
use crate::embed::build_embedding_text;
use crate::parser::{Language, UnitType};

const GEOMETRY: &str = r#"!> Geometry utilities.
module geometry
  use iso_fortran_env, only: dp => real64
  implicit none
  private
  public :: circle_area, shape_t

  real(dp), parameter :: PI = 3.14159265358979_dp

  !> A shape with a radius.
  type, extends(base_t) :: shape_t
    real(dp) :: radius
  contains
    procedure :: area => shape_area
  end type shape_t

  interface area
    module procedure circle_area, square_area
  end interface area

contains

  !> Area of a circle of radius r.
  pure function circle_area(r) result(a)
    real(dp), intent(in) :: r
    real(dp) :: a
    a = PI * r**2
  end function circle_area

  subroutine print_areas(shapes, unit)
    use logging, only: log_message
    class(shape_t), intent(in) :: shapes(:)
    integer, intent(in) :: unit
    integer :: i
    do i = 1, size(shapes)
      call log_message(unit, shapes(i)%area())
      write(unit, *) circle_area(shapes(i)%radius)
    end do
  end subroutine print_areas

  real(dp) function shape_area(self)
    class(shape_t), intent(in) :: self
    shape_area = circle_area(self%radius)
  end function shape_area

end module geometry
"#;

#[test]
fn test_function_embedding_text() {
    let units = parse(GEOMETRY, Language::Fortran, "geometry.f90");

    let unit = get_unit_by_name(&units, "circle_area").unwrap();
    let text = build_embedding_text(unit);
    let expected = r#"Function: circle_area
Signature: pure function circle_area(r) result(a)
Class: geometry
Description: Area of a circle of radius r.
Parameters: r
Returns: real(dp)
Variables: a, r
Uses: iso_fortran_env
File: geometry geometry.f90
Code:
  pure function circle_area(r) result(a)
    real(dp), intent(in) :: r
    real(dp) :: a
    a = PI * r**2
  end function circle_area"#;
    assert_eq!(text, expected);
}

#[test]
fn test_module_and_derived_type() {
    let units = assert_extractor_invariants(GEOMETRY, Language::Fortran, "geometry.f90");

    // The module unit is its specification part, up to `contains`.
    let module = get_unit_by_name(&units, "geometry").unwrap();
    assert_eq!(module.unit_type, UnitType::Class);
    assert_eq!((module.line, module.end_line), (2, 20));
    assert_eq!(module.docstring.as_deref(), Some("Geometry utilities."));

    let shape = get_unit_by_name(&units, "shape_t").unwrap();
    assert_eq!(shape.unit_type, UnitType::Class);
    assert_eq!(shape.extends.as_deref(), Some("base_t"));
    assert_eq!((shape.line, shape.end_line), (11, 15));

    let generic = get_unit_by_name(&units, "area").unwrap();
    assert_eq!(generic.unit_type, UnitType::Class);
    assert_eq!((generic.line, generic.end_line), (17, 19));

    // Procedures end on their `end` statement, not on the next line.
    let circle = get_unit_by_name(&units, "circle_area").unwrap();
    assert_eq!((circle.line, circle.end_line), (24, 28));
    assert_eq!(circle.unit_type, UnitType::Function);
}

#[test]
fn test_subroutine_calls_and_uses() {
    let mut units = parse(GEOMETRY, Language::Fortran, "geometry.f90");
    crate::parser::build_call_graph(&mut units);

    let print = get_unit_by_name(&units, "print_areas").unwrap();
    assert_eq!(print.parameters, vec!["shapes", "unit"]);
    assert_eq!(print.return_type, None);
    assert!(print.has_loops);
    // `shapes(i)` is an element of a declared array, not a call.
    assert_eq!(print.calls, vec!["circle_area", "log_message", "size"]);
    assert_eq!(print.imports, vec!["iso_fortran_env", "logging"]);

    let shape_area = get_unit_by_name(&units, "shape_area").unwrap();
    assert_eq!(shape_area.return_type.as_deref(), Some("real(dp)"));

    let circle = get_unit_by_name(&units, "circle_area").unwrap();
    assert_eq!(circle.called_by, vec!["print_areas", "shape_area"]);
}

#[test]
fn test_program_submodule_and_interfaces() {
    let source = r#"submodule (geometry) geometry_impl
contains
  module procedure square_area
    a = s**2
  end procedure square_area
end submodule geometry_impl

module callbacks
  abstract interface
    function fn_t(x) result(y)
      real, intent(in) :: x
      real :: y
    end function fn_t
  end interface
  interface operator(+)
    module procedure add_vectors
  end interface
end module callbacks

program main
  use geometry
  implicit none
  call print_areas([shape_t(1.0)], 6)
end program main
"#;
    let units = assert_extractor_invariants(source, Language::Fortran, "main.f90");

    assert!(get_unit_by_name(&units, "geometry_impl").is_some());
    let square = get_unit_by_name(&units, "square_area").unwrap();
    assert_eq!(square.unit_type, UnitType::Function);
    assert_eq!((square.line, square.end_line), (3, 5));

    // Interface bodies declare procedures; they are not procedures.
    assert!(get_unit_by_name(&units, "abstract interface fn_t").is_some());
    assert!(get_unit_by_name(&units, "fn_t").is_none());
    assert!(get_unit_by_name(&units, "interface operator(+)").is_some());

    let program = get_unit_by_name(&units, "main").unwrap();
    assert_eq!(program.unit_type, UnitType::Class);
    assert_eq!((program.line, program.end_line), (20, 24));
    assert!(program.calls.contains(&"print_areas".to_string()));
}

/// LAPACK-style fixed form: `*` comments in column 1, `$` continuation in
/// column 6, Doxygen `*>` documentation above the routine.
const DGESV: &str = r#"*> \brief <b> DGESV computes the solution to system of linear equations A * X = B for GE matrices</b>
*
*  =========== DOCUMENTATION ===========
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> DGESV computes the solution to a real system of linear equations
*>    A * X = B.
*> \endverbatim
*
*  =====================================================================
      SUBROUTINE DGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )
      IMPLICIT NONE
*     .. Scalar Arguments ..
      INTEGER            INFO, LDA, LDB, N, NRHS
*     .. Array Arguments ..
      INTEGER            IPIV( * )
      DOUBLE PRECISION   A( LDA, * ), B( LDB, * )
*     .. External Subroutines ..
      EXTERNAL           DGETRF, DGETRS, XERBLA
      INFO = 0
      IF( N.LT.0 ) THEN
         INFO = -1
      ELSE IF( LDA.LT.MAX( 1, N ) ) THEN
         INFO = -4
      END IF
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DGESV ', -INFO )
         RETURN
      END IF
      CALL DGETRF( N, N, A, LDA, IPIV, INFO )
      IF( INFO.EQ.0 ) THEN
         CALL DGETRS( 'No transpose', N, NRHS, A, LDA, IPIV, B, LDB,
     $                INFO )
      END IF
      RETURN
*
*     End of DGESV
*
      END
      DOUBLE PRECISION FUNCTION DNORM( X )
      DOUBLE PRECISION X
      DNORM = ABS( X )
      END
"#;

#[test]
fn test_fixed_form_lapack_routine() {
    let units = assert_extractor_invariants(DGESV, Language::Fortran, "dgesv.f");

    let dgesv = get_unit_by_name(&units, "DGESV").unwrap();
    assert_eq!(dgesv.unit_type, UnitType::Function);
    assert_eq!((dgesv.line, dgesv.end_line), (15, 43));
    assert_eq!(
        dgesv.docstring.as_deref(),
        Some(
            "DGESV computes the solution to system of linear equations A * X = B for GE matrices \
             Purpose: DGESV computes the solution to a real system of linear equations A * X = B."
        )
    );
    assert_eq!(
        dgesv.parameters,
        vec!["N", "NRHS", "A", "LDA", "IPIV", "B", "LDB", "INFO"]
    );
    // MAX( 1, N ) is an intrinsic call; A( LDA, * ) is an array.
    assert_eq!(dgesv.calls, vec!["DGETRF", "DGETRS", "MAX", "XERBLA"]);
    assert!(dgesv.has_branches);

    let dnorm = get_unit_by_name(&units, "DNORM").unwrap();
    assert_eq!(dnorm.return_type.as_deref(), Some("DOUBLE PRECISION"));
    assert_eq!((dnorm.line, dnorm.end_line), (44, 47));
}

/// The same code parses the same whether it is written fixed or free form.
#[test]
fn test_fixed_form_matches_free_form() {
    let fixed = "      SUBROUTINE AXPY( N, A, X, Y )\n      INTEGER N\n      REAL A, X( * ), Y( * )\n      CALL SAXPY( N, A, X, 1,\n     $            Y, 1 )\n      END\n";
    let free = "      SUBROUTINE AXPY( N, A, X, Y )\n      INTEGER N\n      REAL A, X( * ), Y( * )\n      CALL SAXPY( N, A, X, 1, &\n     &            Y, 1 )\n      END\n";
    let from_fixed = parse(fixed, Language::Fortran, "axpy.f");
    let from_free = parse(free, Language::Fortran, "axpy.f90");
    let (a, b) = (
        get_unit_by_name(&from_fixed, "AXPY").unwrap(),
        get_unit_by_name(&from_free, "AXPY").unwrap(),
    );
    assert_eq!((a.line, a.end_line), (b.line, b.end_line));
    assert_eq!(a.calls, b.calls);
    assert_eq!(a.parameters, b.parameters);
    assert_eq!(a.calls, vec!["SAXPY"]);
}

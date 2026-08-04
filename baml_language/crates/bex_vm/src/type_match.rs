//! Runtime `IsType` matching against a [`TyTemplate`], using the canonical type
//! algebra (`baml_type::normalize`) over the running program — the [`BexVm`]
//! itself is the [`baml_type::normalize::TypeContext`].
//!
//! A `match`/`is` pattern template is *complete*: `TyTemplate` has no match-any
//! holes, so given the enclosing frame's realized `type_args` it denotes
//! exactly one type. [`value_matches_template`] therefore substitutes and asks
//! the canonical algebra one covariant membership question (`is_subtype`): a
//! value belongs to `int | string` iff its concrete type is a subtype, and
//! generic argument positions are related invariantly *by the algebra itself*
//! (`int[]` is not `string[]`, `Foo<int>` is not `Foo<string>`).
//!
//! The `ClassWithTypeArgs` fast path (class-pointer identity fixes the class,
//! then each type-arg position is checked separately) uses the same machinery
//! per argument via [`class_type_arg_matches`] — invariantly, since BAML
//! generics are invariant.

use baml_type::normalize;
use bex_vm_types::{RealizedTy, TyTemplate, Value, errors::VmInternalError};

use crate::BexVm;

/// Whether `value` is a member of the type denoted by `template`, with the
/// enclosing frame's realized `type_args` resolving the template's frame
/// references. The `IsType` value matcher for `match` and `is` expressions.
///
/// A substitution failure is a broken compiler/VM invariant, surfaced as an
/// internal error rather than silently mis-answering the test: a frame
/// reference the seeded frame does not supply is a frame-layout bug, and a
/// pattern template's projection reducing opaquely breaks the registry's
/// completeness guarantee (every baked impl rule carries a binding for every
/// declared associated member — pinned or baked from its declared default).
pub(crate) fn value_matches_template(
    vm: &BexVm,
    value: Value,
    template: &TyTemplate,
    frame_type_args: &[RealizedTy],
) -> Result<bool, VmInternalError> {
    // A value with no reconstructible concrete BAML type (an opaque native
    // handle, or a compile-time definition object — see `value_concrete_ty`) is
    // a member of no structural type test. Every *data* value reconstructs
    // faithfully, including the callables: a closure, generic function, or
    // bound method materializes its stored signature templates against the
    // realized frame the value carries (a bound method's drops the applied
    // receiver), so one minted in a generic frame is as precise as any other;
    // futures reconstruct at the `Future<T, E>` their spawn site was typed at.
    let Some(value_ty) = vm.value_concrete_ty(value) else {
        return Ok(false);
    };
    // The canonical algebra operates over `Ty`; a value's concrete type widens
    // into it (a shallow structural conversion — `ConcreteRealizedTy` is not a
    // deep, transmute-compatible family member with a borrowed upcast).
    let value_ty: baml_type::Ty<bex_vm_types::TypeHead> = value_ty.into();
    // Resolve frame references into a realized type, then let the canonical
    // algebra do the work.
    let expected = template.substitute(frame_type_args, vm).map_err(|e| {
        VmInternalError::TypeSubstitution {
            message: e.to_string(),
        }
    })?;
    Ok(normalize::is_subtype(&value_ty, expected.as_ty(), vm))
}

/// Whether `actual` is *invariantly* the type denoted by `template` (resolved
/// against `frame_type_args`) — one class type-arg position of the
/// `ClassWithTypeArgs` check. BAML generics are invariant, so the relation is
/// canonical equivalence, not membership. A substitution failure is a broken
/// invariant, exactly as in [`value_matches_template`].
pub(crate) fn class_type_arg_matches<C: normalize::TypeContext<bex_vm_types::TypeHead>>(
    ctx: &C,
    template: &TyTemplate,
    frame_type_args: &[RealizedTy],
    actual: &baml_type::Ty<bex_vm_types::TypeHead>,
) -> Result<bool, VmInternalError> {
    let expected = template.substitute(frame_type_args, ctx).map_err(|e| {
        VmInternalError::TypeSubstitution {
            message: e.to_string(),
        }
    })?;
    Ok(normalize::equivalent(actual, expected.as_ty(), ctx))
}

#[cfg(test)]
mod tests {
    // The matcher runs at the runtime head, so the tests build their operands
    // there too; `TypeHead::of_name` gives a comparable head with no heap.
    use baml_type::{Interface, Name, ParamTy, QualifiedTypeName, Ty, TypeName, normalize};
    use bex_vm_types::{RealizedTy, RuntimeTy, TyTemplate, TypeHead};

    use super::class_type_arg_matches;

    /// A fail-safe, context-free [`TypeContext`]: no aliases, no interface
    /// memberships, no bounds. It exercises the matcher's *structural* algebra
    /// (invariant generic args, literal widening, union membership) — the parts
    /// that don't need program facts. Nominal facts (a class implementing an
    /// interface) are validated by the VM-backed e2e tests.
    struct EmptyCtx;
    impl normalize::TypeContext<TypeHead> for EmptyCtx {
        /// Heads are content-addressed from names, so even a fact-free context
        /// can answer this — and must, or the `AnyFunction` covariance rule
        /// silently stops firing.
        fn head_lookup(&self, qtn: &QualifiedTypeName) -> Option<TypeHead> {
            Some(TypeHead::of_name(qtn))
        }

        fn alias_def(&self, _name: &TypeHead) -> Option<Ty<TypeHead>> {
            None
        }

        fn implements_interface(
            &self,
            _concrete: &Ty<TypeHead>,
            _interface: &Interface<TypeHead>,
        ) -> bool {
            false
        }

        fn type_var_bound(&self, _param: &ParamTy) -> Vec<Interface<TypeHead>> {
            Vec::new()
        }

        fn interface_requires(
            &self,
            _sub: &Interface<TypeHead>,
            _sup: &Interface<TypeHead>,
        ) -> bool {
            false
        }

        fn enum_variants(&self, _name: &TypeHead) -> Option<Vec<Name>> {
            None
        }

        fn associated_type_bound(
            &self,
            _interface: &Interface<TypeHead>,
            _assoc: Name,
        ) -> Vec<Interface<TypeHead>> {
            Vec::new()
        }

        fn project(
            &self,
            _base: &Ty<TypeHead>,
            _interface: &Interface<TypeHead>,
            _member: &Name,
            _fuel: u32,
        ) -> normalize::ProjectionStep<TypeHead> {
            normalize::ProjectionStep::Opaque
        }
    }

    /// The complete-template value relation over the empty context: substitute
    /// against the frame, then covariant membership — mirroring
    /// `value_matches_template` past the VM-specific concrete-type read. The
    /// frame args are written as `RuntimeTy` for test ergonomics and narrowed
    /// to the realized frame the matcher takes (they are all concrete here).
    fn matches(template: &TyTemplate, frame: &[RuntimeTy], actual: &RuntimeTy) -> bool {
        let frame: Vec<RealizedTy> = frame
            .iter()
            .map(|t| RealizedTy::try_from(t).expect("test frame arg is realized"))
            .collect();
        let Ok(expected) = template.substitute(&frame, &EmptyCtx) else {
            return false;
        };
        baml_type::normalize::is_subtype(actual.as_ty(), expected.as_ty(), &EmptyCtx)
    }

    /// The invariant class-arg relation over the empty context
    /// (`ClassWithTypeArgs`'s per-arg check).
    fn arg_matches(template: &TyTemplate, frame: &[RuntimeTy], actual: &RuntimeTy) -> bool {
        let frame: Vec<RealizedTy> = frame
            .iter()
            .map(|t| RealizedTy::try_from(t).expect("test frame arg is realized"))
            .collect();
        class_type_arg_matches(&EmptyCtx, template, &frame, actual.as_ty())
            .expect("test templates reference only supplied frame slots")
    }

    /// A realized-leaf template from a `RealizedTy`.
    fn leaf(ty: RealizedTy) -> TyTemplate {
        TyTemplate::from(ty)
    }

    /// A head for a user class. Unresolved — these tests compare, and identity
    /// is the tag.
    fn user_class(name: &str) -> bex_vm_types::TypeHead {
        bex_vm_types::TypeHead::of_name(&TypeName::local(Name::new(name)))
    }

    #[test]
    fn list_discriminates_element_type() {
        let int_list_pat = TyTemplate::list(leaf(RealizedTy::int()));
        assert!(matches(
            &int_list_pat,
            &[],
            &RuntimeTy::list(RuntimeTy::int())
        ));
        // `int[]` must NOT match a `string[]` value — element position is invariant.
        assert!(!matches(
            &int_list_pat,
            &[],
            &RuntimeTy::list(RuntimeTy::string())
        ));
        // Nor a bare `int`.
        assert!(!matches(&int_list_pat, &[], &RuntimeTy::int()));
    }

    #[test]
    fn map_discriminates_value_type() {
        let m_int = TyTemplate::map(leaf(RealizedTy::string()), leaf(RealizedTy::int()));
        assert!(matches(
            &m_int,
            &[],
            &RuntimeTy::map(RuntimeTy::string(), RuntimeTy::int())
        ));
        assert!(!matches(
            &m_int,
            &[],
            &RuntimeTy::map(RuntimeTy::string(), RuntimeTy::string())
        ));
    }

    #[test]
    fn class_type_args_are_invariant() {
        let tn = user_class("Foo");
        let foo_int = TyTemplate::class(tn, vec![leaf(RealizedTy::int())]);
        assert!(matches(
            &foo_int,
            &[],
            &RuntimeTy::Class(tn, vec![RuntimeTy::int()], baml_type::TyAttr::default())
        ));
        assert!(!matches(
            &foo_int,
            &[],
            &RuntimeTy::Class(tn, vec![RuntimeTy::string()], baml_type::TyAttr::default())
        ));
    }

    #[test]
    fn type_arg_ref_realizes_frame_slot() {
        // `T` arm with the frame binding `T = int`: an `int` value matches, a
        // `string` value does not.
        let t = TyTemplate::TypeArgRef(0);
        assert!(matches(&t, &[RuntimeTy::int()], &RuntimeTy::int()));
        assert!(!matches(&t, &[RuntimeTy::int()], &RuntimeTy::string()));
    }

    #[test]
    fn type_arg_ref_membership_is_covariant_at_top_level() {
        // `T = int | string`: an `int` value is a *member* of the union arm.
        let t = TyTemplate::TypeArgRef(0);
        let frame = [RuntimeTy::union([RuntimeTy::int(), RuntimeTy::string()])];
        assert!(matches(&t, &frame, &RuntimeTy::int()));
        assert!(matches(&t, &frame, &RuntimeTy::string()));
        assert!(!matches(&t, &frame, &RuntimeTy::bool()));
    }

    #[test]
    fn list_of_type_arg_ref() {
        // `T[]` with `T = int`.
        let t_list = TyTemplate::list(TyTemplate::TypeArgRef(0));
        assert!(matches(
            &t_list,
            &[RuntimeTy::int()],
            &RuntimeTy::list(RuntimeTy::int())
        ));
        assert!(!matches(
            &t_list,
            &[RuntimeTy::int()],
            &RuntimeTy::list(RuntimeTy::string())
        ));
    }

    #[test]
    fn union_top_level_membership() {
        // A bare `int | string` arm: an `int` value is a member; a `bool` is not.
        let u = TyTemplate::union([leaf(RealizedTy::int()), leaf(RealizedTy::string())]);
        assert!(matches(&u, &[], &RuntimeTy::int()));
        assert!(matches(&u, &[], &RuntimeTy::string()));
        assert!(!matches(&u, &[], &RuntimeTy::bool()));
    }

    #[test]
    fn literal_widens_into_base() {
        // A value of literal type `1` is a member of the `int` arm.
        let int_arm = leaf(RealizedTy::int());
        let one = RuntimeTy::Literal(
            baml_type::Literal::Int(1),
            baml_type::Freshness::Regular,
            baml_type::TyAttr::default(),
        );
        assert!(matches(&int_arm, &[], &one));
    }

    // ── Class-arg (invariant) checks — the `ClassWithTypeArgs` per-arg relation ──

    #[test]
    fn class_arg_is_invariant_not_covariant() {
        // A concrete arg is equivalence, not membership: `int` relates to
        // `int`, but NOT to a wider union it is merely a member of.
        let int_arg = leaf(RealizedTy::int());
        assert!(arg_matches(&int_arg, &[], &RuntimeTy::int()));
        assert!(!arg_matches(&int_arg, &[], &RuntimeTy::string()));
        let union_arg = TyTemplate::union([leaf(RealizedTy::int()), leaf(RealizedTy::string())]);
        assert!(!arg_matches(&union_arg, &[], &RuntimeTy::int()));
    }

    #[test]
    fn class_arg_frame_ref_resolves_and_relates_invariantly() {
        // `Foo<#0>`'s arg with `#0 = int`: the instance's stored arg must be
        // equivalent to the frame's binding.
        let t = TyTemplate::TypeArgRef(0);
        assert!(arg_matches(&t, &[RuntimeTy::int()], &RuntimeTy::int()));
        assert!(!arg_matches(&t, &[RuntimeTy::int()], &RuntimeTy::string()));
        // A widened frame binding still relates by *equivalence*: the algebra
        // absorbs subsumed members, so `int | int` ≡ `int`, but `int | string`
        // is not equivalent to `int`.
        let frame = [RuntimeTy::union([RuntimeTy::int(), RuntimeTy::string()])];
        assert!(!arg_matches(&t, &frame, &RuntimeTy::int()));
    }
}

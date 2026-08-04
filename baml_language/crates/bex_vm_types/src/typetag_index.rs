//! Tag → declaration lookup over the compile-time region.
//!
//! The inverse of the content-addressing that lets emit mint heads without a
//! heap: once the pool has addresses, this maps each declaration's
//! [`TypeTag`] to the [`HeapPtr`] a [`TypeHead`](crate::TypeHead) should point
//! at. Used once, by the loader, to fill in every head's access path.

use std::collections::HashMap;

use baml_type::typetag::TypeTag;

use crate::{HeapPtr, Object};

/// Every declared head in a compile-time pool, by tag.
pub struct TagIndex {
    by_tag: HashMap<TypeTag, HeapPtr>,
}

impl TagIndex {
    /// Index every declaration object, keyed by the `type_tag` it carries.
    ///
    /// Only the four declaration kinds mint a head, so only they appear here. A
    /// duplicate tag would mean two declarations share an identity, which emit's
    /// collision check already rejects — so the later entry simply wins rather
    /// than this pass inventing a policy for a state that cannot arise.
    pub fn of_compile_time<H: CompileTimePool>(pool: &H) -> Self {
        let mut by_tag = HashMap::new();
        for slot in 0..pool.compile_time_len() {
            let Some(tag) = pool.compile_time_object(slot).declared_type_tag() else {
                continue;
            };
            by_tag.insert(tag, pool.compile_time_ptr(slot));
        }
        Self { by_tag }
    }

    /// The declaration `tag` names, if this pool carries it.
    #[must_use]
    pub fn get(&self, tag: TypeTag) -> Option<HeapPtr> {
        self.by_tag.get(&tag).copied()
    }

    /// The underlying map, for callers that keep it past load (the runtime needs
    /// it to resolve heads minted from names).
    #[must_use]
    pub fn into_map(self) -> HashMap<TypeTag, HeapPtr> {
        self.by_tag
    }
}

/// What [`TagIndex`] needs of a compile-time pool. Implemented by the heap;
/// stated as a trait so this crate need not depend on `bex_heap`.
pub trait CompileTimePool {
    /// How many objects the region holds.
    fn compile_time_len(&self) -> usize;
    /// The object at `slot`.
    fn compile_time_object(&self, slot: usize) -> &Object;
    /// A pointer to the object at `slot`.
    fn compile_time_ptr(&self, slot: usize) -> HeapPtr;
}

/// Call `f` on every [`TypeHead`](crate::TypeHead) reachable from `object`.
///
/// The walk *within* each type is generated (`visit_heads_mut`), so a new
/// head-bearing position in the type family is covered automatically. This list
/// of object fields is not — adding a type-carrying field to an `Object` variant
/// means adding it here, which is what the loader's closing assertion guards.
///
/// The match is exhaustive on purpose: a new `Object` variant must be classified
/// rather than silently skipped.
pub fn visit_object_heads_mut(object: &mut Object, f: &mut impl FnMut(&mut crate::TypeHead)) {
    match object {
        Object::Class(class) => {
            for field in &mut class.fields {
                field.field_type.visit_heads_mut(f);
                field.field_template.visit_heads_mut(f);
            }
        }
        Object::Interface(iface) => {
            for (_, bounds) in &mut iface.args {
                for bound in bounds {
                    bound.visit_heads_mut(f);
                }
            }
            for req in &mut iface.requires {
                req.visit_heads_mut(f);
            }
            for (_, assoc) in &mut iface.assoc {
                assoc.visit_heads_mut(f);
            }
            for field in &mut iface.fields {
                field.ty.visit_heads_mut(f);
            }
            for method in &mut iface.methods {
                for arg in &mut method.args {
                    arg.visit_heads_mut(f);
                }
                for (_, kwarg) in &mut method.kwargs {
                    kwarg.visit_heads_mut(f);
                }
                method.returns.visit_heads_mut(f);
                method.errors.visit_heads_mut(f);
            }
        }
        Object::ImplRule(rule) => {
            rule.for_ty_pattern.visit_heads_mut(f);
            for bounds in &mut rule.generic_param_bounds {
                for bound in bounds {
                    f(&mut bound.interface);
                    for arg in &mut bound.args {
                        arg.visit_heads_mut(f);
                    }
                    for (_, assoc) in &mut bound.assoc {
                        assoc.visit_heads_mut(f);
                    }
                }
            }
            for arg in &mut rule.interface_args {
                arg.visit_heads_mut(f);
            }
            for (_, assoc) in &mut rule.interface_assoc {
                assoc.visit_heads_mut(f);
            }
            for method in rule.methods.values_mut() {
                for frame in &mut method.frame {
                    frame.visit_heads_mut(f);
                }
            }
        }
        Object::Function(func) => {
            func.return_type.visit_heads_mut(f);
            func.throws_type.visit_heads_mut(f);
            for param in &mut func.param_types {
                param.visit_heads_mut(f);
            }
            for constant in &mut func.bytecode.constants {
                if let crate::ConstValue::Type(template) = constant {
                    template.visit_heads_mut(f);
                }
                if let crate::ConstValue::ClassWithTypeArgs {
                    type_args_templates,
                    ..
                } = constant
                {
                    for template in type_args_templates {
                        template.visit_heads_mut(f);
                    }
                }
            }
        }
        Object::TypeAlias(alias) => alias.definition.visit_heads_mut(f),
        Object::Instance(instance) => {
            for arg in &mut instance.class_type_args {
                arg.visit_heads_mut(f);
            }
        }
        Object::Closure(closure) => {
            for arg in &mut closure.captured_type_args {
                arg.visit_heads_mut(f);
            }
        }
        Object::BoundMethod(bm) => {
            for arg in &mut bm.type_args {
                arg.visit_heads_mut(f);
            }
        }
        Object::GenericFunction(gf) => {
            for arg in &mut gf.type_args {
                arg.visit_heads_mut(f);
            }
        }
        Object::HostClosure(hc) => {
            hc.ret_ty.visit_heads_mut(f);
            hc.throws_ty.visit_heads_mut(f);
            for param in hc.params.iter_mut() {
                param.ty.visit_heads_mut(f);
            }
        }
        Object::UnscheduledFuture(fut) => {
            fut.returns.visit_heads_mut(f);
            fut.throws.visit_heads_mut(f);
        }
        Object::Array(array) => array.element_ty.visit_heads_mut(f),
        Object::Map(map) => {
            map.key_ty.visit_heads_mut(f);
            map.value_ty.visit_heads_mut(f);
        }
        Object::Type(ty) => ty.visit_heads_mut(f),
        // Carry no type: `Package`/`Enum` reference declarations by pointer or
        // hold only names and variant metadata; the rest are plain values.
        Object::Package(_)
        | Object::Enum(_)
        | Object::Variant(_)
        | Object::Cell(_)
        | Object::String(_)
        | Object::Bigint(_)
        | Object::Uint8Array(_)
        | Object::RustData(_)
        | Object::Collector(_)
        | Object::Float(_)
        | Object::Future(_) => {}
        #[cfg(feature = "heap_debug")]
        Object::Sentinel(_) => {}
    }
}

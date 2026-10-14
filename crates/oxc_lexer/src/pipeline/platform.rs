//! Types representing CPU feature sets, used to choose between SIMD and generic implementations.
//!
//! - `Avx2`: x86_64 with AVX2, BMI2 and POPCNT.
//! - `Neon`: aarch64 with Neon.
//! - [`Generic`]: No SIMD. Exists on all platforms.
//!
//! [`Platform`] is an enum of these types. [`Platform::detect`] returns the best one available.
//!
//! Code which has SIMD implementations takes a type implementing [`CpuFeatures`],
//! and uses [`platform_select!`] to pick an implementation.
//!
//! * Given a specific type (e.g. `Avx2`): the choice is made at compile time.
//! * Given a [`Platform`]: the choice is made at runtime.
//!
//! Functions which use [`platform_select!`] typically take a generic `impl CpuFeatures`.
//! This compiles a separate version of the function for each [`CpuFeatures`] type that callers use with it.
//!
//! [`Platform`] also implements [`CpuFeatures`], so functions that receive an `impl CpuFeatures`
//! can do either a compile-time or runtime features check, depending on whether it's called with
//! a concrete type (e.g. `Avx2`) or [`Platform`].
//!
//! # SAFETY
//!
//! These types are unforgeable.
//!
//! * Code outside this file can't create them.
//! * Each has a private field, and `Platform::detect` is the only code which creates them.
//!   It only creates an `Avx2` or a `Neon` if that feature set is available.
//! * A [`Platform`] can only be built from one of these types, so it's unforgeable too.
//! * [`CpuFeatures`] trait is sealed, so no other type can implement it.
//!
//! Therefore, holding one of these types guarantees that that the CPU supports that feature set.
//!
//! The rest of the lexer relies on this for safety. A function which takes `Avx2` or `Neon`
//! can use instructions that rely on support for those features without any checks.
//!
//! DO NOT add any other way to create these types (e.g. a `Default` impl or a public constructor).

/// Trait for getting the [`Platform`] for a specific CPU architecture.
///
/// Implemented by [`Platform`] itself, and all types representing specific CPU feature sets.
#[expect(private_bounds)]
pub(super) trait CpuFeatures: Sealed + Copy {
    fn platform(self) -> Platform;
}

/// Private trait to guarantee that only types defined in this file can implement [`CpuFeatures`].
trait Sealed {}

/// Enum representing CPU feature sets for a platform.
///
/// Used for runtime feature detection.
///
/// Implements [`CpuFeatures`] and also provides a [`detect`] method which initializes [`Platform`]
/// via target feature detection.
///
/// [`detect`]: Self::detect
#[derive(Clone, Copy, Debug)]
pub(super) enum Platform {
    /// AVX2 feature set. Only available on x86_64.
    #[cfg(target_arch = "x86_64")]
    Avx2(Avx2),
    /// Neon feature set. Only available on aarch64.
    #[cfg(target_arch = "aarch64")]
    Neon(Neon),
    /// Exists on all platforms, but can never be created on aarch64
    /// when Neon is enabled at compile time. See [`Generic`].
    Generic(Generic),
}

impl CpuFeatures for Platform {
    /// Get [`Platform`].
    ///
    /// Just returns itself.
    #[inline(always)]
    fn platform(self) -> Platform {
        self
    }
}

impl Sealed for Platform {}

/// Generic implementation.
mod generic {
    use super::*;

    /// Generic CPU feature set.
    ///
    /// On aarch64 with Neon enabled at compile time, this type is uninhabited - `Generic` can never be created.
    /// So [`Platform::Generic`] can never exist there, and the compiler removes code which handles it.
    /// Every aarch64 target with `std` enables Neon by default, so in practice this applies to all aarch64 builds.
    ///
    /// The type still exists there, so code which uses it is still compiled and type-checked.
    /// That means generic code is still linted, and is visible to rust-analyzer, when working on an ARM Mac.
    #[derive(Clone, Copy, Debug)]
    pub struct Generic(GenericInner);

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    type GenericInner = ();

    // When Neon is enabled at compile time, Neon is always available, so the generic implementation
    // is never needed. `Infallible` has no values, which makes `Generic` impossible to create.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    type GenericInner = std::convert::Infallible;

    impl CpuFeatures for Generic {
        /// Get [`Platform`] representing generic CPU feature set.
        #[inline(always)]
        fn platform(self) -> Platform {
            Platform::Generic(self)
        }
    }

    impl Sealed for Generic {}

    impl Platform {
        /// Create [`Platform::Generic`].
        ///
        /// Private within this file. Used only in [`detect`] implementations.
        /// Not available on aarch64 with Neon enabled at compile time, where a `Generic` can't be created.
        ///
        /// [`detect`]: Self::detect
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        pub(super) fn generic() -> Self {
            Generic(()).platform()
        }
    }
}

pub(super) use generic::Generic;

/// x86_64 features.
/// AVX2 or generic.
#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use super::*;

    /// x64_64 AVX2 CPU feature set.
    #[derive(Clone, Copy, Debug)]
    pub struct Avx2(());

    impl CpuFeatures for Avx2 {
        /// Get [`Platform`] representing AVX2 CPU feature set.
        #[inline(always)]
        fn platform(self) -> Platform {
            Platform::Avx2(self)
        }
    }

    impl Sealed for Avx2 {}

    impl Platform {
        /// Detect CPU feature set for x86_64 platform.
        pub fn detect() -> Self {
            // Currently a compile-time check. Later on will be replaced by runtime detection.
            if cfg!(all(
                target_feature = "avx2",
                target_feature = "bmi2",
                target_feature = "popcnt"
            )) {
                Avx2(()).platform()
            } else {
                Platform::generic()
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) use x86_64::Avx2;

/// aarch64 features.
/// Neon or generic.
#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use super::*;

    /// aarch64 Neon CPU feature set.
    #[derive(Clone, Copy, Debug)]
    pub struct Neon(());

    impl CpuFeatures for Neon {
        /// Get [`Platform`] representing Neon CPU feature set.
        #[inline(always)]
        fn platform(self) -> Platform {
            Platform::Neon(self)
        }
    }

    impl Sealed for Neon {}

    impl Platform {
        /// Detect CPU feature set for aarch64 platform.
        pub fn detect() -> Self {
            // Neon is always enabled except on `aarch64*-unknown-none-softfloat` platforms
            // or when it's specifically disabled by compiling with `-C target-feature=-neon`.
            // Either way, it can be detected at compile time. No runtime feature check.
            cfg_select! {
                target_feature = "neon" => Neon(()).platform(),
                _ => Platform::generic(),
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(super) use aarch64::Neon;

/// Fallback implementation for other platforms.
///
/// Only generic feature set is available.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
impl Platform {
    /// Detect CPU feature set for other platform.
    ///
    /// Only generic feature set is available.
    pub fn detect() -> Self {
        Platform::generic()
    }
}

/// Macro for either static or dynamic feature selection.
///
/// Arms use normal `match` syntax, but there must be exactly these 3 arms, in this order:
///
/// - `Avx2(...)`: Only compiled on x86_64.
/// - `Neon(...)`: Only compiled on aarch64.
/// - `Generic(...)`: Compiled everywhere.
///   On aarch64 with Neon enabled at compile time, it's compiled and type-checked,
///   but can never run, so the compiler removes it. See [`Generic`].
///
/// Each arm binds that platform's feature set type (e.g. `Avx2`), so it can be passed on.
/// Fallback arms (e.g. `_ => ...`) and match guards are not allowed.
///
/// Requiring all 3 arms means every call site says what to do on every platform.
/// It also makes the `match` exhaustive on every platform, without a fallback arm.
///
/// # Examples
///
/// ## Dynamic selection
///
/// ```ignore
/// use crate::pipeline::platform::{Platform, platform_select};
///
/// fn classify(platform: Platform) {
///     // Macro compiles down to a runtime match
///     platform_select!(platform, {
///         Avx2(feat) => classify_avx2(self, feat),
///         Neon(feat) => classify_neon(self, feat),
///         Generic(_) => classify_generic(self),
///     });
/// }
/// ```
///
/// ## Static selection
///
/// ```ignore
/// use crate::pipeline::platform::{CpuFeatures, Platform, platform_select};
///
/// fn classify(&mut self, feat: impl CpuFeatures) {
///     // Macro compiles down to a branchless select at runtime
///     platform_select!(feat, {
///         Avx2(feat) => classify_avx2(self, feat),
///         Neon(feat) => classify_neon(self, feat),
///         Generic(_) => classify_generic(self),
///     });
/// }
/// ```
///
/// ## Expansion
///
/// Macro expands to:
///
/// ```ignore
/// {
///     use crate::pipeline::platform::{CpuFeatures, Platform};
///
///     let platform_or_features = platform;
///     let _ = CpuFeatures::platform(platform_or_features);
///
///     match platform_or_features.platform() {
///         #[cfg(target_arch = "x86_64")]
///         Platform::Avx2(feat) => classify_avx2(self, feat),
///
///         #[cfg(target_arch = "aarch64")]
///         Platform::Neon(feat) => classify_neon(self, feat),
///
///         Platform::Generic(_) => classify_generic(self),
///     }
/// }
/// ```
macro_rules! platform_select {
    // Tokens of the arms are consumed one at a time, and copied into `[...]`.
    // `Avx2(...) =>` is rewritten to `Platform::Avx2(...) =>`, with a `#[cfg]` attribute for x86_64.
    // `Neon(...) =>` is rewritten the same way, with a `#[cfg]` attribute for aarch64.
    // `Generic(...) =>` is rewritten to `Platform::Generic(...) =>`, with no `#[cfg]` attribute.
    // All other tokens are copied unchanged, so the arm bodies are parsed by the `match` itself.
    //
    // `[no no no]` records whether the `Avx2`, `Neon` and `Generic` arms have been seen yet.
    // Each arm's rule only matches if the arms before it have been seen, and it hasn't.
    // So the arms must come in that order, once each. Anything else is a compile error.

    // Entry point
    ($platform_or_features:expr, { $($arms:tt)* }) => {
        crate::pipeline::platform::platform_select!(@munch $platform_or_features; [no no no]; []; $($arms)*)
    };

    // `Avx2` arm. Must be first.
    (
        @munch $platform_or_features:expr; [no no no]; [$($done:tt)*];
        Avx2($($pat:tt)*) => $($rest:tt)*
    ) => {
        crate::pipeline::platform::platform_select!(
            @munch $platform_or_features;
            [yes no no];
            [
                $($done)*
                #[cfg(target_arch = "x86_64")]
                Platform::Avx2($($pat)*) =>
            ];
            $($rest)*
        )
    };

    // `Neon` arm. Must follow `Avx2` arm.
    (
        @munch $platform_or_features:expr; [yes no no]; [$($done:tt)*];
        Neon($($pat:tt)*) => $($rest:tt)*
    ) => {
        crate::pipeline::platform::platform_select!(
            @munch $platform_or_features;
            [yes yes no];
            [
                $($done)*
                #[cfg(target_arch = "aarch64")]
                Platform::Neon($($pat)*) =>
            ];
            $($rest)*
        )
    };

    // `Generic` arm. Must follow `Neon` arm.
    (
        @munch $platform_or_features:expr; [yes yes no]; [$($done:tt)*];
        Generic($($pat:tt)*) => $($rest:tt)*
    ) => {
        crate::pipeline::platform::platform_select!(
            @munch $platform_or_features;
            [yes yes yes];
            [
                $($done)*
                Platform::Generic($($pat)*) =>
            ];
            $($rest)*
        )
    };

    // Any other `Name(...) =>` arm - out of order, repeated, or not one of the 3 names
    (@munch $platform_or_features:expr; $seen:tt; $done:tt; $name:ident($($pat:tt)*) => $($rest:tt)*) => {
        crate::pipeline::platform::platform_select!(@error)
    };

    // Any other `=>` - an arm whose pattern isn't `Name(...)` (e.g. `_ =>`), or which has a match guard
    (@munch $platform_or_features:expr; $seen:tt; $done:tt; => $($rest:tt)*) => {
        crate::pipeline::platform::platform_select!(@error)
    };

    // Any other token.
    //
    // Copy 1 token, and go round again.
    //
    // The arm rules only consume `Avx2(...) =>` etc, not the arm's body.
    // So the remaining tokens can't all be copied in one go. They start with the body,
    // and a `Neon(...) =>` which still needs rewriting may come after it.
    //
    // `macro_rules!` can't match "all tokens up to the next `Neon`",
    // because `$($x:tt)*` would also match `Neon`.
    // The only way to find the next arm is to step through the tokens.
    //
    // Consuming the body in the arm rules would avoid this, but would need 2 rules for each arm
    // (block body, or expression followed by `,`), and would reject comma-free bodies like `unsafe { ... }`.
    //
    // A delimited group (e.g. a block body) is a single token, so its contents are never inspected.
    // This rule must come after the arm rules and the error rules above,
    // because `$next:tt` also matches `Avx2` and `=>`.
    (@munch $platform_or_features:expr; $seen:tt; [$($done:tt)*]; $next:tt $($rest:tt)*) => {
        crate::pipeline::platform::platform_select!(
            @munch $platform_or_features;
            $seen;
            [$($done)* $next];
            $($rest)*
        )
    };

    // All tokens consumed, and all 3 arms seen
    (@munch $platform_or_features:expr; [yes yes yes]; [$($arms:tt)*];) => {{
        use crate::pipeline::platform::{CpuFeatures, Platform};

        let platform_or_features = $platform_or_features;
        // Check that `platform_or_features` implements `CpuFeatures`
        let _ = CpuFeatures::platform(platform_or_features);

        match platform_or_features.platform() {
            $($arms)*
        }
    }};

    // All tokens consumed, but an arm is missing
    (@munch $platform_or_features:expr; $seen:tt; $done:tt;) => {
        crate::pipeline::platform::platform_select!(@error)
    };

    // Error for any arms which don't follow the rules
    (@error) => {
        compile_error!(concat!(
            "`platform_select!` must have arms `Avx2(...)`, `Neon(...)` and `Generic(...)`, ",
            "in that order, once each",
        ))
    };
}

pub(super) use platform_select;

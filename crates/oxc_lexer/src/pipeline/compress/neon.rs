use oxc_span::Span;

use crate::lanes::Lanes;

use crate::pipeline::platform::Neon;

use super::compress_impl;

/// [`compress`] for Neon.
///
/// [`compress_impl`] is inlined into it. `#[inline(never)]` so it stays a separate function.
///
/// [`compress`]: super::compress
#[inline(never)]
#[target_feature(enable = "neon")]
pub(super) unsafe fn compress_neon(
    src: &[u8],
    n: usize,
    nb: usize,
    st: *const u64,
    kind: *const u8,
    stage_pos: *mut u32,
    stage_kind: *mut u8,
    out_kinds: *mut u8,
    out_spans: *mut Span,
    lanes: &mut Lanes,
    feat: Neon,
) -> usize {
    compress_impl(src, n, nb, st, kind, stage_pos, stage_kind, out_kinds, out_spans, lanes, feat)
}

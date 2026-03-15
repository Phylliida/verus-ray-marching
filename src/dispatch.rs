use vstd::prelude::*;
use verus_cutedsl::layout::*;
use verus_cutedsl::shape::*;
use verus_cutedsl::predication::*;
use verus_cutedsl::gemm::TensorSpec;
use verus_cutedsl::gemm::tensor_valid;
use verus_vulkan::parallel_dispatch::*;
use verus_vulkan::compute_validation::*;

verus! {

// ── Types ───────────────────────────────────────────────────────────────

/// Ray march dispatch configuration: image dimensions + tile sizes + grid dims.
pub ghost struct RayMarchDispatchConfig {
    /// Image width in pixels.
    pub width: nat,
    /// Image height in pixels.
    pub height: nat,
    /// Tile size in X (pixels per workgroup, horizontal).
    pub tile_x: nat,
    /// Tile size in Y (pixels per workgroup, vertical).
    pub tile_y: nat,
    /// Number of workgroups in X.
    pub grid_x: nat,
    /// Number of workgroups in Y.
    pub grid_y: nat,
}

/// Output image buffer binding: a 2D row-major layout + data size.
pub ghost struct ImageBufferBinding {
    pub layout: LayoutSpec,
    pub data_size: nat,
}

// ── Layout helpers using cutedsl ────────────────────────────────────────

/// Standard row-major image layout: shape = [height, width].
/// Offset(y, x) = y * width + x.
pub open spec fn image_layout(width: nat, height: nat) -> LayoutSpec {
    make_row_major(seq![height, width])
}

/// Standard image buffer binding.
pub open spec fn image_binding(width: nat, height: nat) -> ImageBufferBinding {
    ImageBufferBinding {
        layout: image_layout(width, height),
        data_size: shape_size(seq![height, width]),
    }
}

// ── Grid / tile helpers ─────────────────────────────────────────────────

/// Grid dimensions match tile counts (using cutedsl predication).
pub open spec fn grid_matches_tiles(config: &RayMarchDispatchConfig) -> bool {
    config.grid_x == num_tiles_ceil(config.width, config.tile_x)
    && config.grid_y == num_tiles_ceil(config.height, config.tile_y)
}

/// Linearize 2D tile coords to workgroup ID: wg_id = ti + tj * grid_x.
pub open spec fn tile_to_wg_id(ti: nat, tj: nat, grid_x: nat) -> nat {
    ti + tj * grid_x
}

/// Recover tile X index from workgroup ID.
pub open spec fn wg_id_to_ti(wg_id: nat, grid_x: nat) -> nat {
    wg_id % grid_x
}

/// Recover tile Y index from workgroup ID.
pub open spec fn wg_id_to_tj(wg_id: nat, grid_x: nat) -> nat {
    wg_id / grid_x
}

/// Is pixel (px, py) within workgroup tile (ti, tj) a valid (non-padding) pixel?
/// Uses cutedsl's tile_element_valid on each axis independently.
pub open spec fn pixel_valid(
    config: &RayMarchDispatchConfig,
    ti: nat, tj: nat,
    px: nat, py: nat,
) -> bool {
    tile_element_valid(ti, config.tile_x, px, config.width)
    && tile_element_valid(tj, config.tile_y, py, config.height)
}

/// Global pixel coordinates from tile + local position.
pub open spec fn global_x(ti: nat, tile_x: nat, px: nat) -> nat {
    ti * tile_x + px
}

pub open spec fn global_y(tj: nat, tile_y: nat, py: nat) -> nat {
    tj * tile_y + py
}

/// Pixel linear index in the output buffer (row-major: gy * width + gx).
/// Uses cutedsl's linearize for the canonical mapping.
pub open spec fn pixel_index(
    config: &RayMarchDispatchConfig,
    ti: nat, tj: nat,
    px: nat, py: nat,
) -> nat {
    let gx = global_x(ti, config.tile_x, px);
    let gy = global_y(tj, config.tile_y, py);
    // Row-major linearize: for shape [H, W], linearize([x, y], [W, H]) = x + W*y
    // But our layout is make_row_major([H, W]) with offset = gy*W + gx.
    // Direct arithmetic is clearest and provably equal to layout.offset(...)
    gy * config.width + gx
}

// ── Write set construction ──────────────────────────────────────────────

/// Whether (ti, tj, px, py) represents a valid pixel write by a workgroup.
pub open spec fn is_valid_pixel_write(
    config: &RayMarchDispatchConfig,
    ti: nat, tj: nat,
    px: nat, py: nat,
    idx: nat,
) -> bool {
    px < config.tile_x && py < config.tile_y
    && pixel_valid(config, ti, tj, px, py)
    && pixel_index(config, ti, tj, px, py) == idx
}

/// The set of output buffer indices that workgroup wg_id writes.
pub open spec fn rm_wg_write_set(
    config: &RayMarchDispatchConfig,
    wg_id: nat,
) -> Set<nat> {
    let ti = wg_id_to_ti(wg_id, config.grid_x);
    let tj = wg_id_to_tj(wg_id, config.grid_x);
    Set::new(|idx: nat|
        exists|px: nat, py: nat|
            #[trigger] is_valid_pixel_write(config, ti, tj, px, py, idx)
    )
}

// ── Safety predicate ────────────────────────────────────────────────────

/// Full ray march dispatch safety predicate.
pub open spec fn ray_march_dispatch_safe(
    config: &RayMarchDispatchConfig,
    output: &ImageBufferBinding,
    limits: ComputeLimits,
) -> bool {
    // Image dimensions positive
    &&& config.width > 0
    &&& config.height > 0
    // Tile sizes positive
    &&& config.tile_x > 0
    &&& config.tile_y > 0
    // Grid matches tile counts
    &&& grid_matches_tiles(config)
    // Vulkan dispatch limits
    &&& dispatch_dimensions_valid(config.grid_x, config.grid_y, 1, limits)
    // Output buffer is valid tensor (data large enough for layout)
    &&& tensor_valid(&TensorSpec { layout: output.layout, data_size: output.data_size })
    // Output layout matches the standard image layout
    &&& output.layout == image_layout(config.width, config.height)
    // Data size sufficient
    &&& output.data_size >= config.width * config.height
}

// ── ParallelKernelSpec conversion ───────────────────────────────────────

/// Instantiate the parallel kernel spec for ray marching.
/// `pixel_value` is an abstract function mapping (gx, gy) to the output value
/// (e.g., ray intersection result, color, depth).
pub open spec fn ray_march_as_parallel_kernel(
    config: &RayMarchDispatchConfig,
    output: &ImageBufferBinding,
    pixel_value: spec_fn(nat, nat) -> int,
) -> ParallelKernelSpec {
    ParallelKernelSpec {
        num_workgroups: config.grid_x * config.grid_y,
        write_set: |wg_id: nat| rm_wg_write_set(config, wg_id),
        write_value: |wg_id: nat, idx: nat| {
            let ti = wg_id_to_ti(wg_id, config.grid_x);
            let tj = wg_id_to_tj(wg_id, config.grid_x);
            let pair = choose|pair: (nat, nat)|
                #[trigger] is_valid_pixel_write(
                    config, ti, tj, pair.0, pair.1, idx);
            let gx = global_x(ti, config.tile_x, pair.0);
            let gy = global_y(tj, config.tile_y, pair.1);
            pixel_value(gx, gy)
        },
        data_size: output.data_size,
    }
}

// ── Proofs ──────────────────────────────────────────────────────────────

/// Different workgroup IDs map to different (ti, tj) pairs when grid_x > 0.
pub proof fn lemma_wg_id_to_tile_injective(
    wg_id1: nat, wg_id2: nat, grid_x: nat,
)
    requires
        grid_x > 0,
        wg_id1 != wg_id2,
    ensures
        wg_id_to_ti(wg_id1, grid_x) != wg_id_to_ti(wg_id2, grid_x)
        || wg_id_to_tj(wg_id1, grid_x) != wg_id_to_tj(wg_id2, grid_x),
{
    if wg_id_to_ti(wg_id1, grid_x) == wg_id_to_ti(wg_id2, grid_x)
        && wg_id_to_tj(wg_id1, grid_x) == wg_id_to_tj(wg_id2, grid_x)
    {
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(wg_id1 as int, grid_x as int);
        vstd::arithmetic::div_mod::lemma_fundamental_div_mod(wg_id2 as int, grid_x as int);
    }
}

/// Inner helper: no index belongs to both write sets of distinct workgroups.
/// Different tiles write to different pixel ranges (non-overlapping 2D tiles).
proof fn lemma_disjoint_tiles_inner(
    config: &RayMarchDispatchConfig,
    w1: nat, w2: nat,
)
    requires
        config.tile_x > 0,
        config.tile_y > 0,
        config.grid_x > 0,
        w1 != w2,
        wg_id_to_ti(w1, config.grid_x) != wg_id_to_ti(w2, config.grid_x)
        || wg_id_to_tj(w1, config.grid_x) != wg_id_to_tj(w2, config.grid_x),
    ensures
        rm_wg_write_set(config, w1).disjoint(rm_wg_write_set(config, w2)),
{
    let s1 = rm_wg_write_set(config, w1);
    let s2 = rm_wg_write_set(config, w2);
    let ti1 = wg_id_to_ti(w1, config.grid_x);
    let tj1 = wg_id_to_tj(w1, config.grid_x);
    let ti2 = wg_id_to_ti(w2, config.grid_x);
    let tj2 = wg_id_to_tj(w2, config.grid_x);

    // Show no idx is in both sets by showing different tiles → different global coords → different pixel_index
    assert forall|idx: nat, px1: nat, py1: nat, px2: nat, py2: nat|
        #[trigger] is_valid_pixel_write(config, ti1, tj1, px1, py1, idx)
        && #[trigger] is_valid_pixel_write(config, ti2, tj2, px2, py2, idx)
        implies false
    by {
        let gx1 = global_x(ti1, config.tile_x, px1);
        let gy1 = global_y(tj1, config.tile_y, py1);
        let gx2 = global_x(ti2, config.tile_x, px2);
        let gy2 = global_y(tj2, config.tile_y, py2);

        // Different tiles → non-overlapping ranges on at least one axis
        if ti1 != ti2 {
            if ti1 < ti2 {
                assert(gx1 < (ti1 + 1) * config.tile_x) by (nonlinear_arith)
                    requires px1 < config.tile_x, gx1 == ti1 * config.tile_x + px1;
                assert((ti1 + 1) * config.tile_x <= ti2 * config.tile_x) by (nonlinear_arith)
                    requires ti1 < ti2, config.tile_x > 0;
                assert(gx2 >= ti2 * config.tile_x) by (nonlinear_arith)
                    requires gx2 == ti2 * config.tile_x + px2;
            } else {
                assert(gx2 < (ti2 + 1) * config.tile_x) by (nonlinear_arith)
                    requires px2 < config.tile_x, gx2 == ti2 * config.tile_x + px2;
                assert((ti2 + 1) * config.tile_x <= ti1 * config.tile_x) by (nonlinear_arith)
                    requires ti2 < ti1, config.tile_x > 0;
                assert(gx1 >= ti1 * config.tile_x) by (nonlinear_arith)
                    requires gx1 == ti1 * config.tile_x + px1;
            }
            assert(gx1 != gx2);
        } else {
            assert(tj1 != tj2);
            if tj1 < tj2 {
                assert(gy1 < (tj1 + 1) * config.tile_y) by (nonlinear_arith)
                    requires py1 < config.tile_y, gy1 == tj1 * config.tile_y + py1;
                assert((tj1 + 1) * config.tile_y <= tj2 * config.tile_y) by (nonlinear_arith)
                    requires tj1 < tj2, config.tile_y > 0;
                assert(gy2 >= tj2 * config.tile_y) by (nonlinear_arith)
                    requires gy2 == tj2 * config.tile_y + py2;
            } else {
                assert(gy2 < (tj2 + 1) * config.tile_y) by (nonlinear_arith)
                    requires py2 < config.tile_y, gy2 == tj2 * config.tile_y + py2;
                assert((tj2 + 1) * config.tile_y <= tj1 * config.tile_y) by (nonlinear_arith)
                    requires tj2 < tj1, config.tile_y > 0;
                assert(gy1 >= tj1 * config.tile_y) by (nonlinear_arith)
                    requires gy1 == tj1 * config.tile_y + py1;
            }
            assert(gy1 != gy2);
        }

        // Different global coords → different pixel_index
        // pixel_index = gy * width + gx. If gx differs: obvious. If gy differs:
        // gy1 * width + gx1 != gy2 * width + gx2 when gy1 != gy2 and gx1,gx2 < width
        // (This is row-major injectivity for valid coords.)
        // But wait — we only know both indices equal idx. If (gx1,gy1) != (gx2,gy2)
        // and both map to the same idx, that's a contradiction for valid coords.
        // pixel_index(...) = gy*width + gx is injective when gx < width.
        if gx1 != gx2 || gy1 != gy2 {
            // Both equal idx, so gy1*width + gx1 == gy2*width + gx2
            // For injectivity: if gy1 == gy2 then gx1 == gx2. Contradiction with gx1 != gx2.
            // If gy1 != gy2: WLOG gy1 < gy2. Then gy1*w + gx1 < (gy1+1)*w <= gy2*w <= gy2*w + gx2.
            if gx1 != gx2 && gy1 == gy2 {
                // gy1*w + gx1 == gy2*w + gx2 and gy1==gy2 → gx1==gx2. Contradiction.
            }
            if gy1 != gy2 {
                if gy1 < gy2 {
                    assert(gx1 < config.width);
                    assert(gy1 * config.width + gx1 < (gy1 + 1) * config.width) by (nonlinear_arith)
                        requires gx1 < config.width;
                    assert((gy1 + 1) * config.width <= gy2 * config.width) by (nonlinear_arith)
                        requires gy1 < gy2, config.width > 0;
                } else {
                    assert(gx2 < config.width);
                    assert(gy2 * config.width + gx2 < (gy2 + 1) * config.width) by (nonlinear_arith)
                        requires gx2 < config.width;
                    assert((gy2 + 1) * config.width <= gy1 * config.width) by (nonlinear_arith)
                        requires gy2 < gy1, config.width > 0;
                }
            }
        }
    }

    assert forall|idx: nat| !(s1.contains(idx) && s2.contains(idx)) by {}
    assert(s1.disjoint(s2));
}

/// Write disjointness: distinct workgroups write to disjoint pixel sets.
pub proof fn lemma_rm_writes_disjoint(
    config: &RayMarchDispatchConfig,
)
    requires
        config.tile_x > 0,
        config.tile_y > 0,
        config.width > 0,
        config.grid_x > 0,
        grid_matches_tiles(config),
    ensures
        forall|w1: nat, w2: nat|
            w1 < config.grid_x * config.grid_y
            && w2 < config.grid_x * config.grid_y
            && w1 != w2
            ==> #[trigger] rm_wg_write_set(config, w1)
                .disjoint(rm_wg_write_set(config, w2)),
{
    assert forall|w1: nat, w2: nat|
        w1 < config.grid_x * config.grid_y
        && w2 < config.grid_x * config.grid_y
        && w1 != w2
        implies #[trigger] rm_wg_write_set(config, w1)
            .disjoint(rm_wg_write_set(config, w2))
    by {
        lemma_wg_id_to_tile_injective(w1, w2, config.grid_x);
        lemma_disjoint_tiles_inner(config, w1, w2);
    }
}

/// All pixel writes land within the output buffer.
pub proof fn lemma_rm_writes_in_bounds(
    config: &RayMarchDispatchConfig,
    output: &ImageBufferBinding,
)
    requires
        config.width > 0,
        config.height > 0,
        config.tile_x > 0,
        config.tile_y > 0,
        output.data_size >= config.width * config.height,
    ensures
        forall|w: nat, idx: nat|
            w < config.grid_x * config.grid_y
            && #[trigger] rm_wg_write_set(config, w).contains(idx)
            ==> idx < output.data_size,
{
    assert forall|ti: nat, tj: nat, px: nat, py: nat, idx: nat|
        #[trigger] is_valid_pixel_write(config, ti, tj, px, py, idx)
        implies idx < output.data_size
    by {
        // Unfold pixel_valid → tile_element_valid → global coords in bounds
        let gx = global_x(ti, config.tile_x, px);
        let gy = global_y(tj, config.tile_y, py);
        assert(tile_element_valid(ti, config.tile_x, px, config.width));
        assert(ti * config.tile_x + px < config.width);
        assert(gx < config.width);
        assert(tile_element_valid(tj, config.tile_y, py, config.height));
        assert(tj * config.tile_y + py < config.height);
        assert(gy < config.height);

        // pixel_index = gy * width + gx == idx
        assert(gy * config.width + gx == idx);

        // gy * width + gx < height * width since gx < width, gy < height
        assert(gy * config.width + gx < config.height * config.width) by (nonlinear_arith)
            requires gx < config.width, gy < config.height, config.width > 0;
        assert(config.height * config.width == config.width * config.height) by (nonlinear_arith);
    }
}

/// Master theorem: parallel ray march dispatch equals sequential execution.
pub proof fn lemma_ray_march_parallel_dispatch_correct(
    config: &RayMarchDispatchConfig,
    output: &ImageBufferBinding,
    limits: ComputeLimits,
    pixel_value: spec_fn(nat, nat) -> int,
    initial_data: Seq<int>,
)
    requires
        ray_march_dispatch_safe(config, output, limits),
        initial_data.len() == output.data_size,
    ensures ({
        let k = ray_march_as_parallel_kernel(config, output, pixel_value);
        parallel_result(k, initial_data)
            =~= sequential_result(k, initial_data, k.num_workgroups)
    }),
{
    let k = ray_march_as_parallel_kernel(config, output, pixel_value);

    // grid_x > 0: num_tiles_ceil(width, tile_x) >= 1 since width > 0 && tile_x > 0
    let den = config.tile_x as int;
    let num = (config.width + config.tile_x - 1) as int;
    assert(num >= den);
    vstd::arithmetic::div_mod::lemma_div_by_self(den);
    vstd::arithmetic::div_mod::lemma_div_is_ordered(den, num, den);
    assert(config.grid_x > 0);

    // 1. Write disjointness
    lemma_rm_writes_disjoint(config);

    // 2. Write in-bounds
    lemma_rm_writes_in_bounds(config, output);

    // 3. Apply master parallel ↔ sequential theorem
    lemma_parallel_equals_sequential(k, initial_data);
}

} // verus!

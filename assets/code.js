/* Interactive source workbench data.
 *
 * Every block is excerpted from the MagiAttention repository (main branch,
 * 2026-08 snapshot) with real line numbers. Blocks may elide debug prints,
 * static compile branches for other architectures, and long assertion lists;
 * every elision is marked with a `# …` comment. Do not treat these snippets
 * as runnable in isolation — follow the GitHub link for full context.
 */
(function () {
  "use strict";

  var C = String.raw;

  window.MAGI_CODE = {
    /* ------------------------------------------------------------------ *
     * Chapter 00 · AttnSlice contract & host-side dispatch
     * ------------------------------------------------------------------ */
    attnslice: {
      path: "magi_attention/kernel/cutedsl/flex_flash_attn.py",
      blocks: [
        {
          id: "01",
          title: "前向入口：q/k_ranges + mask_type",
          start: 74,
          end: 112,
          code: C`def _flex_flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    q_ranges: torch.Tensor | None = None,
    k_ranges: torch.Tensor | None = None,
    mask_type: int = MT_MAP.full,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    softmax_scale: float | None = None,
    softcap: float | None = None,
    sink: torch.Tensor | None = None,
    sink_layout: AttnSinkLayout = "sh",
    pack_gqa: bool | None = None,
    flex_attn_args: TorchFlexAttnArgs | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for FlexFlashAttention.

    Args:
        q_ranges/k_ranges: [N, 2] int32 cuda tensors of [start, end) q/k
            ranges. For now only ranges equivalent to a cu_seqlens
            partition are supported (see ranges_to_cu_seqlens); they are
            collapsed to cu_seqlens before reaching the kernel.
        out: Optional pre-allocated output tensor.
        lse: Optional pre-allocated log-sum-exp tensor.

    Returns:
        (output, lse):
        - output: (batch, seqlen_q, num_head, head_dim_v),
          or (total_q, num_head, head_dim_v) if q_ranges is provided.
        - lse: (batch, num_head, seqlen_q), or (num_head, total_q).
    """`
        },
        {
          id: "02",
          title: "Step-1：ranges 折叠为 cu_seqlens",
          start: 113,
          end: 131,
          code: C`    arch, major_arch = get_device_arch()
    validate_arch(arch, major_arch)

    assert (
        sink_layout == "sh"
    ), f"only sink_layout='sh' is supported, got {sink_layout!r}"

    # Step-1 hack: only q/k ranges equivalent to a cu_seqlens partition are
    # supported, so collapse them to cu_seqlens here and keep the
    # kernel-facing internals on cu_seqlens for now.
    cu_seqlens_q = ranges_to_cu_seqlens(q_ranges)
    cu_seqlens_k = ranges_to_cu_seqlens(k_ranges)

    # Unpack the torch FlexAttention-style / block-sparse args (fwd uses these).
    flex_attn_args = flex_attn_args or TorchFlexAttnArgs()
    score_mod = flex_attn_args.score_mod
    mask_mod = flex_attn_args.mask_mod
    aux_tensors = flex_attn_args.aux_tensors
    block_sparse_tensors = flex_attn_args.block_sparse_tensors`
        },
        {
          id: "03",
          title: "out / lse 的分配与累加语义",
          start: 191,
          end: 228,
          code: C`    out_torch_dtype = q.dtype
    device = q.device
    q_batch_seqlen_shape = (
        (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    )
    lse_shape = (  # (b, nh, sq) or (nh, tq)
        (batch_size, num_head, seqlen_q)
        if cu_seqlens_q is None
        else (num_head, total_q)
    )

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape,
            num_head,
            head_dim_v,
            dtype=out_torch_dtype,
            device=device,
        )
    else:
        validate_tensor(
            out, "out",
            (*q_batch_seqlen_shape, num_head, head_dim_v),
            out_torch_dtype, device,
        )

    if lse is None:
        lse = torch.empty(lse_shape, dtype=torch.float32, device=device)
    else:
        validate_tensor(lse, "lse", lse_shape, torch.float32, device)

    if seqlen_k == 0 or total_q == 0:
        out.zero_()
        if lse is not None:
            lse.fill_(float("-inf"))
        return out, lse`
        },
        {
          id: "04",
          title: "mask 类型折叠与调度启发式",
          start: 230,
          end: 316,
          code: C`    dtype = to_cute_dtype(q.dtype)
    use_block_sparsity = block_sparse_tensors is not None

    local = False
    # NOTE: only a single mask type shared by all q/k ranges is supported
    # for now, so collapse mask_type down to the legacy causal bool for the
    # host-side heuristics and for the kernels that still take is_causal
    # (all but SM100).
    causal = mask_type == MT_MAP.causal
    if mask_mod is not None:
        causal = False
        mask_type = MT_MAP.full

    requested_use_clc_scheduler = is_ffa_clc_enabled()
    requested_disable_2cta = is_ffa_2cta_disabled(is_fwd=True)

    # default
    tile_m, tile_n = 128, 128
    # … (省略 SM80/SM90/SM120 的 tile 选择分支)

    seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
    if major_arch == 10:
        q_stage = 2 if seqlen_q_packgqa > tile_m else 1
    else:
        q_stage = 1

    use_2cta_instrs = (
        major_arch in [10, 11]
        and not requested_disable_2cta
        and not causal
        and not local
        and cu_seqlens_q is None
        and not use_block_sparsity
        and int(math.ceil(head_dim / 16) * 16) in [128, 192]
        and int(math.ceil(head_dim_v / 16) * 16) == 128
        and seqlen_q_packgqa > 2 * tile_m
        and (tile_m % qhead_per_kvhead == 0 or not pack_gqa)
    )

    # CLC regressed for varlen MHA and dense noncausal. Imbalanced varlen
    # shapes keep more K/V blocks in flight and hurt L2; dense noncausal
    # mostly just pays work-stealing overhead.
    is_varlen_mha = is_varlen and qhead_per_kvhead == 1
    is_dense_noncausal = not is_varlen and not causal and not local
    use_clc_scheduler = (
        requested_use_clc_scheduler
        and not is_varlen_mha
        and not is_dense_noncausal
    )`
        },
        {
          id: "05",
          title: "compile_key：一切静态分支进 key",
          start: 342,
          end: 373,
          code: C`    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        mask_type,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        block_sparse_broadcast_pattern,
        aux_tensor_metadata,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        sink is not None,
        # … (省略两项 block_sparse 元数据开关)
        tile_m,
        tile_n,
        q_stage,
        pack_gqa,
        arch,
        use_2cta_instrs,
        q_subtile_factor,
        mma_pv_is_rs,
        intra_wg_overlap,
        use_clc_scheduler,
        magiattn_cutedsl.is_ffa_debug_mode_enabled(),
    )

    if compile_key not in _flex_flash_attn_fwd.compile_cache:`
        },
        {
          id: "06",
          title: "SM100 kernel 实例化",
          start: 445,
          end: 467,
          code: C`            case 10 | 11:
                ffa_fwd_obj = FFAFwdSm100(
                    head_dim=head_dim,
                    head_dim_v=head_dim_v,
                    qhead_per_kvhead=qhead_per_kvhead,
                    mask_type=mask_type,
                    is_local=local,
                    is_split_kv=False,
                    pack_gqa=pack_gqa,
                    m_block_size=tile_m,
                    n_block_size=tile_n,
                    q_stage=q_stage,
                    is_persistent=not causal and not local
                        and cu_seqlens_q is None,
                    score_mod=score_mod,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    paged_kv_non_tma=False,
                    is_varlen_q=cu_seqlens_q is not None,
                    q_subtile_factor=q_subtile_factor,
                    use_2cta_instrs=use_2cta_instrs,
                    use_clc_scheduler=use_clc_scheduler,
                    debug_print=magiattn_cutedsl.is_ffa_debug_mode_enabled(),
                )`
        },
        {
          id: "07",
          title: "编译缓存与 tvm-ffi 启动",
          start: 521,
          end: 557,
          code: C`        _flex_flash_attn_fwd.compile_cache[compile_key] = cute.compile(
            *compile_args, options="--enable-tvm-ffi"
        )

    q_call, k_call, v_call = q.detach(), k.detach(), v.detach()
    call_args = [
        q_call,
        k_call,
        v_call,
        out.detach(),
        lse,
        softmax_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        None,  # seqlen_used_q
        None,  # seqlen_used_k
        None,  # page_table
        None,  # window_size_left
        None,  # window_size_right
        sink,
    ]
    if major_arch in [10, 11]:
        # FP8 descale tensors removed; SM100 descale slot is always None.
        call_args.append(None)
    call_args.extend(
        [
            block_sparse_call_tuple(normalized_block_sparse_tensors),
            aux_tensors,
        ]
    )

    _flex_flash_attn_fwd.compile_cache[compile_key](*call_args)

    return out, lse


_flex_flash_attn_fwd.compile_cache = get_jit_cache("fwd")`
        }
      ]
    },

    /* ------------------------------------------------------------------ *
     * Chapter 01 · Blackwell execution substrate (TMEM / UMMA / TMA)
     * ------------------------------------------------------------------ */
    blackwell: {
      path: "magi_attention/kernel/cutedsl/ffa_fwd_sm100.py",
      blocks: [
        {
          id: "01",
          title: "CTA group 与 MMA tiler",
          start: 232,
          end: 270,
          code: C`        self.cta_group_size = 2 if self.use_2cta_instrs else 1

        # NOTE: cta_tiler M includes only 1 CTA, the scheduler will take
        # into account the cluster shape.
        # (tileQ128*stageQ, tileK128, tileHD128) per CTA
        # which shards Q/S/P/O along sq dim to Qi/Si/Pi/Oi, i={0,1}
        self.cta_tiler = (
            self.q_stage * m_block_size,
            n_block_size,
            self.head_dim_padded,
        )

        # NOTE: With 2CTA, the MMA tiler M covers both CTAs, so it's
        # cta_group_size * m_block_size. Each CTA owns m_block_size rows
        # and n_block_size//2 cols of sA/sB across 2 CTAs, and then
        # produces [m_block_size x n_block_size] partial tC each
        self.mma_tiler_qk = (  # (tileQ128*CTA2, tileK128, tileHD128)
            self.cta_group_size * m_block_size,
            n_block_size,
            self.head_dim_padded,
        )
        self.mma_tiler_pv = (  # (tileQ128*CTA2, tileHD128, tileK128)
            self.cta_group_size * m_block_size,
            self.head_dim_v_padded,
            n_block_size,
        )

        # epi_tile is per-CTA (not full 2CTA)
        # since each CTA writes its own O portion
        self.epi_tile = (self.m_block_size, self.head_dim_v_padded)

        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.cluster_shape_mn = (2, 1) if self.use_2cta_instrs else (1, 1)
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )`
        },
        {
          id: "02",
          title: "TMEM 512 列的静态规划",
          start: 399,
          end: 418,
          code: C`        self.tmem_s_offset = [0, self.n_block_size]  # [0, tileK) = [0, 128)
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size
            + i * self.head_dim_v_padded
            for i in range(self.q_stage)
        ]  # [2*tileK, 2*tileK + tileHD) = [256, 384)
        self.tmem_total = (
            self.tmem_o_offset[-1] + self.head_dim_v_padded
        )  # 2 * (tileK + tileHD) = 512
        assert self.tmem_total <= self.tmem_alloc_cols

        # bf16 tP only needs half of fp32 tS space
        self.tmem_s_to_p_offset = self.n_block_size // 2
        self.tmem_p_offset = [  # [tileK//2, tileK//2 + tileK] = [64, 192)
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]

        # vec buffer for row_max & row_sum with tmem shape [128, 2)
        # reuse tS space since we don't need tS after softmax
        self.tmem_vec_offset = self.tmem_s_offset`
        },
        {
          id: "03",
          title: "tcgen05 TiledMMA：S=QKᵀ 与 O=PV",
          start: 782,
          end: 833,
          code: C`        # Thr Layout VMNK: (2,1,1,1):(1,0,0,0)
        # MMA Atom
        # ThrID:           2:1
        # Shape MNK:       (256,128,16)
        # TV Layout A:     (2,(128,16)):(128,(1,256))
        # TV Layout B:     (2,(64,16)):(64,(1,128))
        # TV Layout C:     (2,(128,128)):(128,(1,256))
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            q_major_mode,
            k_major_mode,
            self.qk_acc_dtype,
            self.cta_group,
            self.mma_tiler_qk[:2],
        )

        self.cta_group_shape = tiled_mma_qk.thr_id.shape  # (2,1)
        cta_layout_vmnk = (  # (CTA_V(2),CTA_M1,CTA_N1,CTA_K1)
            cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (self.cta_group_shape,),
            )
        )

        # --- Make tiled MMA for O=PV ---

        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = tcgen05.OperandMajorMode.K

        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.v_dtype,
            p_major_mode,
            v_major_mode,
            self.pv_acc_dtype,
            self.cta_group,
            self.mma_tiler_pv[:2],
            p_source,
        )`
        },
        {
          id: "04",
          title: "SMEM 布局：K/V 复用同一块物理内存",
          start: 835,
          end: 856,
          code: C`        # --- Make smem layout for sQ/sK/sV/sO ---

        # sQ: S<3,4,3> o 0 o (MMA_sA=(128,16),MMA_Q1,MMA_HD=(4,2),stageQ)
        # sK: S<3,4,3> o 0 o (MMA_sB=(64,16),MMA_K1,MMA_HD=(4,2),stageK)
        # tP: S<3,4,3> o 0 o (MMA_sA=(128,16),MMA_Q1,MMA_K=(4,2),stageS)
        # sV: S<3,4,3> o 0 o (MMA_sB=(64,16),MMA_K1,MMA_HD=(4,2),stageK)
        # sO: S<3,4,3> o 0 o (EPI_Q=(8,16),EPI_HD=(64,2),EPI_STAGE=(1,2))
        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.s_stage
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.q_stage
        )`
        },
        {
          id: "05",
          title: "TMA atom：G2S 加载与 S2G 写回",
          start: 914,
          end: 980,
          code: C`        # --- Make tiled TMA G2S-copy for Q/K/V ---

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)

        tma_atom_Q = None
        if const_expr(self.use_tma_Q):
            tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                mQ,
                cute.select(sQ_layout, mode=[0, 1, 2]),  # slice out stage dim
                self.mma_tiler_qk,
                tiled_mma_qk,
                cta_layout_vmnk.shape,
            )
        # … (省略非 TMA 的 cp.async 回退分支)

        tma_atom_K = None
        tma_atom_V = None
        if const_expr(self.use_tma_KV):
            tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mK,
                cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                cta_layout_vmnk.shape,
            )
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mV,
                cute.select(sV_layout, mode=[0, 1, 2]),
                self.mma_tiler_pv,
                tiled_mma_pv,
                cta_layout_vmnk.shape,
            )

        # --- Make tiled TMA S2G-copy of O ---

        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_O = None
        if const_expr(self.use_tma_O):
            tma_atom_O, mO = cpasync.make_tiled_tma_atom(
                tma_store_op, mO,
                cute.select(sO_layout, mode=[0, 1]), self.epi_tile
            )`
        },
        {
          id: "06",
          title: "TMEM 张量视图：tStS / tOtO / tOrP",
          start: 1703,
          end: 1742,
          code: C`        # --- Make tmem fragments of tS/tP/tO ---

        thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
        thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

        # tStS: (MMA_tC=(128,128),1,1,S_STAGE2):((65536,1),0,0,128)
        qk_acc_shape = thr_mma_qk.partition_shape_C(  # (tileQ128*CTA2,tileK128)
            self.mma_tiler_qk[:2]
        )
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))

        # tOtO: (MMA_tC=(128,128),1,1,2):((65536,1),0,0,128)
        pv_acc_shape = thr_mma_pv.partition_shape_C(  # (tileQ128*CTA2,tileHD128)
            self.mma_tiler_pv[:2]
        )
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.q_stage))
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)

        # tOrP: P 直接以 bf16 视图叠放在 tS 的 TMEM 空间上
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)  # reuse tS for tP
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        # Need to multiply by width ratio bc tP is in v_dtype
        # but tmem offsets are in FP32
        tP_width_ratio = Float32.width // self.v_dtype.width  # 2 for bf16
        tP_stage_stride = (  # 256 for bf16
            self.tmem_p_offset[1] - self.tmem_p_offset[0]  # 192-64=128
        ) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(
                tOrP.layout,
                cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)),
            ),
        )`
        },
        {
          id: "07",
          title: "TMEM 的分配与归还（仅 MMA warp）",
          start: 1931,
          end: 1975,
          code: C`        if warp_idx == self.mma_warp_id:
            # --- Decrease rmem usage ---

            cute.arch.setmaxregister_decrease(self.num_regs_other)

            # --- Alloc and retrieve tmem buffer ---

            tmem.allocate(self.tmem_alloc_cols)  # cute.arch.alloc_tmem
            tmem.wait_for_alloc()  # tmem_alloc_barrier.arrive_and_wait
            tmem_ptr = tmem.retrieve_ptr(  # cute.arch.retrieve_tmem_ptr
                self.qk_acc_dtype
            )

            # --- Enter mma loop ---

            self.mma(
                tiled_mma_qk, tiled_mma_pv,
                sQ, sK, sV,
                tStS, tOtO, tOrP,
                pipeline_q, pipeline_kv,
                pipeline_s_p_o, pipeline_p_lastsplit, pipeline_o_acc,
                is_leader_cta,
                block_info, num_splits, SeqlenInfoCls,
                blocksparse_tensors,
                tile_scheduler=tile_scheduler,
                is_print_block=is_print_block,
            )

            # --- Dealloc tmem buffer ---

            tmem.relinquish_alloc_permit()
            tmem.wait_for_alloc()
            tmem.free(tmem_ptr)  # dealloc_mbar.arrive_wait + dealloc_tmem`
        }
      ]
    },

    /* ------------------------------------------------------------------ *
     * Chapter 02 · Warp specialization & pipelines
     * ------------------------------------------------------------------ */
    pipeline: {
      path: "magi_attention/kernel/cutedsl/ffa_fwd_sm100.py",
      blocks: [
        {
          id: "01",
          title: "16 个 warp 的角色表",
          start: 353,
          end: 397,
          code: C`        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14,)
        self.empty_warp_ids = (15,)
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
                *self.empty_warp_ids,
            )
        )

        # q_stage==1 时 softmax1 让位；varlen 时 correction 兼任 epilogue
        if self.q_stage == 1:
            if not self.use_tma_KV or not self.use_tma_Q:
                self.empty_warp_ids = self.empty_warp_ids + self.load_warp_ids
                self.load_warp_ids = self.softmax1_warp_ids
            else:
                self.empty_warp_ids = self.empty_warp_ids + self.softmax1_warp_ids
            self.softmax1_warp_ids = ()
        elif not self.use_tma_KV:
            self.load_warp_ids = (14, 15)
            self.empty_warp_ids = ()

        if self.use_correction_warps_for_epi:
            self.empty_warp_ids = self.empty_warp_ids + self.epilogue_warp_ids
            self.epilogue_warp_ids = self.correction_warp_ids

        self.clc_scheduler_warp_id = (
            self.empty_warp_ids[0] if self.use_clc_scheduler else None
        )`
        },
        {
          id: "02",
          title: "寄存器重分配：512 的账本",
          start: 420,
          end: 450,
          code: C`        # Look up tuning config for register counts and ex2_emu params
        _tune_key = (
            self.use_2cta_instrs,
            self.is_causal,
            self.head_dim_padded,
            self.is_sm103,
        )
        self._tune = _TUNING_CONFIG.get(_tune_key, {})
        if "ex2_emu_freq" in self._tune:
            self.enable_ex2_emu = self._tune["ex2_emu_freq"] > 0
        if self.head_dim_padded < 96:
            self.num_regs_softmax = 200 if not paged_kv_non_tma else 184
            self.num_regs_correction = 64
            self.num_regs_other = 48 if not paged_kv_non_tma else 80
        else:
            if not paged_kv_non_tma and "num_regs_softmax" in self._tune:
                self.num_regs_softmax = self._tune["num_regs_softmax"]
                self.num_regs_correction = self._tune["num_regs_correction"]
            elif not paged_kv_non_tma:
                self.num_regs_softmax = 192
                self.num_regs_correction = 80
            else:
                self.num_regs_softmax = 184
                self.num_regs_correction = 64

            self.num_regs_total = 512
            self.num_regs_other = (
                self.num_regs_total
                - self.num_regs_softmax * 2
                - self.num_regs_correction
            )`
        },
        {
          id: "03",
          title: "Load 流水线：TMA → UMMA",
          start: 1440,
          end: 1498,
          code: C`        # Load Q pipeline:
        #   producer: load warp
        #     acquire: producer_acquire_w_index_phase(stage, phase) [before TMA]
        #     commit:  TMA hardware arrive on mbar_load_Q           [implicit]
        #   consumer: mma warp
        #     wait:    consumer_wait_w_index_phase(stage, phase)    [per Q-stage]
        #     release: consumer_release_w_index(stage)              [mma epilogue]
        #   full  = sQ[stage] written by TMA
        #   empty = mma warp released after all KV-blocks for this Q-tile
        if const_expr(self.use_tma_Q):
            pipeline_q = ffa_pipeline.PipelineTmaUmma.create(
                barrier_storage=mbar_load_Q,
                num_stages=self.q_stage,
                producer_group=load_warp,
                consumer_group=mma_warp,
                tx_count=self.tma_copy_bytes["Q"],
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,  # sync later by our own
            )
        # … (省略非 TMA 的 PipelineAsyncUmma 分支)

        # Load KV pipeline:
        #   producer: load warp   (TMA hardware commit)
        #   consumer: mma warp    (QK GEMM 消费 K, PV GEMM 消费 V)
        #   full  = sK[stage]/sV[stage] written by TMA
        #   empty = mma warp finished QK GEMM (K) or PV GEMM (V)
        if const_expr(self.use_tma_KV):
            pipeline_kv = ffa_pipeline.PipelineTmaUmma.create(
                barrier_storage=mbar_load_KV,
                num_stages=self.kv_stage,
                producer_group=load_warp,
                consumer_group=mma_warp,
                tx_count=self.tma_copy_bytes["K"],
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )`
        },
        {
          id: "04",
          title: "S/P/O 三态流水线：一个槽位两种转移",
          start: 1500,
          end: 1554,
          code: C`        # S/P/O triple-state pipeline (MMA ↔ softmax+correction):
        #   This pipeline has dual semantics — it tracks two transitions
        #   per slot:
        #     (1) S-full:    MMA commits after QK GEMM;
        #                    softmax waits before T2R load S
        #     (2) P+O-empty: softmax + correction both release;
        #                    MMA waits before PV GEMM
        #   The consumer group is softmax_correction_threads_cluster (both
        #   warps), so the "empty" barrier requires arrivals from BOTH
        #   softmax (P written) AND correction (O rescaled).
        #   This is why softmax+correction are bundled together as consumers.
        pipeline_s_p_o = ffa_pipeline.PipelineUmmaAsync.create(
            barrier_storage=mbar_S_full_P_full_O_rescaled,
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=softmax_correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        # P-lastsplit pipeline (softmax → MMA hardware, split_P_arrive only):
        #   PV GEMM starts as soon as the 1st half of P is ready in tmem;
        #   the 2nd half arrival is signalled here so the hardware GEMM
        #   knows when the full P is ready before reading the 2nd half.
        pipeline_p_lastsplit = (
            ffa_pipeline.PipelineAsyncUmma.create(
                barrier_storage=mbar_P_full_lastsplit,
                num_stages=self.q_stage,
                producer_group=softmax_warps_cluster,
                consumer_group=mma_warp,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )
        )`
        },
        {
          id: "05",
          title: "O-acc 流水线：只护住最后一块",
          start: 1556,
          end: 1580,
          code: C`        # O-accumulation pipeline (MMA → correction, FINAL tile only):
        #   Unlike a typical ring-buffer pipeline, this is used ONLY for
        #   the last KV-block of each Q-tile. During the mainloop,
        #   correction does NOT need to wait for O because:
        #     by the time softmax signals (sm_stats_barrier) that S(i) is
        #     done, O(i-1) from PV GEMM must have also completed
        #     (GEMM ordering guarantee).
        #   Only in the epilogue (last KV-block) does the ordering break —
        #   softmax signals row_sum before the next tile's GEMM starts, so
        #   correction MUST explicitly wait for final O.
        #
        #   producer: mma warp
        #     commit:  producer_commit_w_index(stage)   [epilogue GEMM only]
        #   consumer: correction warp (epilogue only)
        #     wait:    consumer_wait_w_index_phase(stage, phase)
        #   full  = final O[stage] written to tmem by last PV GEMM
        pipeline_o_acc = ffa_pipeline.PipelineUmmaAsync.create(
            barrier_storage=mbar_O_full,
            num_stages=self.q_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads_cluster,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )`
        },
        {
          id: "06",
          title: "sScale 的 RAW/WAR 双保险",
          start: 1603,
          end: 1641,
          code: C`        # Softmax-stats pipeline (softmax → correction): WAR guard on sScale.
        #   sScale[stage] is written by softmax (corr_scale in mainloop,
        #   final row_sum/row_max in epilogue) and read by correction.
        #   Two hazards exist on this shared slot:
        #     - RAW: correction must not read before softmax writes
        #            → handled by sm_stats_barrier
        #     - WAR: softmax must not overwrite before correction reads
        #            the prev value → handled HERE
        #
        #   The correction mainloop releases with a CROSS index
        #   (q_stage-1-stage) instead of stage, turning this WAR
        #   backpressure into round-robin traffic control: a single
        #   correction warp group serves two softmax warp groups,
        #   staggering them so only one softmax wg overlaps with
        #   correction at a time while the other parks on its acquire.
        pipeline_sm_stats = ffa_pipeline.PipelineAsync.create(
            barrier_storage=mbar_softmax_stats,
            num_stages=self.q_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )

        # sm_stats NamedBarrier (softmax → correction): RAW guard on sScale.
        #   A 2-warp rendezvous signalling "sScale[stage] has been written,
        #   safe to read". Used for BOTH corr_scale (mainloop) and final
        #   row_sum/row_max (epilogue).
        sm_stats_barrier = ffa_pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0),
            num_threads=cute.arch.WARP_SIZE * 2,
        )`
        },
        {
          id: "07",
          title: "warp 分派：一份代码，七种角色",
          start: 1894,
          end: 2115,
          code: C`        # //////////////////////////////////////////////////////////////
        #  Load Warp
        # //////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                thr_mma_qk, thr_mma_pv, mQ, mK, mV, sQ, sK, sV,
                mPageTable, tma_atom_Q, tma_atom_K, tma_atom_V,
                gmem_tiled_copy_Q, pipeline_q, pipeline_kv,
                block_info, num_splits, SeqlenInfoCls,
                blocksparse_tensors, tile_scheduler=tile_scheduler,
                is_print_block=is_print_block,
            )

        # //////////////////////////////////////////////////////////////
        #  MMA Warp — 唯一分配 TMEM 的 warp
        # //////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.allocate(self.tmem_alloc_cols)
            # … (见第 1 章 Block 07)

        # //////////////////////////////////////////////////////////////
        #  Softmax WarpGroup 0/1 — 唯一增加寄存器的角色
        # //////////////////////////////////////////////////////////////
        if (
            const_expr(self.q_stage == 2) and warp_idx <= self.softmax1_warp_ids[-1]
        ) or (const_expr(self.q_stage == 1) and warp_idx <= self.softmax0_warp_ids[-1]):
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)

            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=softmax_scale_log2,
                # … (省略十余个透传参数)
            )
            if const_expr(not self.s0_s1_barrier):
                stage = Int32(
                    0
                    if const_expr(self.q_stage == 1)
                    or warp_idx < self.softmax1_warp_ids[0]
                    else 1
                )
                softmax_loop(stage=stage, tStS=tStS)
            # … (省略 s0_s1_barrier=True 的双 WG 分支)

            tmem_alloc_barrier.arrive()

        # //////////////////////////////////////////////////////////////
        #  Correction WarpGroup
        # //////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)

            self.correction_loop(
                thr_mma_qk, thr_mma_pv, tStS, tOtO, sScale,
                mO, mLSE, sO,
                pipeline_s_p_o, pipeline_o_acc,
                pipeline_sm_stats, sm_stats_barrier, pipeline_o_epi,
                learnable_sink, descale_tensors,
                gmem_tiled_copy_O, tma_atom_O, softmax_scale_log2,
                block_info, num_splits, SeqlenInfoCls,
                tile_scheduler=tile_scheduler,
                blocksparse_tensors=blocksparse_tensors,
                is_print_block=is_print_block,
            )

            tmem_alloc_barrier.arrive()`
        }
      ]
    },

    /* ------------------------------------------------------------------ *
     * Chapter 03 · Block-level masking
     * ------------------------------------------------------------------ */
    mask: {
      path: "magi_attention/kernel/cutedsl/mask.py",
      blocks: [
        {
          id: "01",
          title: "BlockInfo：整块跳过的几何",
          path: "magi_attention/kernel/cutedsl/block_info.py",
          start: 38,
          end: 74,
          code: C`    @cute.jit
    def get_n_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        split_idx: Int32 = 0,
        num_splits: Int32 = 1,
    ) -> Tuple[Int32, Int32]:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
        if const_expr(
            self.is_causal or (self.is_local and self.window_size_right is not None)
        ):
            m_idx_max = (m_block + 1) * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_max = cute.ceil_div(m_idx_max, self.qhead_per_kvhead_packgqa)
            # 端对齐 causal：行 i 能看的最大列 ≈ i + (seqlen_k - seqlen_q)
            n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_right = (
                n_idx if const_expr(self.is_causal) else n_idx + self.window_size_right
            )
            n_block_max = min(n_block_max, cute.ceil_div(n_idx_right, self.tile_n))
        n_block_min = 0
        if const_expr(self.is_local and self.window_size_left is not None):
            m_idx_min = m_block * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa > 1):
                m_idx_min = m_idx_min // self.qhead_per_kvhead_packgqa
            n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q
            n_idx_left = n_idx - self.window_size_left
            n_block_min = cutlass.max(n_idx_left // self.tile_n, 0)
        # … (省略 split_kv 对 [n_block_min, n_block_max) 的再切分)
        return n_block_min, n_block_max`
        },
        {
          id: "02",
          title: "三段主循环：partial → full → partial",
          path: "magi_attention/kernel/cutedsl/ffa_fwd_sm100.py",
          start: 3395,
          end: 3479,
          code: C`                    # --- Mainloop-1: S0/S1 with causal masking ---
                    if const_expr(self.is_causal or self.is_local):
                        n_block_min_causal_local_mask = (
                            block_info.get_n_block_min_causal_local_mask(
                                seqlen_info, m_block, n_block_min
                            )
                        )
                        for n_tile in cutlass.range(
                            n_block_max - n_block_min_causal_local_mask, unroll=1
                        ):
                            n_block = n_block_max - 1 - n_tile
                            (...) = softmax_step(
                                ...,
                                n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_causal_local_mask
                        )

                    # --- Mainloop-2: S0/S1 w/o masking ---
                    # NOTE: The remaining iterations have no masking,
                    # but may still need mask_mod
                    n_block_min_before_local_mask = (
                        block_info.get_n_block_min_before_local_mask(
                            seqlen_info, m_block, n_block_min
                        )
                    )
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_before_local_mask, unroll=1
                    ):
                        n_block = n_block_max - n_tile - 1
                        (...) = softmax_step(..., n_block=n_block)  # 无 mask_fn

                    # --- Mainloop-3: S0/S1 with local masking on the left ---
                    if const_expr(
                        self.is_local and block_info.window_size_left is not None
                    ):
                        n_block_max = cutlass.min(
                            n_block_max, n_block_min_before_local_mask
                        )
                        for n_tile in cutlass.range(
                            0, n_block_max - n_block_min, unroll=1
                        ):
                            n_block = n_block_max - 1 - n_tile
                            (...) = softmax_step(
                                ...,
                                n_block=n_block,
                                mask_fn=partial(mask_fn, mask_seqlen=False),
                            )`
        },
        {
          id: "03",
          title: "apply_mask_sm100：入口与坐标",
          start: 457,
          end: 493,
          code: C`    @cute.jit
    def apply_mask_sm100(
        self,
        acc_S: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        aux_tensors: Optional[list] = None,
        # … (省略 fastdiv/head_divmod/rBitmask 等参数)
        r2p: bool = True,
    ) -> None:
        assert not (
            mask_causal and mask_local
        ), "mask_causal and mask_local cannot be both True"
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(
            acc_shape if not self.swap_AB else acc_shape[::-1]
        )
        tScS = thr_mma.partition_C(cS)
        tScS = tScS[(None, None), 0, 0]
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        # To handle edge cases of completely masked out rows where
        # n_block_max = 0, we treat negative n_blocks as 0th n_block
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n`
        },
        {
          id: "04",
          title: "causal / local 的列界计算",
          start: 585,
          end: 650,
          code: C`        else:  # Causal or local
            causal_row_offset = self.seqlen_k - n_block * self.tile_n - self.seqlen_q
            row_idx = tScS_t2r[0][0] + m_block * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa != 1):
                row_idx = row_idx // self.qhead_per_kvhead_packgqa
            if const_expr(mask_causal):
                col_limit_right = row_idx + causal_row_offset + 1
                if const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                ncol = const_expr(cute.size(tScS_t2r.shape))
                if const_expr(not r2p):
                    for i in cutlass.range(ncol, unroll_full=True):
                        acc_S[i] = (
                            -Float32.inf
                            if tScS_t2r[i][1] >= col_limit_right
                            else acc_S[i]
                        )
                else:
                    mask_r2p_lambda(
                        acc_S,
                        lambda s: r2p_bitmask_below(col_limit_right, s),
                        rank1=True,
                    )
            else:  # local / sliding window
                # … (省略 window 左右界的推导)
                if const_expr(not r2p):
                    for i in cutlass.range(cute.size(tScS_t2r.shape), unroll_full=True):
                        col_idx = tScS_t2r[i][1]
                        acc_S[i] = (
                            -Float32.inf
                            if col_idx >= col_limit_right or col_idx < col_limit_left
                            else acc_S[i]
                        )
                else:
                    # Dual-bound R2P masking for SM100.
                    # Masks elements where:
                    #   NOT (col_limit_left <= col < col_limit_right)
                    def mask_gen_fn(s: int) -> Uint32:
                        return r2p_bitmask_below(
                            col_limit_right, s
                        ) & r2p_bitmask_above(col_limit_left, s)

                    mask_r2p_lambda(acc_S, mask_gen_fn, rank1=True)`
        },
        {
          id: "05",
          title: "R2P：一条位掩码顶 32 次比较",
          start: 37,
          end: 86,
          code: C`@cute.jit
def r2p_bitmask_below(limit: Int32, s: int) -> Uint32:
    """32-bit R2P bitmask keeping positions < limit (exclusive upper bound).

    Positions 0..limit-1 in chunk s get bit=1 (keep), the rest bit=0 (mask).
    Uses inline PTX to avoid shift-by-type-width UB.
    """
    m = max((s + 1) * MASK_R2P_CHUNK_SIZE - limit, 0)
    return cutedsl_utils.shr_u32(Uint32(0xFFFFFFFF), Uint32(m))


@cute.jit
def r2p_bitmask_above(limit: Int32, s: int) -> Uint32:
    """32-bit R2P bitmask keeping positions >= limit (inclusive lower bound)."""
    n = max(limit - s * MASK_R2P_CHUNK_SIZE, 0)
    return cutedsl_utils.shl_u32(Uint32(0xFFFFFFFF), Uint32(n))


@cute.jit
def mask_r2p_lambda(
    X: cute.Tensor,
    mask_gen_fn: cutlass.Constexpr[MaskGenFn],
    rank1: bool = False,
) -> None:
    """Apply R2P masking with a custom bitmask generator.

    mask_gen_fn(chunk_idx) -> Uint32: bit i set means column
    chunk_idx * chunk_size + i is KEPT; bit i clear means masked to -inf.
    """
    ncol = const_expr(
        cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape)
    )
    # 32-column chunks. The mask_gen_fn returns a Uint32 bitmask (1=keep).
    CHUNK_SIZE = MASK_R2P_CHUNK_SIZE
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, CHUNK_SIZE)):
        mask = mask_gen_fn(s)
        # This needs to be range_constexpr,
        # o/w the compiler can't generate the R2P instruction
        for i in cutlass.range_constexpr(min(CHUNK_SIZE, ncol - s * CHUNK_SIZE)):
            in_bound = cutlass.Boolean(mask & (Uint32(1) << i))
            c = s * CHUNK_SIZE + i
            if const_expr(rank1):
                X[c] = X[c] if in_bound else -Float32.inf
            else:
                for r in cutlass.range_constexpr(cute.size(X.shape[0])):
                    X[r, c] = X[r, c] if in_bound else -Float32.inf`
        },
        {
          id: "06",
          title: "mask_mod：FlexAttention 可编程谓词",
          start: 527,
          end: 583,
          code: C`        elif const_expr(not mask_causal and not mask_local and mask_mod is not None):
            # Block sparse case w/ mask_mod
            batch_idx_ssa = cutedsl_utils.scalar_to_ssa(batch_idx, cutlass.Int32)

            ncol = const_expr(cute.size(tScS_t2r.shape))
            for i in cutlass.range_constexpr(ncol):
                row_coord = tScS_t2r[i][0] if not self.swap_AB else tScS_t2r[i][1]
                col_coord = tScS_t2r[i][1] if not self.swap_AB else tScS_t2r[i][0]
                global_row = row_coord + m_block * self.tile_m
                global_col = col_coord + n_block * self.tile_n

                if const_expr(self.qhead_per_kvhead_packgqa != 1):
                    # PackGQA：把折叠行还原成 (mask_row, head_offset)
                    mask_row, head_offset = divmod(global_row, head_divmod)
                    head_idx_for_mod = (
                        head_idx * self.qhead_per_kvhead_packgqa + head_offset
                    )
                else:
                    head_idx_for_mod = head_idx
                    mask_row = global_row

                # … (省略 fastdiv 对 varlen 行列坐标的归一化)

                mask_value = mask_mod(
                    batch_idx_ssa,
                    head_idx_ssa,
                    mask_row_ssa,
                    kv_idx_ssa,
                    self.seqlen_info,
                    aux_tensors,
                )
                cond = cutlass.Boolean(cutedsl_utils.ssa_to_scalar(mask_value))
                acc_S[i] = acc_S[i] if cond else -Float32.inf
                if const_expr(mask_seqlen):
                    acc_S[i] = -Float32.inf if global_col >= self.seqlen_k else acc_S[i]
                if check_q_boundary:
                    acc_S[i] = -Float32.inf if mask_row >= self.seqlen_q else acc_S[i]`
        },
        {
          id: "07",
          title: "ranges → cu_seqlens 与 MT_MAP 现状",
          path: "magi_attention/kernel/cutedsl/ffa_utils.py",
          start: 44,
          end: 122,
          code: C`@dataclass(frozen=True)
class _MaskTypeMap:
    # 与 magi_attention.common.enum.AttnMaskType 对齐的整数映射
    full: ClassVar[int] = 0
    causal: ClassVar[int] = 1

    # TODO: support inv_causal and bi_causal
    # inv_causal: ClassVar[int] = 2
    # bi_causal: ClassVar[int] = 3

    def is_valid(self, mask_type: int) -> bool:
        """Check if the given mask type is valid."""
        return mask_type in range(2)  # Update if more mask types are added


MT_MAP = _MaskTypeMap()

# …

def ranges_to_cu_seqlens(ranges: torch.Tensor | None) -> torch.Tensor | None:
    """Collapse q/k ranges down to a cu_seqlens tensor (step-1 hack).

    contiguous, non-overlapping intervals starting at 0:
    [[0, e0], [e0, e1], ...] -> [0, e0, e1, ...].
    """
    if ranges is None:
        return None
    # … (省略连续性校验)
    cu_seqlens = torch.cat([ranges[:1, 0], ranges[:, 1]]).to(torch.int32)
    return cu_seqlens.contiguous()`
        }
      ]
    },

    /* ------------------------------------------------------------------ *
     * Chapter 04 · Online softmax on TMEM
     * ------------------------------------------------------------------ */
    softmax: {
      path: "magi_attention/kernel/cutedsl/ffa_fwd_sm100.py",
      blocks: [
        {
          id: "01",
          title: "T2R：把 S 从 TMEM 搬进寄存器",
          start: 3623,
          end: 3633,
          code: C`        # ///////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////

        # --- T2R copy S ---

        # Wait for tSi to be full
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)

        # T2R copy from tSi to rSi
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)`
        },
        {
          id: "02",
          title: "score_mod 与 mask 的挂载点",
          start: 3635,
          end: 3656,
          code: C`        # --- Update row_max/corr_scale ---

        # Apply score_mod on rSi if needed
        if const_expr(self.score_mod is not None):
            self.apply_score_mod(
                tSrS_t2r,
                thr_tmem_load,
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                softmax,
                seqlen,
                aux_tensors,
                fastdiv_mods,
                head_divmod,
            )

        # Apply mask fn on rSi if needed
        if const_expr(mask_fn is not None):
            mask_fn(tSrS_t2r, n_block=n_block)`
        },
        {
          id: "03",
          title: "row_max 更新与 corr_scale 发布",
          start: 3658,
          end: 3668,
          code: C`        # Update row_max and corr_scale
        row_max, corr_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        # R2S copy corr_scale if not the first KV tile
        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = corr_scale

        # Arrive corr_scale to be full
        # to notify the correction warp group to correct O
        sm_stats_barrier.arrive_w_index(index=stage * num_softmax_warps + warp_idx)`
        },
        {
          id: "04",
          title: "exp2 变换并把 P 写回 TMEM",
          start: 3670,
          end: 3718,
          code: C`        # --- Apply unnormalized softmax ---

        # Apply (rSi - row_max)
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)

        # Inter-softmax sequence barrier wait
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.wait(stage, s0_s1_sequence_phase)

        # Apply exp2((rSi - row_max)) and copy to rPi
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,  # bf16 view of rPi
            ex2_emu_freq=self.ex2_emu_freq if const_expr(mask_fn is None) else 0,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )

        # Inter-softmax sequence barrier arrive
        if const_expr(self.s0_s1_barrier):
            pipeline_s0_s1_sequence.sync_object_full.arrive(1 - stage, dst=None)

        # --- R2T copy P ---

        # R2T copy rPi to tPi
        r2t_cpy_iter_count = cute.size(tStP_r2t.shape[2])  # CPY_KFP32View4
        for i in cutlass.range_constexpr(r2t_cpy_iter_count):
            cute.copy(
                thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i]
            )

            # Release 1st half tSi to be empty => 1st half tPi to be full
            # to notify mma warp that the 1st half of tPi is ready
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = (
                    r2t_cpy_iter_count * self.split_P_arrive // self.n_block_size
                )
                if const_expr(split_P_arrive_idx == i + 1):
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_s_p_o.consumer_release_w_index(stage)

        # Release tSi to be empty / Commit (2nd half) tPi to be full
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(stage)
        else:
            pipeline_s_p_o.consumer_release_w_index(stage)`
        },
        {
          id: "05",
          title: "背压与 row_sum 的错峰更新",
          start: 3720,
          end: 3738,
          code: C`        # --- Backpressure ---

        # WAR acquire: before the next write to sScale[stage] (next step's
        # corr_scale, or the tile-end row_sum), wait until correction wg
        # has read the value just published above.
        # NOTE: With the correction wg mainloop's cross-release between two
        # stages, this also staggers the two softmax wgs, and allows
        # current stage of softmax computation to overlap with its
        # corresponding correction of O.
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)

        # Update row_sum with corr_scale in rmem
        softmax.update_row_sum(tSrS_t2r.load(), corr_scale, is_first)

        # Flip phases for the next KV tile
        return (
            mma_si_consumer_phase ^ 1,
            sm_stats_producer_phase ^ 1,
            s0_s1_sequence_phase ^ 1,
        )`
        },
        {
          id: "06",
          title: "ex2 仿真的调参表",
          start: 75,
          end: 111,
          code: C`# === TUNING KNOBS (agent-editable) ===
# Keys: (use_2cta_instrs: bool, is_causal: bool,
#        head_dim_padded: int, is_sm103: bool)
# Values:
#   ex2_emu_freq: int — how often to use emulated exp2
#                       (0=all hardware exp2, higher=more emulation).
#                       SM103 has fast native exp2, so set freq=0 there.
#   ex2_emu_res: int — (hd256 only) fragment-pairs per period to emulate.
#   ex2_emu_start_frg: int — fragment index to start emulation from
#   num_regs_softmax: int — register count for softmax warps
#   num_regs_correction: int — register count for correction warps
#   num_regs_other is derived: 512 - num_regs_softmax*2 - num_regs_correction
_TUNING_CONFIG = {
    (True, False, 128, False): {
        "ex2_emu_freq": 10,
        "ex2_emu_start_frg": 1,
        "num_regs_softmax": 176,
        "num_regs_correction": 88,
    },
    (False, True, 128, False): {
        "ex2_emu_freq": 16,
        "ex2_emu_start_frg": 1,
        "num_regs_softmax": 192,
        "num_regs_correction": 72,
    },
    # … (省略 hd192/hd256 与 SM103 的其它条目)
}`
        },
        {
          id: "07",
          title: "update_row_max：acc_scale 的来源",
          path: "magi_attention/kernel/cutedsl/softmax.py",
          start: 255,
          end: 275,
          code: C`    @cute.jit
    def update_row_max(
        self, acc_S_row: cute.TensorSSA, is_first: int
    ) -> Tuple[Float32, Float32]:
        if cutlass.const_expr(is_first):
            row_max_new = self._compute_row_max(acc_S_row)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale = 0.0
        else:
            row_max_old = self.row_max[0]
            row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
            acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
            if cutlass.const_expr(self.rescale_threshold > 0.0):
                if acc_scale_ >= -self.rescale_threshold:
                    row_max_new = row_max_old
                    row_max_safe = row_max_old
                    acc_scale = 1.0
        self.row_max[0] = row_max_new
        return row_max_safe, acc_scale`
        },
        {
          id: "08",
          title: "ex2_emulation_2：多项式仿真 exp2",
          path: "magi_attention/kernel/cutedsl/cutedsl_utils.py",
          start: 820,
          end: 838,
          code: C`@dsl_user_op
def ex2_emulation_2(
    x: Float32, y: Float32, *, poly_degree: int = 3, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    # We assume x <= 127.0 and y <= 127.0
    fp32_round_int = float(2**23 + 2**22)
    xy_clamped = (cute.arch.fmax(x, -127.0), cute.arch.fmax(y, -127.0))
    # We want to round down here, so that the fractional part is in [0, 1)
    xy_rounded = cute.arch.add_packed_f32x2(
        xy_clamped, (fp32_round_int, fp32_round_int), rnd="rm"
    )
    # The integer floor of x & y are now in the last 8 bits of xy_rounded
    # We want the next 2 ops to round to nearest even.
    xy_rounded_back = sub_packed_f32x2(xy_rounded, (fp32_round_int, fp32_round_int))
    xy_frac = sub_packed_f32x2(xy_clamped, xy_rounded_back)
    xy_frac_ex2 = evaluate_polynomial_2(*xy_frac, POLY_EX2[poly_degree], loc=loc, ip=ip)
    x_out = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0], loc=loc, ip=ip)
    y_out = combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1], loc=loc, ip=ip)
    return x_out, y_out`
        }
      ]
    },

    /* ------------------------------------------------------------------ *
     * Chapter 05 · Correction, epilogue & LSE
     * ------------------------------------------------------------------ */
    correction: {
      path: "magi_attention/kernel/cutedsl/ffa_fwd_sm100.py",
      blocks: [
        {
          id: "01",
          title: "sScale：每行两个槽位",
          start: 1695,
          end: 1701,
          code: C`        # sScale: (stageQ*tileQ*2=512):(1)
        # each q stage x each q row has 2 scale slot:
        #   1. corr_scale for mainloop correction
        #   2. final row_sum/row_max for epilogue correction
        sScale = storage.sScale.get_tensor(
            cute.make_layout(self.q_stage * self.m_block_size * 2)
        )`
        },
        {
          id: "02",
          title: "主循环：用 corr_scale 修正 tO(i-1)",
          start: 3948,
          end: 3993,
          code: C`                # --- Mainloop: correct tO(i-1) with corr_scale ---

                for i in cutlass.range(1, total_block_count, unroll=1):
                    for stage in cutlass.range_constexpr(self.q_stage):
                        # Wait for sScale(i) with corr_scale(i) to be full
                        sm_stats_barrier.arrive_and_wait_w_index(
                            index=stage * num_corr_warps + warp_idx
                        )

                        # Load corr_scale(i) from sScale(i)
                        corr_scale = sScale[tidx + stage * self.m_block_size]
                        should_rescale = (
                            cute.arch.vote_ballot_sync(corr_scale < 1.0) != 0
                        )

                        # NOTE: we don't need wait tO(i-1) to be full,
                        # since by the time the sScale(i) is ready,
                        # tS(i) must have been done, so tO(i-1) must have
                        # been done as well.

                        # Rescale tO(i-1) with corr_scale(i) if needed
                        if should_rescale:
                            self.correction_rescale(
                                thr_mma_pv,
                                tOtO[None, None, None, stage],
                                tidx,
                                corr_scale,
                            )

                        # Release tO(i) to be empty
                        # to notify mma warp that tO(i-1) has been rescaled
                        pipeline_s_p_o.consumer_release_w_index(stage)

                        # WAR release with CROSS index (q_stage-1-stage):
                        # free the *other* stage's slot instead of this one,
                        # so a single correction wg serves both softmax wgs
                        # round-robin.
                        pipeline_sm_stats.consumer_release_w_index(
                            self.q_stage - 1 - stage
                        )

                    # Flip phase for next KV tile
                    sm_stats_consumer_phase ^= 1`
        },
        {
          id: "03",
          title: "correction_rescale：T2R → 乘 → R2T",
          start: 4377,
          end: 4401,
          code: C`        # --- Rescale tO(i) ---

        for i in cutlass.range_constexpr(num_corr_tiles_hd):  # restHD loops
            # T2R copy tO(i) -> rO(i)
            tOrO_i = cute.make_rmem_tensor(tOrO_t2r_shape_i, self.pv_acc_dtype)
            tOtO_t2r_i = cute.make_tensor(
                tOtO_t2r.iterator + i * corr_tile_hd, tOtO_t2r.layout
            )
            cute.copy(thr_tmem_load, tOtO_t2r_i, tOrO_i)

            # Rescale rO(i) with corr_scale
            for j in cutlass.range(0, cute.size(tOrO_i), 2, unroll_full=True):
                tOrO_i[j], tOrO_i[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_i[j], tOrO_i[j + 1]), (scale, scale)
                )

            # R2T copy rO(i) -> tO(i)
            tOtO_r2t_i = cute.make_tensor(
                tOtO_r2t.iterator + i * corr_tile_hd, tOtO_r2t.layout
            )
            cute.copy(thr_tmem_store, tOrO_i, tOtO_r2t_i)

        # Ensure all stores to tO are visible to mma warps
        # with tcgen05.wait::st
        cute.arch.fence_view_async_tmem_store()`
        },
        {
          id: "04",
          title: "尾声：row_sum 归一化系数",
          start: 4032,
          end: 4085,
          code: C`                # Correct tO(-1) and write to smem/gmem
                for stage in cutlass.range_constexpr(self.q_stage):
                    # Wait for sScale(end) with final row_sum/row_max
                    sm_stats_barrier.arrive_and_wait_w_index(
                        index=stage * num_corr_warps + warp_idx
                    )

                    # Load final row_sum/row_max from sScale(end)
                    row_sum = sScale[tidx + stage * self.m_block_size]
                    if const_expr(mLSE is not None or learnable_sink is not None):
                        row_max = sScale[
                            tidx
                            + stage * self.m_block_size
                            + self.q_stage * self.m_block_size
                        ]

                    # WAR release (direct, epilogue)
                    pipeline_sm_stats.consumer_release_w_index(stage)

                    # Correct final row_sum with learnable sink if needed
                    if const_expr(learnable_sink is not None):
                        LOG2_E = math.log2(math.e)
                        sink_val = learnable_sink_val[stage]
                        if row_max == -Float32.inf:
                            row_max = sink_val * (LOG2_E / softmax_scale_log2_eff)
                            row_sum = max_offset_scale
                        else:
                            row_sum += cute.math.exp2(
                                sink_val * LOG2_E
                                - row_max * softmax_scale_log2_eff
                                + max_offset,
                                fastmath=True,
                            )

                    # Compute scale for tO(-1) row-sum normalization
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)
                    rowsum_norm_scale = cute.arch.rcp_approx(
                        row_sum if not acc_O_mn_row_is_zero_or_nan else 1.0
                    )

                    # Wait for tO(-1) to be full
                    # NOTE: we need to explicitly wait for tO(-1) since we
                    # don't have tS(end) to guarantee the GEMM ordering.
                    pipeline_o_acc.consumer_wait_w_index_phase(
                        stage, o_corr_consumer_phase
                    )`
        },
        {
          id: "05",
          title: "correction_epilogue 写 sO 并释放",
          start: 4095,
          end: 4124,
          code: C`                    # Correct tO(-1) with row-sum normalization,
                    # and write to smem buffer, and then gmem buffer
                    # in corr-epi mode
                    gO_stage = (
                        gO[None, None, stage] if const_expr(gO is not None) else None
                    )
                    self.correction_epilogue(
                        thr_mma_pv,
                        tOtO[None, None, None, stage],
                        tidx,
                        stage,
                        m_block,
                        seqlen_info.seqlen_q,
                        rowsum_norm_scale,
                        sO[None, None, stage],
                        mO_cur,
                        gO_stage,
                        gmem_tiled_copy_O,
                    )

                    # Signal for the next work tile that tO are already
                    # read, so mma warp can write to them
                    pipeline_s_p_o.consumer_release_w_index(stage)

                    # Commit sO to be full
                    # to notify the epilogue warp to write to gmem
                    if const_expr(not self.use_correction_warps_for_epi):
                        pipeline_o_epi.producer_commit_w_index(stage)`
        },
        {
          id: "06",
          title: "LSE 的合成与写出",
          start: 4173,
          end: 4223,
          code: C`            # --- Compute LSE and write to gmem ---

            if const_expr(mLSE is not None):
                if const_expr(not seqlen_info.has_cu_seqlens_q):
                    mLSE_cur = mLSE[None, head_idx, batch_idx]
                else:
                    offset = (
                        seqlen_info.offset_q
                        if const_expr(not self.pack_gqa)
                        else (0, seqlen_info.offset_q)
                    )
                    mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])

                for stage in cutlass.range_constexpr(self.q_stage):
                    m_tile_idx = (
                        m_block * self.q_stage + stage
                    ) * self.cta_group_size + mma_tile_coord_v
                    row_sum, row_max, acc_O_mn_row_is_zero_or_nan = stats[stage]
                    LN2 = math.log(2.0)
                    lse = (
                        (
                            row_max * softmax_scale_log2_eff
                            + (cute.math.log2(row_sum, fastmath=True) - max_offset)
                        )
                        * LN2
                        if not acc_O_mn_row_is_zero_or_nan
                        else -Float32.inf
                    )
                    seqlen_q = (
                        seqlen_info.seqlen_q
                        if const_expr(not self.pack_gqa)
                        else seqlen_info.seqlen_q * self.qhead_per_kvhead
                    )
                    gLSE = cute.local_tile(
                        mLSE_cur, (self.m_block_size,), (m_tile_idx,)
                    )
                    if tidx < seqlen_q - m_tile_idx * self.m_block_size:
                        # This actually just works with PackGQA too
                        gLSE[tidx] = lse`
        }
      ]
    },

    /* ------------------------------------------------------------------ *
     * Chapter 06 · Tile schedulers & CLC
     * ------------------------------------------------------------------ */
    scheduler: {
      path: "magi_attention/kernel/cutedsl/tile_scheduler.py",
      blocks: [
        {
          id: "01",
          title: "调度器选型的静态决策",
          path: "magi_attention/kernel/cutedsl/ffa_fwd_sm100.py",
          start: 339,
          end: 351,
          code: C`        self.sched_stages = 1
        self.scheduling_mode = (
            SchedulingMode.CLC if self.use_clc_scheduler else SchedulingMode.STATIC
        )

        if is_varlen_q:
            self.TileScheduler = SingleTileVarlenScheduler
        elif self.is_causal or self.is_local or self.use_clc_scheduler:
            self.TileScheduler = SingleTileLPTScheduler
        elif self.is_persistent:
            self.TileScheduler = StaticPersistentTileScheduler
        else:
            self.TileScheduler = SingleTileScheduler`
        },
        {
          id: "02",
          title: "L2 swizzle：估算 L2 能装几个 head",
          start: 452,
          end: 489,
          code: C`            size_one_kv_head = (
                args.seqlen_k * (args.headdim + args.headdim_v) * args.element_size
            )
            size_one_head = size_one_kv_head
            size_l2 = 50 * 1024 * 1024  # 40 MB for K & V
            # Swizzle is the size of each "section". Round swizzle to a
            # power of 2. swizzle is how many heads can fit in L2.
            log2_floor = lambda n: 31 - cutedsl_utils.clz(n)
            swizzle = (
                1
                if size_l2 < size_one_head
                else (1 << log2_floor(size_l2 // size_one_head))
            )
            # If we're in the last section (called residual), we don't want
            # to divide by swizzle. Instead we divide by the remainder.
            num_hb_quotient = (args.num_head * args.num_batch) // swizzle
            num_hb_remainder = (args.num_head * args.num_batch) % swizzle
            return SingleTileLPTScheduler.Params(
                total_blocks=args.num_block * args.num_head * args.num_batch,
                num_block=args.num_block,
                num_head=args.num_head,
                num_batch=args.num_batch,
                l2_minor=Int32(swizzle),
                num_head_divmod=FastDivmodDivisor(args.num_head),
                l2_minor_divmod=FastDivmodDivisor(swizzle),
                l2_major_divmod=FastDivmodDivisor(swizzle * args.num_block),
                l2_minor_residual_divmod=FastDivmodDivisor(max(num_hb_remainder, 1)),
                num_hb_quotient=Int32(num_hb_quotient),
                # … (省略 split_kv/cluster 参数透传)
                lpt=args.lpt,
            )`
        },
        {
          id: "03",
          title: "LPT 坐标映射：最重的块先跑",
          start: 605,
          end: 632,
          code: C`    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            assert self.clc is not None  # mypy
            work = self.clc.get_current_work()
            self._tile_idx = work.tile_idx[0]
            return self.clc_work_to_coords(work)
        # Static path: L2-swizzled coordinate mapping
        params = self.params
        # Implement LPT scheduling coordinate calculation
        bidhb, l2_mod = divmod(self._tile_idx, params.l2_major_divmod)
        # If we're in the last section (called residual), we don't want to
        # divide by swizzle. Instead we want to divide by the remainder.
        block, bidhb_residual = 0, 0
        if bidhb < params.num_hb_quotient:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_divmod)
        else:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_residual_divmod)
        bidhb_actual = bidhb * params.l2_minor + bidhb_residual
        batch_idx, head_idx = divmod(bidhb_actual, params.num_head_divmod)
        # Longest-processing-time-first
        if const_expr(params.lpt):
            block = params.num_block - 1 - block
        is_valid = self._tile_idx < params.total_blocks
        return WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(self._split_idx)),
            is_valid,
        )`
        },
        {
          id: "04",
          title: "ClcState：硬件动态派工的状态机",
          start: 53,
          end: 106,
          code: C`@dataclass
class ClcState(ParamsBase):
    """Owns the runtime state shared by CLC-capable tile schedulers.

    FFAFwdSm100 constructs this state because it owns the CLC
    response buffer, mbarrier storage, and launch geometry needed to
    initialize the hardware scheduler and async pipeline. Individual tile
    schedulers then consume this state and map the returned hardware work
    tiles into their own logical WorkTileInfo coordinates.
    """

    _hw_scheduler: ClcDynamicPersistentTileScheduler
    _pipeline: PipelineClcFetchAsync
    _consumer_state: PipelineState
    _producer_state: PipelineState

    def initial_work_tile_info(self):
        return self._hw_scheduler.initial_work_tile_info()

    def get_current_work(self):
        return self._hw_scheduler.get_current_work()

    def prefetch_next_work(self, *, loc=None, ip=None):
        self._pipeline.producer_acquire(self._producer_state, loc=loc, ip=ip)
        mbarrier_addr = self._pipeline.producer_get_barrier(
            self._producer_state, loc=loc, ip=ip
        )
        self._hw_scheduler.advance_to_next_work(mbarrier_addr, loc=loc, ip=ip)
        self._producer_state.advance(loc=loc, ip=ip)

    def consumer_wait(self, *, loc=None, ip=None):
        self._pipeline.consumer_wait(self._consumer_state, loc=loc, ip=ip)

    def consumer_release(self, *, loc=None, ip=None):
        self._pipeline.consumer_release(self._consumer_state, loc=loc, ip=ip)
        self._consumer_state.advance(loc=loc, ip=ip)

    def producer_tail(self, *, loc=None, ip=None):
        self._pipeline.producer_tail(self._producer_state, loc=loc, ip=ip)`
        },
        {
          id: "05",
          title: "CLC producer / consumer 分工",
          start: 642,
          end: 658,
          code: C`    def prefetch_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.prefetch_next_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.consumer_wait(loc=loc, ip=ip)
            work = self.get_current_work()
            self.clc.consumer_release(loc=loc, ip=ip)
            return work
        # Single tile scheduler:
        # set to invalid tile_idx to indicate no more work
        self._tile_idx = self.params.total_blocks
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.producer_tail(loc=loc, ip=ip)`
        },
        {
          id: "06",
          title: "CLC 调度 warp 的完整流程",
          path: "magi_attention/kernel/cutedsl/ffa_fwd_sm100.py",
          start: 4886,
          end: 4913,
          code: C`    def clc_scheduler_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        # /////////////////////////////////////////////////////////////
        #  Persistent tile scheduler loop
        # /////////////////////////////////////////////////////////////
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            tile_scheduler.prefetch_next_work()

            # Advance to next Q tile
            work_tile = tile_scheduler.advance_to_next_work()

        tile_scheduler.producer_tail()

    @cute.jit
    def empty_warp(
        self,
        tile_scheduler: TileSchedulerProtocol,
    ):
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            # Advance to next Q tile
            work_tile = tile_scheduler.advance_to_next_work()`
        },
        {
          id: "07",
          title: "host 侧的 CLC 回退启发式",
          path: "magi_attention/kernel/cutedsl/flex_flash_attn.py",
          start: 307,
          end: 316,
          code: C`    is_varlen = cu_seqlens_q is not None or cu_seqlens_k is not None

    # CLC regressed for varlen MHA and dense noncausal. Imbalanced varlen
    # shapes keep more K/V blocks in flight and hurt L2; dense noncausal
    # mostly just pays work-stealing overhead.
    is_varlen_mha = is_varlen and qhead_per_kvhead == 1
    is_dense_noncausal = not is_varlen and not causal and not local
    use_clc_scheduler = (
        requested_use_clc_scheduler and not is_varlen_mha and not is_dense_noncausal
    )`
        }
      ]
    },

    /* ------------------------------------------------------------------ *
     * Chapter 07 · Backward on SM100
     * ------------------------------------------------------------------ */
    backward: {
      path: "magi_attention/kernel/cutedsl/ffa_bwd_sm100.py",
      blocks: [
        {
          id: "01",
          title: "preprocess：D = rowsum(dO ⊙ O)",
          path: "magi_attention/kernel/cutedsl/ffa_bwd_preprocess.py",
          start: 341,
          end: 398,
          code: C`            # Sum across the "k" dimension
            pdpsum = (tOrO.load().to(Float32) * tOrdO.load().to(Float32)).reduce(
                cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1)
            )
            threads_per_row = gmem_tiled_copy_O.layout_src_tv_tiled[0].shape[0]
            pdpsum = cutedsl_utils.warp_reduce(
                pdpsum, operator.add, width=threads_per_row
            )
            PdP_sum = cute.make_rmem_tensor(cute.size(tOrO, mode=[1]), Float32)
            PdP_sum.store(pdpsum)

            # Write PdPsum from rmem -> gmem
            gPdPsum = cute.local_tile(mPdPsum_cur, (self.tile_m,), (m_block,))
            if tOcO[0, 0, 0][1] == 0:
                for m in cutlass.range(cute.size(PdP_sum), unroll_full=True):
                    row = tOcO[0, m, 0][0]
                    PdPsum_val = 0.0
                    if row < seqlen_limit:
                        PdPsum_val = PdP_sum[m]
                        if const_expr(mdLSE is not None):
                            PdPsum_val -= gdLSE[row]  # D' = D - dLSE
                    gPdPsum[row] = PdPsum_val

            # Clear dQaccum（atomic 累加的起点）
            if const_expr(mdQaccum is not None):
                # … (省略寻址)
                zero = cute.make_rmem_tensor_like(tdQgdQaccum)
                zero.fill(0.0)
                cute.copy(gmem_tiled_copy_dQaccum, zero, tdQgdQaccum)

            # LSE -> LSE * log2(e)，供主 kernel 的 exp2 路径
            if const_expr(mLSE is not None):
                LOG2_E = math.log2(math.e)
                if tidx < seqlen_q_rounded - m_block * self.tile_m:
                    gLSElog2[tidx] = lse * LOG2_E if lse != -Float32.inf else 0.0`
        },
        {
          id: "02",
          title: "反向 16 warp 角色与 5 个 MMA tiler",
          start: 120,
          end: 183,
          code: C`        # CTA tiler：反向以 K-block 为中心
        self.cta_tiler = (tile_n, tile_m, self.tile_hdim)

        # S.T  = K @ Q.T   => (tileK128*CTA2, tileQ128, tileHD128)
        self.mma_tiler_kq = (self.cta_group_size * tile_n, tile_m, self.tile_hdim)
        # dP.T = V @ dO.T  => (tileK128*CTA2, tileQ128, tileHD128)
        self.mma_tiler_vdo = (self.cta_group_size * tile_n, tile_m, self.tile_hdimv)
        # dV   = P.T @ dO  => (tileK128*CTA2, tileHD128, tileQ128)
        self.mma_tiler_pdo = (self.cta_group_size * tile_n, self.tile_hdimv, tile_m)
        # dK   = dS.T @ Q  => (tileK128*CTA2, tileHD128, tileQ128)
        self.mma_tiler_dsq = (self.cta_group_size * tile_n, self.tile_hdim, tile_m)
        # dQ   = dS @ K    => (tileQ128, tileHD128, tileK128*CTA2)
        # NOTE: for 2-CTA mode, reduction is along cluster-wide tileK dim.
        self.mma_tiler_dsk = (tile_m, self.tile_hdim, tile_n * self.cta_group_size)

        # …

        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.relay_warp_id = 14
        self.empty_warp_id = 15

        # 16 warps -> 512 threads
        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.reduce_warp_ids,
                *self.compute_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.relay_warp_id,
                self.empty_warp_id,
            )
        )`
        },
        {
          id: "03",
          title: "TMEM 复用：dQ 与 dP 叠在一起",
          start: 195,
          end: 227,
          code: C`        # TMEM buffer distribution
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        if self.use_2cta_instrs and self.tile_hdim == 192 and self.tile_hdimv == 128:
            # … (省略 hd192 特殊布局)
            pass
        else:
            self.tmem_S_offset = 0
            self.tmem_P_offset = 0  # embedded in left-half of S

            self.tmem_dV_offset = self.tmem_S_offset + self.tile_n

            self.tmem_dP_offset = self.tmem_dV_offset + self.tile_hdimv
            self.tmem_dS_offset = self.tmem_dP_offset  # embedded in left-half of dP

            # NOTE:
            # 1. in 1-CTA mode: tdQ (tileQ,tileHD) is fully overlapped with
            #    tdP, where dP(i) GEMM waits until dQ(i-1) GEMM finished and
            #    tdQ(i-1) consumed by dQacc wg
            # 2. in 2-CTA mode: tdQ (tileQ//2,tileHD) is embedded in the
            #    right-half of S, where S(i) GEMM waits until dQ(i-1)
            #    finished and tdQ(i-1) consumed by dQacc wg
            self.tmem_dQ_offset = (
                (self.tmem_S_offset + (self.tile_hdim // 2))
                if self.use_2cta_instrs  # 2-CTA: embedded in right-half of S
                else self.tmem_dP_offset  # 1-CTA: fully overlapped with dP
            )

            self.tmem_dK_offset = self.tmem_dP_offset + self.tile_m`
        },
        {
          id: "04",
          title: "主循环：5 GEMM 软件流水",
          start: 3739,
          end: 3833,
          code: C`                    # ////////////////////////////////////////////
                    #  Mainloop: GEMM for S(i),dK(i-1),dQ(i-1),dP(i),dV(i)
                    # ////////////////////////////////////////////

                    handle_Q_next = handle_Q
                    for i in cutlass.range(1, main_loop_iters, unroll=1):
                        # --- GEMM: S(i).T = K @ Q(i).T ---
                        handle_Q_next = pipeline_Q_consumer.wait_and_advance()
                        mma_s_qk_fn(B_idx=handle_Q_next.index)
                        pipeline_S_P.sync_object_full.arrive(
                            0, pipeline_S_P.producer_mask, cta_group
                        )

                        # --- GEMM: dK(i-1) = dS(i-1).T @ Q(i-1) ---
                        pipeline_dS.consumer_wait(consumer_state_dS)
                        mma_dk_dsq_fn(B_idx=handle_Q.index,
                                      zero_init=not accumulate_dK)
                        accumulate_dK = True
                        handle_Q.release()  # sQ(i-1) -> empty

                        # --- GEMM: dQ(i-1) = dS(i-1) @ K ---
                        # NOTE: no extra waits needed —
                        #   dS(i-1) already read for dK; tdQ(i-1) freed
                        #   before dP(i-1) GEMM
                        mma_dq_dsk_fn()
                        pipeline_dQ.sync_object_full.arrive(
                            0, pipeline_dQ.producer_mask, cta_group
                        )
                        pipeline_dS.consumer_release(consumer_state_dS)
                        consumer_state_dS.advance()

                        # --- GEMM: dP(i) = V @ dO(i).T ---
                        pipeline_dO.consumer_wait(consumer_state_dO)
                        # Acquire tdQ(i-1) empty => tdP(i) empty
                        # NOTE: in 1-CTA mode, tdQ is overlapped with tdP,
                        # so when tdQ(i-1) is consumed by dQacc warp,
                        # its tmem buffer is empty for tdP(i)
                        pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
                        mma_dp_vdo_fn(B_idx=consumer_state_dO.index)
                        pipeline_dP.sync_object_full.arrive(
                            0, pipeline_dP.producer_mask, cta_group
                        )
                        producer_phase_acc ^= 1

                        # --- GEMM: dV(i) = P(i).T @ dO(i) ---
                        pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                        mma_dv_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)
                        pipeline_dO.consumer_release(consumer_state_dO)
                        consumer_state_dO.advance()
                        handle_Q = handle_Q_next`
        },
        {
          id: "05",
          title: "softmax 重计算与 dS = P⊙(dP−D)",
          start: 4574,
          end: 4699,
          code: C`                    # Apply softmax-fwd: F = rS - rLSE, P = exp(F)
                    for v in cutlass.range_constexpr(
                        cute.size(tSrS_t2r, mode=[0]) // 2
                    ):
                        lse_pair = (tSrLSE[2 * v], tSrLSE[2 * v + 1])

                        # Apply F = rS * scale - rLSE = fma(rS, scale, -rLSE)
                        (
                            tSrS_cur[2 * v],
                            tSrS_cur[2 * v + 1],
                        ) = cute.arch.fma_packed_f32x2(
                            ((tSrS_cur[2 * v], tSrS_cur[2 * v + 1])),
                            (softmax_scale_log2, softmax_scale_log2),
                            (-lse_pair[0], -lse_pair[1]),
                        )

                        # Apply P = exp2(F)
                        tSrS_cur[2 * v] = cute.math.exp2(tSrS_cur[2 * v], fastmath=True)
                        tSrS_cur[2 * v + 1] = cute.math.exp2(
                            tSrS_cur[2 * v + 1], fastmath=True
                        )

                    # Type cast from rS to rP, then R2T copy rP to tP
                    cutedsl_utils.cvt_f16(tSrS_cur, tSrP_r2t[None, stage, 0, 0])
                    # … (省略 fence / barrier / R2T copy)

                # ////////////////////////////////////////////
                #  Softmax-bwd: rdS.T = rP.T * (rdP.T - rdPsum)
                # ////////////////////////////////////////////

                pipeline_dPsum.consumer_wait(consumer_state_dPsum)
                pipeline_dP.consumer_wait(consumer_state_S_P_dP)

                    # Apply softmax-bwd: rdS = rP * (rdP - rdPsum)
                    for v in cutlass.range_constexpr(
                        cute.size(tdPrdP_t2r, mode=[0]) // 2
                    ):
                        dPsum_pair = (tSrdPsum[2 * v], tSrdPsum[2 * v + 1])
                        (
                            tdPrdP_cur[2 * v],
                            tdPrdP_cur[2 * v + 1],
                        ) = quack.activation.sub_packed_f32x2(
                            (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), dPsum_pair
                        )
                        (
                            tdPrdP_cur[2 * v],
                            tdPrdP_cur[2 * v + 1],
                        ) = cute.arch.mul_packed_f32x2(
                            (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                            (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                        )`
        },
        {
          id: "06",
          title: "dQ：TMA atomic reduce + 确定性信号量",
          start: 5347,
          end: 5402,
          code: C`                    # Semaphore acquire
                    if const_expr(self.deterministic and stage == 0):
                        if not m_block_oob_upper:
                            lock_value = self._dq_semaphore_lock_value(
                                iter_idx, curr_q_cnt,
                                curr_dq_write_order, curr_dq_write_order_full,
                                blocksparse_tensors, block_info, seqlen_info,
                                m_block, n_block_cta_group,
                            )
                            cutedsl_utils.wait_eq(
                                mdQ_semaphore_cur[(m_block, None)].iterator,
                                tidx, cta_rank_in_cluster, lock_value,
                            )

                    # Sync before S2G copy
                    self.reduce_sync_barrier.arrive_and_wait()

                    # S2G copy dQacc (TMA atomic reduce)
                    if is_tma_warp and not m_block_oob_upper:
                        with cute.arch.elect_one():
                            copy_utils.cpasync_reduce_bulk_add_f32(
                                sdQacc[None, smem_idx].iterator,
                                gdQacc_cur[None, stage + stage_offset_cta].iterator,
                                self.tma_copy_bytes["dQ"] // 1,
                            )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(
                            self.sdQacc_stage - 1, read=read_flag
                        )

                    # Sync after S2G copy
                    self.reduce_sync_barrier.arrive_and_wait()
                    dQ_tma_store_producer_state.advance()

                    # Semaphore release：允许下一个 writer 写同一 m_block
                    if const_expr(
                        self.deterministic and stage == 0 and delay_semaphore_release
                    ):
                        if m_block > m_block_min:
                            cutedsl_utils.arrive_inc(
                                mdQ_semaphore_cur[(m_block - 1, None)].iterator,
                                tidx, cta_rank_in_cluster, 1,
                            )`
        },
        {
          id: "07",
          title: "postprocess：dQ = cast(dQaccum × scale)",
          path: "magi_attention/kernel/cutedsl/ffa_bwd_postprocess.py",
          start: 544,
          end: 649,
          code: C`            # Step 1: load dQaccum from gmem to smem
            cute.copy(g2s_tiled_copy_dQaccum, tdQgdQaccum, tdQsdQaccumg2s)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            # Step 2: load dQ from smem to rmem（乘 softmax_scale 并转 dtype）
            # …
            rdQ.store((acc.load() * scale).to(self.dtype))

            # Step 3: Copy dQ from register to smem
            # Step 4: load dQ from smem to rmem（合并访存重排）
            # Step 5: copy dQ from rmem to gmem, with seqlen predicate
            # … (省略各步的 tiled copy 细节)`
        }
      ]
    },

    /* ------------------------------------------------------------------ *
     * Chapter 08 · Communication–computation overlap
     * ------------------------------------------------------------------ */
    overlap: {
      path: "magi_attention/functional/dist_attn.py",
      blocks: [
        {
          id: "01",
          title: "前向 overlap 主环：prefetch / compute / reduce",
          start: 3181,
          end: 3294,
          code: C`        # init kernel barrier for native grpcoll to ensure comm kernel is always preceded by compute kernel
        kernel_barrier_fetch = KernelBarrier(
            dist_attn_runtime.fwd_kernel_barrier_fetch_target
        )
        kernel_barrier_reduce = KernelBarrier(
            dist_attn_runtime.fwd_kernel_barrier_reduce_target
        )

        # get local qkv and pre-fetch qkv for remote stage(s)
        local_q, local_kv = dist_attn_runtime.get_curr_q_kv_and_fetch_next(
            local_q=local_q,
            local_kv=(local_k, local_v),
            overlap_stage=None,
            kernel_barrier=kernel_barrier_fetch,
        )

        kernel_barrier_fetch.synchronize()
        (
            partial_local_out,
            partial_local_meta,
        ) = dist_attn_runtime.apply_fwd_partial_attn(
            q=local_q,
            kv=local_kv,
            overlap_stage=None,
            # …
        )
        partial_local_lse = partial_local_meta.lse
        # …

        # loop into remote stages
        for ith_overlap_stage in range(dist_attn_runtime.overlap_degree):
            # … debug logging elided

            # reset kernel barrier for next stage
            kernel_barrier_fetch.reset()

            # wait for ith remote qkv prepared and pre-fetch (i+1)th remote qkv
            (
                curr_remote_q,
                curr_remote_kv,
            ) = dist_attn_runtime.get_curr_q_kv_and_fetch_next(
                local_q=local_q,
                local_kv=local_kv,
                overlap_stage=ith_overlap_stage,
                kernel_barrier=kernel_barrier_fetch,
            )

            if not dist_attn_runtime.is_last_remote_stage(
                overlap_stage=ith_overlap_stage
            ):
                kernel_barrier_fetch.synchronize()

            if not dist_attn_runtime.is_first_remote_stage(
                overlap_stage=ith_overlap_stage
            ):
                kernel_barrier_reduce.synchronize()

            # apply fwd partial attn with ith remote qkv
            # overlapped with (i+1)th pre-fetch
            (
                partial_remote_out,
                partial_remote_meta,
            ) = dist_attn_runtime.apply_fwd_partial_attn(
                q=curr_remote_q,
                kv=curr_remote_kv,
                out_acc=partial_local_out
                if dist_attn_runtime.fwd_out_lse_use_acc
                else None,
                lse_acc=partial_local_lse
                if dist_attn_runtime.fwd_out_lse_use_acc
                else None,
                overlap_stage=ith_overlap_stage,
                # …
            )
            partial_remote_lse = (
                partial_remote_meta.lse if partial_remote_meta is not None else None
            )
            # …

            # reset kernel barrier for next stage
            kernel_barrier_reduce.reset()

            # reduce ith partial out with partial lse
            # overlapped with (i+1)th fwd partial attn and maybe (i+2)th pre-fetch
            dist_attn_runtime.reduce_partial_out_lse(
                partial_remote_out=partial_remote_out,
                partial_remote_lse=partial_remote_lse,
                partial_local_out=partial_local_out,
                partial_local_lse=partial_local_lse,
                ref_remote_out=curr_remote_q,
                overlap_stage=ith_overlap_stage,
                kernel_barrier=kernel_barrier_reduce,
            )

        # prepare reduced local out and lse
        # before returning from forward and saving for backward
        local_out, local_lse = dist_attn_runtime.prepare_reduced_local_out_lse(
            partial_local_out=partial_local_out,
            partial_local_lse=partial_local_lse,
            ref_local_out=local_q,
        )`
        },
        {
          id: "02",
          title: "等当前段 · 发下一段：wait_post_process 与两种预取姿势",
          start: 367,
          end: 456,
          code: C`    def get_curr_q_kv_and_fetch_next(
        self,
        local_q: torch.Tensor,
        local_kv: FusedOrTupleTensor,
        overlap_stage: int | None = None,
        kernel_barrier: KernelBarrier | None = None,
    ) -> tuple[torch.Tensor, FusedOrTupleTensor]:
        # … docstring elided
        next_stage = self.get_next_stage(overlap_stage)
        is_host_stage = self.is_host_stage(overlap_stage)
        is_last_remote_stage = self.is_last_remote_stage(overlap_stage)

        # wait for host/remote qkv prepared for current stage
        if is_host_stage:
            local_q, local_kv = self._maybe_flatten_local_qkv_head_groups(
                local_q=local_q,
                local_kv=local_kv,
            )
            local_kv = self._maybe_concat(*local_kv, need_concat=self.concat_kv)
            curr_q, curr_kv = local_q, local_kv
        else:
            curr_remote_stage = self.get_curr_remote_stage(overlap_stage)
            (
                remote_q_work,
                remote_q_buffer,
            ) = self.remote_q_work_with_buffer_per_stage[curr_remote_stage]
            curr_q = remote_q_work.wait_post_process(remote_q_buffer)

            (
                remote_kv_work,
                remote_kv_buffer,
            ) = self.remote_kv_work_with_buffer_per_stage[curr_remote_stage]
            curr_kv = remote_kv_work.wait_post_process(remote_kv_buffer)

        # pre-fetch remote qkv for next stage(s)
        if self.prefetch_stage_by_stage and not is_last_remote_stage:
            # if using stage-by-stage prefetch, we only pre-fetch the next stage
            # to avoid blocking the current ffa fwd
            (
                self.remote_q_work_with_buffer_per_stage[next_stage]
            ) = self._fetch_remote_q(
                local_q=local_q,
                overlap_stage=next_stage,
                buffer_name=GrpCollBufferName.GroupCastQO,
                kernel_barrier=kernel_barrier,
            )
            (
                self.remote_kv_work_with_buffer_per_stage[next_stage]
            ) = self._fetch_remote_kv(
                local_kv=local_kv,
                overlap_stage=next_stage,
                buffer_name=GrpCollBufferName.GroupCastDefault,
                kernel_barrier=kernel_barrier,
            )
        elif is_host_stage:
            # when not using stage-by-stage prefetch,
            # we issue all fetch-remote comms in advance of ffa fwd
            # and ffa fwd can still overlap with these comms
            # with the support of non-zero 'sm_margin', thanks to persistent kernel design
            self.remote_q_work_with_buffer_per_stage = [
                self._fetch_remote_q(
                    local_q=local_q,
                    overlap_stage=ith_stage,
                    buffer_name=GrpCollBufferName.GroupCastQO,
                    kernel_barrier=kernel_barrier,
                )
                for ith_stage in range(self.overlap_degree)
            ]
            self.remote_kv_work_with_buffer_per_stage = [
                self._fetch_remote_kv(
                    local_kv=local_kv,
                    overlap_stage=ith_stage,
                    buffer_name=GrpCollBufferName.GroupCastDefault,
                    kernel_barrier=kernel_barrier,
                )
                for ith_stage in range(self.overlap_degree)
            ]

        return curr_q, curr_kv`
        },
        {
          id: "03",
          title: "prefetch_stage_by_stage 与 fwd_sm_margin",
          start: 1048,
          end: 1099,
          code: C`    @property
    def prefetch_stage_by_stage(self) -> bool:
        """
        NOTE:
        1. When CUDA_DEVICE_MAX_CONNECTIONS == 1, prefetch must be done stage-by-stage to avoid blocking
           the FFA forward/backward computation; otherwise only the last stage's prefetch can overlap with
           computation.
        2. When native grpcoll is enabled, prefetch must also be done stage-by-stage to avoid blocking
           the FFA forward/backward computation; otherwise only the last stage's prefetch can overlap with
           computation (unless allocating many grpcoll buffers, which is very memory intensive because each
           grpcoll_buffer's memory is managed separately).
        """
        return (
            env.general.is_cuda_device_max_connections_one()
            or env.comm.is_native_grpcoll_enable()
        )

    # … kernel_backend / use_native_grpcoll / enable_qo_comm 等属性省略

    @property
    def fwd_sm_margin(self) -> int:
        """
        Get the forward sm_margin reserved for communication.

        1. When native grpcoll is enabled, a kernel barrier guarantees the correct ordering
           between communication and compute kernels, so no additional sm_margin is required;
           return 0.
        2. Otherwise, return the saved sm_margin for communication to allow communication to
           properly overlap with computation.
        """
        if env.comm.is_native_grpcoll_enable():
            return 0
        else:
            return env.comm.ffa_fwd_sm_margin_save_for_comm()`
        },
        {
          id: "04",
          title: "group_cast：一段发送到多个 rank",
          path: "magi_attention/comm/primitive/grpcoll/_group_collective.py",
          start: 81,
          end: 205,
          code: C`def group_cast(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_sizes: list[int] | torch.Tensor,
    output_split_sizes: list[int] | torch.Tensor,
    dst_indices: list[list[int]] | torch.Tensor,
    src_index: list[int] | torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = False,
    cast_lse: bool = False,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    """Group cast interface

    Args:
        input (torch.Tensor): input tensor with shape [input_seqlen, ...]
        # …

        dst_indices (list[list[int]] | torch.Tensor):
            the 2D destination rank indices list / tensor for each input split to send to,
            # … 每个输入段一张目的 rank 清单（可多播）

        src_index (list[int] | torch.Tensor):
            the 1D source rank index list / tensor for each output split to receive from,
            where len(src_index) == len(output_split_sizes)

            NOTE:
                1. the order of the output splits are "stable",
                i.e. the ones from the same source will be in the same order as the input splits
        # …
    """

    if env.comm.is_hierarchical_comm_enable():
        # NOTE: a workaround to reduce inter-comm overhead by hierarchical group-cast
        return hier_group_cast_impl_with_a2av(
            # … 节点内 a2av + 跨节点 a2av 两级
        )

    if env.comm.is_native_grpcoll_enable():
        # NOTE: a feature under early development
        return native_group_cast_impl(
            # … NVLink/RDMA 对称缓冲 · 免 pack/unpack
        )

    # fall back to the a2a-v implementation
    return a2av_group_cast_impl(
        input=input,
        output=output,
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        dst_indices=dst_indices,
        src_index=src_index,
        group=group,
        async_op=async_op,
        cast_lse=cast_lse,
        input_lse=input_lse,
        output_lse=output_lse,
        **kwargs,
    )`
        },
        {
          id: "05",
          title: "group_reduce：合并多个 rank 的对应数据段",
          path: "magi_attention/comm/primitive/grpcoll/_group_collective.py",
          start: 255,
          end: 408,
          code: C`def group_reduce(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_sizes: list[int] | torch.Tensor,
    output_split_sizes: list[int] | torch.Tensor,
    dst_index: list[int] | torch.Tensor,
    src_indices: list[list[int]] | torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = False,
    reduce_op: GroupReduceOp = "sum",
    acc_reduce: bool = True,
    comm_dtype: torch.dtype | None = None,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    """Group reduce interface

    Args:
        # …
        src_indices (list[list[int]] | torch.Tensor):
            the 2D source rank indices list / tensor for each output split to reduce from,

            NOTE:
                # …
                3. since any reduce operation satisfies the commutative property,
                the order to reduce to the same output split does not matter, except for numerical errors

        reduce_op (GroupReduceOp): the reduce operation to use. Defaults to "sum"
            - "sum": sum reduction
            - "avg": average reduction
            - "lse": log-sum-exp weighted average reduction, with lse correction

            NOTE:
                if reduce_op is "lse", the user is required to pass "input_lse" and "output_lse",
                and we only support input/output with shape [seqlen, num_heads, head_dim]
                while input_lse/output_lse with shape [seqlen, num_heads] for now

        acc_reduce (bool): whether to accumulate the reduction to the given output buffer. Defaults to ''True''.
        # …
    """

    if env.comm.is_hierarchical_comm_enable():
        # NOTE: a workaround to reduce inter-comm overhead by hierarchical group collective
        # which might be deprecated when the native hierarchical group collective is ready
        return hier_group_reduce_impl_with_a2av(
            # …
        )

    if env.comm.is_native_grpcoll_enable():
        # NOTE: the new feature under development
        # which might be the default implementation in the future
        return native_group_reduce_impl(
            # … fp32 归约融合进通信 kernel · comm_dtype 可低精传输
        )

    # fall back to the original a2a-v implementation
    return a2av_group_reduce_impl(
        # … a2av 收齐 partial 段 · post_process 里本地 sum/lse 归约
    )`
        },
        {
          id: "06",
          title: "a2av 降解：pack → all2all_v → post_process",
          path: "magi_attention/comm/primitive/grpcoll/_a2av_grpcoll_impl.py",
          start: 69,
          end: 179,
          code: C`def a2av_group_cast_impl(
    input: torch.Tensor,
    output: torch.Tensor | None,
    input_split_sizes: list[int] | torch.Tensor,
    output_split_sizes: list[int] | torch.Tensor,
    dst_indices: list[list[int]] | torch.Tensor,
    src_index: list[int] | torch.Tensor,
    group: dist.ProcessGroup,
    async_op: bool = False,
    cast_lse: bool = False,
    input_lse: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    **kwargs,
) -> WorkWithPostProcessFn:
    """Group-cast implementation based on all2all_v"""

    # ---------    check     --------- #
    # … 形状与类型断言省略

    # ---------    calc group cast a2a args     --------- #

    (
        a2a_output,
        a2a_input,
        a2a_output_split_size_list,
        a2a_input_split_size_list,
        post_process_fn,
    ) = calc_group_cast_a2a_args(
        input=input,
        output=output,
        input_split_size_list=input_split_sizes,
        output_split_size_list=output_split_sizes,
        dst_indices_list=dst_indices,
        src_index_list=src_index,
        world_size=dist.get_world_size(group),
        cast_lse=cast_lse,
        input_lse=input_lse,
        output_lse=output_lse,
        **kwargs,
    )

    # ---------    lauch a2a comm kernel     --------- #

    if cast_lse:
        # NOTE: we can not fuse lse comm with out comm based on nccl APIs
        # due to different shape and dtype
        # … 两次 all2all_v：out 一次、lse 一次
        work = [work_out, work_lse]
    else:
        work = all2all_v(
            input=a2a_input,
            output=a2a_output,
            input_split_size_list=a2a_input_split_size_list,
            output_split_size_list=a2a_output_split_size_list,
            group=group,
            async_op=async_op,
        )

    return WorkWithPostProcessFn(
        work=GeneralWork(work=work),
        post_process_fn=post_process_fn,
        async_op=async_op,
    )`
        },
        {
          id: "07",
          title: "归约进 kernel：out_acc / lse_acc 与 sm_margin",
          start: 1300,
          end: 1336,
          code: C`            else:
                partial_out, meta = _flex_flash_attn_forward(
                    q=q,
                    k=k,
                    v=v,
                    # NOTE: sink token needs to be applied only once
                    # thus we only apply it at the host stage if not skipped
                    sink=sink if is_host_stage else None,
                    sink_layout="sh",
                    out=out_acc,  # directly reduce to out_acc
                    lse=lse_acc,  # directly reduce to lse_acc
                    **attn_arg.to_ffa_args(is_bwd=False),
                    softmax_scale=softmax_scale,
                    softcap=softcap,
                    # NOTE: always use high-precision for the partial out,
                    # to reduce the error caused by the out/lse correction
                    out_type=self.hp_dtype,
                    # NOTE: when using accumulative buffer, we need to always enable atomic reduction
                    # unless it is the first call when accumulative buffer is still None
                    disable_fwd_atomic_reduction=(
                        attn_arg.disable_fwd_atomic_reduction and out_acc is None
                    ),
                    deterministic=self.deterministic,
                    sm_margin=self.fwd_sm_margin,
                    # optional args below mainly for sparse attn
                    # …
                    return_max_logits=return_max_logits,
                    max_logits=max_logits_acc,  # directly reduce to max_logits_acc
                )

        return partial_out, meta`
        },
        {
          id: "08",
          title: "OverlapConfig：degree 的四种语义",
          path: "magi_attention/meta/solver/overlap_solver.py",
          start: 71,
          end: 129,
          code: C`@dataclass(frozen=True)
class OverlapConfig:
    """The config dataclass for multi-stage overlapping.

    The ''degree'' parameter controls both the overlap behavior and the number
    of remote pipeline stages:

    - ''degree=0'': **no overlap** -- blocking communication + merged attn_arg,
      completely avoids LSE reduce precision loss.
    - ''degree=1'': local + 1 remote stage, no multi-stage chunking.
    - ''degree=N (N>=2)'': local + N remote stages (static multi-stage overlap).
    - ''degree=None'': dynamic mode -- the overlap solver automatically
      determines the optimal degree at runtime.
    """

    mode: AttnOverlapMode = AttnOverlapMode.STATIC

    degree: int | None = 1
    dynamic_max_degree: int | None = (
        8  # only used in dynamic mode, if None, then no limit
    )

    min_chunk_size: int = 512
    max_num_chunks: int = 64

    # TODO: use another non-trivial alg as default in the future
    alg: OverlapAlg = UniformOverlapAlg()

    calc_cost_factor: float = (
        1.0  # define: calc_cost = calc_cost_factor * calc_area (unit: μs)
    )
    comm_cost_factor: float = (
        1.0  # define: comm_cost = comm_cost_factor * comm_size (unit: μs)
    )
    # …

    def __post_init__(self):
        # DEVIATION: degree=0 is normalized to degree=1, max_num_chunks forced to 1
        # Reason: degree=0 is a user-facing shorthand for "no overlap" (blocking
        #   comm + merged attn_arg), but pipeline scheduling requires degree>=1.
        # Recovery: self._no_overlap / self.no_overlap preserves the original intent.
        object.__setattr__(self, "_no_overlap", self.degree == 0)
        # … 其余断言省略`
        },
        {
          id: "09",
          title: "成本模型：max(通信ᵢ, 计算ᵢ₋₁) 求和",
          path: "magi_attention/meta/solver/overlap_solver.py",
          start: 381,
          end: 433,
          code: C`    def _get_best_solution_from_dict(
        self,
        solution_dict: dict[int, OverlapSolution],
    ) -> OverlapSolution:
        return sorted(
            solution_dict.values(),
            # NOTE: the cmp key is bi-level:
            # 1. first level: minimize the overall cost (resolution as 1 μs)
            # 2. if the overall cost is approximately equal,
            #   then second level: minimize the overlap degree
            key=lambda sol: (round(sol.overall_cost), sol.overlap_degree),
        )[0]

    def _calc_overall_cost(
        self,
        stage_costs: list[OverlapStageCost],
        partitions: list[list[int]],
        overlap_degree: int,
    ) -> float:
        # HACK: for now, with the hypothesis that:
        # every internal comm/calc cost pair can be perfectly overlapped by the larger one
        # we just calc the overall cost as the sum of two parts:
        # 1. the sum of the maximum of ith comm cost and (i-1)th calc cost pair, for i in [0,1,...,overlap_degree-1]
        # 2. the last remote calc cost, with the index of -1

        overall_cost = 0.0
        for i in range(overlap_degree):
            if i == 0:
                # first remote comm cost overlapped with the host calc cost
                overall_cost += max(
                    # first remote comm cost
                    sum(stage_costs[idx].comm_cost for idx in partitions[0]),
                    # host calc cost
                    stage_costs[0].calc_cost,
                )
            else:  # ith remote comm cost overlapped with (i-1)th remote calc cost
                overall_cost += max(
                    # ith remote comm cost
                    sum(stage_costs[idx].comm_cost for idx in partitions[i]),
                    # (i-1)th remote calc cost
                    sum(
                        stage_costs[idx].calc_cost
                        for idx in partitions[i - 1]
                        if idx != 0
                    ),
                )

        # last remote calc cost
        overall_cost += sum(
            stage_costs[idx].calc_cost for idx in partitions[-1] if idx != 0
        )

        return overall_cost`
        }
      ]
    }
  };
})();

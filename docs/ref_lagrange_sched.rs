use dashmap::DashMap;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::mem;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use rayon::prelude::*;

// ============================================================
// 常数
// ============================================================
const INTERVAL_MS: u64 = 500;

const E_ALU: f64 = 1.0;
const E_LLC: f64 = 200.0;
const E_MISS: f64 = 6400.0;
const E_CTX: f64 = 1_000_000.0;

const ETA_LLC: f64 = 0.05;
const ETA_MEM: f64 = 0.05;
const ETA_TIER: f64 = 0.1;
const ETA_CPU: f64 = 0.1;

const TURBO_MAX: f64 = 9.0;
const C_CPU: f64 = 1.0 + TURBO_MAX;
const C_LLC: f64 = 0.05 * C_CPU;
const C_MEM: f64 = 0.10 * C_CPU;

const REF_MISS: f64 = 0.05;
const REF_L2_KB: f64 = 1024.0;

const TEMP: f64 = 0.1;

const CONGESTION_BETA: f64 = 1.0;
const V_FR_EWMA: f64 = 0.3;
const DZ_EWMA: f64 = 0.3;
const CONE_EWMA: f64 = 0.4;
const CONE_SCALE_MIN: f64 = 0.15;
const CONE_SCALE_MAX: f64 = 1.00;
const DZ_WEIGHT: f64 = 5.0;
const DN_WEIGHT: f64 = 1.0;

const FLOAT_EPS: f64 = 1e-12;

const NICE_MIN: i32 = -19;
const NICE_MAX: i32 = 19;

const RAYON_THREADS: usize = 8;

static PERF_EVENT_IOC_ENABLE: u64 = 0x2400;

// ============================================================
// perf_event 结构体
// ============================================================
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct perf_event_attr {
    type_: u32, size: u32, config: u64, sample_period: u64, sample_type: u64,
    read_format: u64, disabled: u64, inherit: u64, pinned: u64, exclusive: u64,
    exclude_user: u64, exclude_kernel: u64, exclude_hv: u64, exclude_idle: u64,
    mmap: u64, comm: u64, freq: u64, inherit_stat: u64, enable_on_exec: u64,
    task: u64, watermark: u64, precise_ip: u64, mmap_data: u64, sample_id_all: u64,
    exclude_host: u64, exclude_guest: u64, exclude_callchain_kernel: u64,
    exclude_callchain_user: u64, mmap2: u64, comm_exec: u64, use_clockid: u64,
    context_switch: u64, write_backward: u64, namespaces: u64, ksymbol: u64,
    bpf_event: u64, aux_output: u64, cgroup: u64, text_poke: u64, build_id: u64,
    __reserved_1: u64, __reserved_2: u64, __reserved_3: u64, __reserved_4: u64,
    __reserved_5: u64,
}

const PERF_TYPE_HARDWARE: u32 = 0;
const PERF_TYPE_SOFTWARE: u32 = 1;
const PERF_TYPE_RAW: u32 = 4;
const PERF_COUNT_HW_INSTRUCTIONS: u64 = 0;
const PERF_COUNT_HW_CPU_CYCLES: u64 = 1;
const PERF_COUNT_SW_CONTEXT_SWITCHES: u64 = 1;
const RAW_LL_CACHE_RD: u64 = 0x0036;
const RAW_LL_CACHE_MISS_RD: u64 = 0x0037;

fn perf_event_open(attr: &mut perf_event_attr, pid: i32, cpu: i32, group_fd: i32, flags: u32) -> i32 {
    unsafe { libc::syscall(libc::SYS_perf_event_open, attr, pid, cpu, group_fd, flags) as i32 }
}

// ============================================================
// Tier / Config
// ============================================================
#[derive(Clone, Debug)]
struct Tier {
    cores: u32,
    l2_kb: u32,
    l2_shared: bool,
    cap: f32,
}

#[derive(Clone)]
struct Config {
    pub tiers: Vec<Tier>,
    pub total_cores: u32,
    pub rho: Vec<f32>,
    pub affinity_enabled: bool,
    pub cpu_of_tier: Vec<Vec<usize>>,
}

impl Config {
    fn load() -> Self {
        let mut tier_cores: Vec<u32> = vec![3, 2, 2, 1];
        let mut tier_l2_kb: Vec<u32> = vec![512, 512, 512, 1024];
        let mut tier_l2_shared: Option<Vec<bool>> = Some(vec![true, true, true, false]);
        let mut tier_cap: Vec<f32> = vec![0.3, 0.7, 0.8, 1.0];
        let mut affinity_enabled = false;

        if let Ok(content) = fs::read_to_string("./config.prop") {
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with('#') || trimmed.is_empty() { continue; }
                if let Some(pos) = trimmed.find('=') {
                    let key = trimmed[..pos].trim().to_uppercase();
                    let value = trimmed[pos + 1..].trim();
                    match key.as_str() {
                        "TIER_CORES" => {
                            let v: Vec<u32> = value.split(',')
                                .filter_map(|x| x.trim().parse().ok()).collect();
                            if !v.is_empty() { tier_cores = v; }
                        }
                        "TIER_L2_KB" => {
                            let v: Vec<u32> = value.split(',')
                                .filter_map(|x| x.trim().parse().ok()).collect();
                            if !v.is_empty() { tier_l2_kb = v; }
                        }
                        "TIER_L2_SHARED" => {
                            let v: Vec<bool> = value.split(',')
                                .filter_map(|x| {
                                    let t = x.trim();
                                    if t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("yes") {
                                        Some(true)
                                    } else if t == "0" || t.eq_ignore_ascii_case("false") || t.eq_ignore_ascii_case("no") {
                                        Some(false)
                                    } else { None }
                                }).collect();
                            if !v.is_empty() { tier_l2_shared = Some(v); }
                        }
                        "TIER_CAP" => {
                            let v: Vec<f32> = value.split(',')
                                .filter_map(|x| x.trim().parse().ok()).collect();
                            if !v.is_empty() { tier_cap = v; }
                        }
                        "AFFINITY" => {
                            let t = value.trim();
                            affinity_enabled = t == "1"
                                || t.eq_ignore_ascii_case("true")
                                || t.eq_ignore_ascii_case("yes");
                        }
                        _ => {}
                    }
                }
            }
        }

        let k = tier_cores.len()
            .max(tier_l2_kb.len())
            .max(tier_cap.len())
            .max(tier_l2_shared.as_ref().map(|v| v.len()).unwrap_or(0));
        while tier_cores.len() < k { tier_cores.push(*tier_cores.last().unwrap_or(&1)); }
        while tier_l2_kb.len() < k { tier_l2_kb.push(*tier_l2_kb.last().unwrap_or(&1024)); }
        while tier_cap.len() < k { tier_cap.push(*tier_cap.last().unwrap_or(&1.0)); }

        let mut l2_shared_vec: Vec<bool> = tier_l2_shared.unwrap_or_else(|| {
            tier_cores.iter().map(|&c| c > 1).collect()
        });
        while l2_shared_vec.len() < k {
            l2_shared_vec.push(tier_cores[l2_shared_vec.len()] > 1);
        }

        let mut tiers = Vec::with_capacity(k);
        for i in 0..k {
            tiers.push(Tier {
                cores: tier_cores[i],
                l2_kb: tier_l2_kb[i],
                l2_shared: l2_shared_vec[i],
                cap: tier_cap[i],
            });
        }
        let total_cores: u32 = tiers.iter().map(|t| t.cores).sum();
        let rho: Vec<f32> = tiers.iter()
            .map(|t| t.cores as f32 / total_cores as f32)
            .collect();

        // 顺序映射：TIER_CORES 从左到右 = CPU0 到 CPU_last
        let mut cpu_of_tier: Vec<Vec<usize>> = Vec::with_capacity(k);
        let mut offset = 0usize;
        for t in &tiers {
            let cpus: Vec<usize> = (offset..offset + t.cores as usize).collect();
            cpu_of_tier.push(cpus);
            offset += t.cores as usize;
        }

        Self { tiers, total_cores, rho, affinity_enabled, cpu_of_tier }
    }

    fn num_tiers(&self) -> usize { self.tiers.len() }
    fn m(&self) -> usize { 3 + self.num_tiers() }

    /// 从 CPU 号推断它属于哪个 tier（基于顺序映射）
    fn tier_of_cpu(&self, cpu: usize) -> Option<usize> {
        for (k, cpus) in self.cpu_of_tier.iter().enumerate() {
            if cpus.contains(&cpu) {
                return Some(k);
            }
        }
        None
    }
}

// ============================================================
// NodeData
// ============================================================
struct NodeData {
    tid: i32,
    j: AtomicU64,
    c_llc: AtomicU64,
    m_mem: AtomicU64,
    tier: AtomicU32,
    insn_share: AtomicU64,
}

impl NodeData {
    fn new(tid: i32) -> Self {
        Self {
            tid,
            j: AtomicU64::new(0),
            c_llc: AtomicU64::new(0.001_f64.to_bits()),
            m_mem: AtomicU64::new(0.001_f64.to_bits()),
            tier: AtomicU32::new(0),
            insn_share: AtomicU64::new(0),
        }
    }

    fn read(&self) -> (f64, f64, f64, usize, f64) {
        (
            f64::from_bits(self.j.load(Ordering::Relaxed)),
            f64::from_bits(self.c_llc.load(Ordering::Relaxed)),
            f64::from_bits(self.m_mem.load(Ordering::Relaxed)),
            self.tier.load(Ordering::Relaxed) as usize,
            f64::from_bits(self.insn_share.load(Ordering::Relaxed)),
        )
    }
}

// ============================================================
// 共享状态
// ============================================================
struct State {
    lambda: Vec<AtomicU64>,
    gamma: AtomicU64,
    nodes: DashMap<i32, Arc<NodeData>>,
    cfg: Arc<Config>,
}

impl State {
    fn new(cfg: Arc<Config>) -> Self {
        let m = cfg.m();
        let lambda = (0..m).map(|_| AtomicU64::new(0)).collect();
        Self {
            lambda,
            gamma: AtomicU64::new(1.0_f64.to_bits()),
            nodes: DashMap::new(),
            cfg,
        }
    }

    fn read_lambda(&self, r: usize) -> f64 {
        f64::from_bits(self.lambda[r].load(Ordering::Relaxed))
    }
    fn write_lambda(&self, r: usize, v: f64) {
        self.lambda[r].store(v.to_bits(), Ordering::Release);
    }
    fn read_all_lambdas(&self) -> Vec<f64> {
        self.lambda.iter().map(|a| f64::from_bits(a.load(Ordering::Relaxed))).collect()
    }
    fn read_gamma(&self) -> f64 {
        f64::from_bits(self.gamma.load(Ordering::Acquire))
    }
    fn write_gamma(&self, v: f64) {
        self.gamma.store(v.to_bits(), Ordering::Release);
    }
}

// ============================================================
// 容量 / 步长
// ============================================================
fn capacity(r: usize, cfg: &Config) -> f64 {
    match r {
        0 => C_CPU,
        1 => C_LLC,
        2 => C_MEM,
        _ => cfg.rho[r - 3] as f64,
    }
}

fn eta_for(r: usize) -> f64 {
    match r {
        0 => ETA_CPU,
        1 => ETA_LLC,
        2 => ETA_MEM,
        _ => ETA_TIER,
    }
}

// ============================================================
// 单线程的 w / probs / best_tier
// ============================================================
fn compute_thread_w(
    lambdas: &[f64],
    c_llc: f64,
    m_mem: f64,
    insn_share: f64,
    cfg: &Config,
) -> (f64, Vec<f64>, usize) {
    let k = cfg.num_tiers();
    let mut pi_k = Vec::with_capacity(k);
    for kk in 0..k {
        let t = &cfg.tiers[kk];
        let cap_kb = (t.l2_kb as f64).max(64.0);
        let amount = insn_share * c_llc;
        let pressure = amount / cap_kb;
        let ref_pressure = REF_MISS / REF_L2_KB;
        let sens = (1.0 + pressure / ref_pressure).max(1.0);

        let pi = lambdas[0]
               + lambdas[1] * c_llc
               + lambdas[2] * m_mem
               + lambdas[3 + kk] * sens;
        pi_k.push(pi);
    }

    let min_pi = pi_k.iter().cloned().fold(f64::MAX, f64::min);
    let t_safe = TEMP.max(1e-6);
    let mut probs = Vec::with_capacity(k);
    let mut exp_sum = 0.0;
    let mut best_k = 0usize;
    let mut best_p = -1.0f64;
    for (kk, &pi) in pi_k.iter().enumerate() {
        let e = (-(pi - min_pi) / t_safe).exp();
        probs.push(e);
        exp_sum += e;
        if e > best_p {
            best_p = e;
            best_k = kk;
        }
    }
    for p in probs.iter_mut() { *p /= exp_sum; }

    let pi_lse = min_pi - t_safe * exp_sum.ln();
    let w = (-pi_lse).exp();

    (w, probs, best_k)
}

// ============================================================
// Affinity
// ============================================================
fn set_affinity(tid: i32, cpus: &[usize]) -> bool {
    if cpus.is_empty() { return false; }
    unsafe {
        let mut set: libc::cpu_set_t = mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &cpu in cpus {
            libc::CPU_SET(cpu, &mut set);
        }
        let ret = libc::sched_setaffinity(
            tid as libc::pid_t,
            mem::size_of::<libc::cpu_set_t>(),
            &set,
        );
        ret == 0
    }
}

// ============================================================
// 累加器
// ============================================================
struct Accum {
    s: Vec<f64>,
    w_list: Vec<(i32, f64)>,
    best_tiers: Vec<(i32, usize)>,
}

// ============================================================
// Rayon 并行计算
// ============================================================
fn compute_all_rayon(
    nodes: &DashMap<i32, Arc<NodeData>>,
    lambdas: &[f64],
    cfg: &Config,
) -> Accum {
    let k = cfg.num_tiers();
    let m = cfg.m();

    let thread_data: Vec<(i32, f64, f64, f64)> = nodes.iter()
        .map(|entry| {
            let tid = *entry.key();
            let (_, c, mm, _tier, insn) = entry.value().read();
            (tid, c, mm, insn)
        })
        .collect();

    thread_data
        .par_iter()
        .fold(
            || Accum {
                s: vec![0.0f64; m],
                w_list: Vec::new(),
                best_tiers: Vec::new(),
            },
            |mut acc, &(tid, c, mm, insn)| {
                let (w, probs, best_k) = compute_thread_w(lambdas, c, mm, insn, cfg);
                acc.s[0] += w;
                acc.s[1] += w * c;
                acc.s[2] += w * mm;
                for kk in 0..k {
                    acc.s[3 + kk] += insn * probs[kk];
                }
                acc.w_list.push((tid, w));
                acc.best_tiers.push((tid, best_k));
                acc
            },
        )
        .reduce(
            || Accum {
                s: vec![0.0f64; m],
                w_list: Vec::new(),
                best_tiers: Vec::new(),
            },
            |mut a, mut b| {
                for i in 0..m {
                    a.s[i] += b.s[i];
                }
                a.w_list.append(&mut b.w_list);
                a.best_tiers.append(&mut b.best_tiers);
                a
            },
        )
}

// ============================================================
// 测量线程
// ============================================================
struct PerfHandles {
    insn_fd: i32, cycles_fd: i32, llc_fd: i32, llc_miss_fd: i32, ctx_fd: i32,
    last_insn: u64, last_cycles: u64, last_llc: u64, last_miss: u64, last_ctx: u64,
}

fn measure_loop(state: Arc<State>) {
    let mut handles: HashMap<i32, PerfHandles> = HashMap::new();
    loop {
        let tids = get_all_tids();
        let tids_set: std::collections::HashSet<i32> = tids.iter().copied().collect();

        handles.retain(|tid, h| {
            let alive = tids_set.contains(tid);
            if !alive {
                unsafe {
                    if h.insn_fd >= 0 { libc::close(h.insn_fd); }
                    if h.cycles_fd >= 0 { libc::close(h.cycles_fd); }
                    if h.llc_fd >= 0 { libc::close(h.llc_fd); }
                    if h.llc_miss_fd >= 0 { libc::close(h.llc_miss_fd); }
                    if h.ctx_fd >= 0 { libc::close(h.ctx_fd); }
                }
            }
            alive
        });
        state.nodes.retain(|tid, _| tids_set.contains(tid));

        let mut measured: Vec<(i32, f64, f64, f64, u64, usize)> = Vec::new();
        let mut total_d_insn: u64 = 0;

        for &tid in &tids {
            let h = handles.entry(tid).or_insert_with(|| create_handles(tid));
            if h.insn_fd < 0 { continue; }
            let obs_tier = {
                let cpu = read_cpu(tid).unwrap_or(0);
                state.cfg.tier_of_cpu(cpu).unwrap_or(state.cfg.num_tiers() - 1)
            };
            let cap = state.cfg.tiers.get(obs_tier).map(|t| t.cap as f64).unwrap_or(1.0);
            if let Some((j, c_llc, m_mem, d_insn)) = measure_one(h, cap) {
                measured.push((tid, j, c_llc, m_mem, d_insn, obs_tier));
                total_d_insn = total_d_insn.saturating_add(d_insn);
            } else {
                if let Some(nd) = state.nodes.get(&tid) {
                    nd.insn_share.store(0, Ordering::Relaxed);
                }
            }
        }

        let inv_total = if total_d_insn > 0 { 1.0 / total_d_insn as f64 } else { 0.0 };
        for (tid, j, c_llc, m_mem, d_insn, tier) in measured {
            let insn_share = d_insn as f64 * inv_total;
            let nd = state.nodes.entry(tid).or_insert_with(|| Arc::new(NodeData::new(tid)));
            nd.j.store(j.to_bits(), Ordering::Relaxed);
            nd.c_llc.store(c_llc.to_bits(), Ordering::Relaxed);
            nd.m_mem.store(m_mem.to_bits(), Ordering::Relaxed);
            nd.tier.store(tier as u32, Ordering::Relaxed);
            nd.insn_share.store(insn_share.to_bits(), Ordering::Relaxed);
        }

        thread::sleep(Duration::from_millis(INTERVAL_MS));
    }
}

fn create_handles(tid: i32) -> PerfHandles {
    let mut h = PerfHandles {
        insn_fd: -1, cycles_fd: -1, llc_fd: -1, llc_miss_fd: -1, ctx_fd: -1,
        last_insn: 0, last_cycles: 0, last_llc: 0, last_miss: 0, last_ctx: 0,
    };
    h.insn_fd = create_hw(tid, PERF_COUNT_HW_INSTRUCTIONS);
    h.cycles_fd = create_hw(tid, PERF_COUNT_HW_CPU_CYCLES);
    let (lf, mf) = create_llc(tid);
    h.llc_fd = lf; h.llc_miss_fd = mf;
    h.ctx_fd = create_sw(tid, PERF_COUNT_SW_CONTEXT_SWITCHES);
    if h.insn_fd >= 0 { h.last_insn = read_u64(h.insn_fd).unwrap_or(0); }
    if h.cycles_fd >= 0 { h.last_cycles = read_u64(h.cycles_fd).unwrap_or(0); }
    if h.llc_fd >= 0 && h.llc_miss_fd >= 0 {
        if let Some((l, m)) = read_pair(h.llc_fd, h.llc_miss_fd) {
            h.last_llc = l; h.last_miss = m;
        }
    }
    if h.ctx_fd >= 0 { h.last_ctx = read_u64(h.ctx_fd).unwrap_or(0); }
    h
}

fn measure_one(h: &mut PerfHandles, cap: f64) -> Option<(f64, f64, f64, u64)> {
    let insn = read_u64(h.insn_fd)?;
    let cycles = read_u64(h.cycles_fd).unwrap_or(0);
    let (llc, miss) = read_pair(h.llc_fd, h.llc_miss_fd).unwrap_or((0, 0));
    let ctx = read_u64(h.ctx_fd).unwrap_or(0);

    let d_insn = insn.saturating_sub(h.last_insn);
    let d_cycles = cycles.saturating_sub(h.last_cycles);
    let d_llc = llc.saturating_sub(h.last_llc);
    let d_miss = miss.saturating_sub(h.last_miss);
    let d_ctx = ctx.saturating_sub(h.last_ctx);

    h.last_insn = insn; h.last_cycles = cycles;
    h.last_llc = llc; h.last_miss = miss; h.last_ctx = ctx;

    if d_insn == 0 { return None; }

    let j_raw = E_ALU * d_insn as f64
              + E_LLC * d_llc as f64
              + E_MISS * d_miss as f64
              + E_CTX * d_ctx as f64;

    let cap_eff = cap.max(0.05);
    let j = j_raw / cap_eff;

    let c_llc = if d_llc > 0 { (d_miss as f64 / d_llc as f64).min(1.0) } else { 0.001 };
    let cpi = if d_insn > 0 { (d_cycles as f64 / d_insn as f64).max(0.1) } else { 1.0 };
    let m_mem = (c_llc * cpi).max(0.001);

    Some((j, c_llc, m_mem, d_insn))
}

// ============================================================
// perf 辅助
// ============================================================
fn create_hw(tid: i32, config: u64) -> i32 {
    let mut attr: perf_event_attr = unsafe { mem::zeroed() };
    attr.size = mem::size_of::<perf_event_attr>() as u32;
    attr.type_ = PERF_TYPE_HARDWARE; attr.config = config; attr.exclude_hv = 1;
    let fd = perf_event_open(&mut attr, tid, -1, -1, 0);
    if fd >= 0 { unsafe { libc::ioctl(fd, PERF_EVENT_IOC_ENABLE as i32, 0); } return fd; }
    attr.exclude_kernel = 1;
    let fd2 = perf_event_open(&mut attr, tid, -1, -1, 0);
    if fd2 >= 0 { unsafe { libc::ioctl(fd2, PERF_EVENT_IOC_ENABLE as i32, 0); } return fd2; }
    -1
}

fn create_sw(tid: i32, config: u64) -> i32 {
    let mut attr: perf_event_attr = unsafe { mem::zeroed() };
    attr.size = mem::size_of::<perf_event_attr>() as u32;
    attr.type_ = PERF_TYPE_SOFTWARE; attr.config = config;
    attr.exclude_hv = 1; attr.exclude_kernel = 1;
    let fd = perf_event_open(&mut attr, tid, -1, -1, 0);
    if fd >= 0 { unsafe { libc::ioctl(fd, PERF_EVENT_IOC_ENABLE as i32, 0); } }
    fd
}

fn create_llc(tid: i32) -> (i32, i32) {
    let open_one = |config: u64, ex_k: u64| -> i32 {
        let mut attr: perf_event_attr = unsafe { mem::zeroed() };
        attr.size = mem::size_of::<perf_event_attr>() as u32;
        attr.type_ = PERF_TYPE_RAW; attr.config = config; attr.disabled = 1;
        attr.exclude_hv = 1; attr.exclude_kernel = ex_k;
        let fd = perf_event_open(&mut attr, tid, -1, -1, 0);
        if fd < 0 { return -1; }
        unsafe { if libc::ioctl(fd, PERF_EVENT_IOC_ENABLE as i32, 0) == -1 { libc::close(fd); return -1; } }
        fd
    };
    let fda = open_one(RAW_LL_CACHE_RD, 0);
    let fdm = open_one(RAW_LL_CACHE_MISS_RD, 0);
    if fda >= 0 && fdm >= 0 { return (fda, fdm); }
    if fda >= 0 { unsafe { libc::close(fda); } }
    if fdm >= 0 { unsafe { libc::close(fdm); } }
    let fda2 = open_one(RAW_LL_CACHE_RD, 1);
    let fdm2 = open_one(RAW_LL_CACHE_MISS_RD, 1);
    if fda2 >= 0 && fdm2 >= 0 { return (fda2, fdm2); }
    if fda2 >= 0 { unsafe { libc::close(fda2); } }
    if fdm2 >= 0 { unsafe { libc::close(fdm2); } }
    (-1, -1)
}

fn read_u64(fd: i32) -> Option<u64> {
    if fd < 0 { return None; }
    let mut val: u64 = 0;
    unsafe {
        let ret = libc::read(fd, &mut val as *mut u64 as *mut libc::c_void, 8);
        if ret == 8 { Some(val) } else { None }
    }
}

fn read_pair(fd_a: i32, fd_m: i32) -> Option<(u64, u64)> {
    if fd_a < 0 || fd_m < 0 { return None; }
    let mut a: u64 = 0; let mut m: u64 = 0;
    unsafe {
        if libc::read(fd_a, &mut a as *mut u64 as *mut libc::c_void, 8) != 8 { return None; }
        if libc::read(fd_m, &mut m as *mut u64 as *mut libc::c_void, 8) != 8 { return None; }
    }
    Some((a, m))
}

// ============================================================
// CPU / nice / 系统
// ============================================================
fn read_cpu(tid: i32) -> Option<usize> {
    let content = fs::read_to_string(format!("/proc/{}/stat", tid)).ok()?;
    let pos = content.rfind(')')?;
    let rest = &content[pos + 1..];
    let fields: Vec<&str> = rest.split_whitespace().collect();
    fields.get(36).and_then(|s| s.parse::<usize>().ok())
}

fn weight_to_nice_relative(w: f64, w_median: f64) -> i32 {
    if w <= 0.0 || w_median <= 0.0 { return 0; }
    let ratio = (w / w_median).max(1e-6);
    let nice_f = -(ratio.ln()) / (1.25_f64).ln();
    nice_f.round().clamp(NICE_MIN as f64, NICE_MAX as f64) as i32
}

fn set_nice(tid: i32, nice: i32) {
    unsafe { libc::setpriority(libc::PRIO_PROCESS, tid as u32, nice); }
}

fn get_all_tids() -> Vec<i32> {
    if let Ok(content) = fs::read_to_string("/dev/cpuset/top-app/tasks") {
        let tids: Vec<i32> = content.lines().filter_map(|l| l.trim().parse().ok()).collect();
        if !tids.is_empty() { return tids; }
    }
    let mut result = Vec::new();
    if let Ok(entries) = fs::read_dir("/proc") {
        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Ok(pid) = name.parse::<i32>() {
                if pid < 100 { continue; }
                if fs::metadata(format!("/proc/{}/stat", pid)).is_ok() { result.push(pid); }
            }
        }
    }
    result
}

// ============================================================
// Main
// ============================================================
fn main() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(RAYON_THREADS)
        .build_global()
        .unwrap();

    let cfg = Arc::new(Config::load());
    let m = cfg.m();

    println!("==== 完全 Rayon + 可选 Affinity（{}ms） ====", INTERVAL_MS);
    println!("[映射]  TIER_CORES 从左到右 = CPU0 到 CPU_last");
    for (i, t) in cfg.tiers.iter().enumerate() {
        println!("  Tier {}: {} 核 (CPU={:?}), L2={}KB {}, cap={:.2}, ρ={:.3}",
            i, t.cores, cfg.cpu_of_tier[i], t.l2_kb,
            if t.l2_shared { "[共享]" } else { "[独立]" },
            t.cap, cfg.rho[i]);
    }
    println!("[并行]  Rayon {} 线程", RAYON_THREADS);
    println!("[亲和]  AFFINITY = {}（config.prop）", cfg.affinity_enabled);
    std::io::stdout().flush().unwrap();

    let state = Arc::new(State::new(cfg.clone()));

    {
        let state_meas = state.clone();
        thread::spawn(move || measure_loop(state_meas));
    }

    let mut last_w_norm: HashMap<i32, f64> = HashMap::new();
    let mut last_s_llc: f64 = 0.0;
    let mut last_s_mem: f64 = 0.0;
    let mut last_n: usize = 0;
    let mut initialized: bool = false;

    let mut v_fr_ewma: f64 = 0.0;
    let mut dz_ewma: f64 = 0.0;
    let mut cone_ewma: f64 = 1.0;
    let mut cycle: u64 = 0;
    let mut affinity_failures: u64 = 0;

    loop {
        thread::sleep(Duration::from_millis(INTERVAL_MS));
        cycle += 1;

        let lambdas = state.read_all_lambdas();
        let acc = compute_all_rayon(&state.nodes, &lambdas, &cfg);

        let sum_w: f64 = acc.w_list.iter().map(|(_, w)| w).sum();
        if sum_w < FLOAT_EPS { continue; }

        // λ 更新
        {
            let gamma = state.read_gamma();
            let s = &acc.s;
            (0..m).into_par_iter().for_each(|r| {
                let cap = capacity(r, &cfg);
                let eta = eta_for(r);
                let old = state.read_lambda(r);
                let err = s[r] - cap;
                let new = (old + gamma * eta * err).max(0.0);
                state.write_lambda(r, new);
            });
        }

        // nice 映射
        {
            let mut ws: Vec<f64> = acc.w_list.iter().map(|(_, w)| w / sum_w).collect();
            ws.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let median = ws[ws.len() / 2];

            acc.w_list.par_iter().for_each(|(tid, w)| {
                let nice = weight_to_nice_relative(w / sum_w, median);
                set_nice(*tid, nice);
            });
        }

        // Affinity（可选）
        if cfg.affinity_enabled {
            let cpu_of_tier = &cfg.cpu_of_tier;
            let failures: u64 = acc.best_tiers.par_iter()
                .map(|(tid, best_k)| {
                    let cpus = &cpu_of_tier[*best_k];
                    if set_affinity(*tid, cpus) { 0 } else { 1 }
                })
                .sum();
            affinity_failures = failures;
        }

        // γ 更新
        let mut current_w_norm: HashMap<i32, f64> = HashMap::with_capacity(acc.w_list.len());
        for (tid, w) in &acc.w_list {
            current_w_norm.insert(*tid, w / sum_w);
        }

        let mut v_fr_sq = 0.0f64;
        for (tid, w_new) in &current_w_norm {
            if let Some(w_old) = last_w_norm.get(tid) {
                let dw = w_new - w_old;
                let w_avg = ((w_new + w_old) / 2.0).max(FLOAT_EPS);
                v_fr_sq += dw * dw / w_avg;
            }
        }
        let v_fr = v_fr_sq.sqrt();

        let s_llc = acc.s[1];
        let s_mem = acc.s[2];
        let n_now = current_w_norm.len();
        let dn_rate = if initialized && last_n > 0 {
            (n_now as f64 - last_n as f64).abs() / last_n as f64
        } else { 0.0 };

        let dz = if initialized {
            (s_llc - last_s_llc).abs()
            + (s_mem - last_s_mem).abs()
            + DN_WEIGHT * dn_rate
        } else { 0.0 };

        last_s_llc = s_llc;
        last_s_mem = s_mem;
        last_n = n_now;
        initialized = true;

        v_fr_ewma = (1.0 - V_FR_EWMA) * v_fr_ewma + V_FR_EWMA * v_fr;
        dz_ewma = (1.0 - DZ_EWMA) * dz_ewma + DZ_EWMA * dz;

        let mut max_rel_err = 0.0;
        for r in 0..m {
            let cap_r = capacity(r, &cfg).max(FLOAT_EPS);
            let rel = (acc.s[r] - cap_r).abs() / cap_r;
            if rel > max_rel_err { max_rel_err = rel; }
        }

        let signal = v_fr_ewma.max(dz_ewma * DZ_WEIGHT);
        let cone_base = (1.0 / (1.0 + CONGESTION_BETA * signal))
            .clamp(CONE_SCALE_MIN, CONE_SCALE_MAX);
        cone_ewma = (1.0 - CONE_EWMA) * cone_ewma + CONE_EWMA * cone_base;

        let boost = 1.0 + max_rel_err;
        let gamma = cone_ewma * boost;
        state.write_gamma(gamma);

        last_w_norm = current_w_norm;

        if cycle % 10 == 0 {
            let lambda_str: Vec<String> = lambdas.iter()
                .map(|v| format!("{:.2}", v)).collect();

            let mut tier_counts = vec![0usize; cfg.num_tiers()];
            for (_, best_k) in &acc.best_tiers {
                tier_counts[*best_k] += 1;
            }
            let tier_str: Vec<String> = tier_counts.iter()
                .map(|c| format!("{}", c)).collect();

            let phase = if sum_w > 1.0 { "超" } else { "内" };
            let aff = if cfg.affinity_enabled {
                format!("aff_fail={}", affinity_failures)
            } else {
                "aff=off".to_string()
            };
            println!(
                "[{}] N={} | λ=[{}] | tgt=[{}] | Σw={:.2}[{}] | γ={:.2} | err={:.2} | {}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                acc.w_list.len(),
                lambda_str.join(","),
                tier_str.join(","),
                sum_w, phase, gamma, max_rel_err, aff,
            );
            std::io::stdout().flush().unwrap();
        }
    }
}
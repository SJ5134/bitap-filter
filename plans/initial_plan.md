# Initial Plan: Bitap Edit-Distance Filter IP for DNA Read Pairs

## Project Team
- Siddh Javeri (sj5134@nyu.edu)
- Srijan Gupta (sg9109@nyu.edu)

## IP Definition

### What the IP does
The IP takes a batch of DNA sequence pairs, each a short **read** (up to 128 bases) and a
**reference window** (up to 128 bases). For every pair it computes the minimum number of
edits (substitutions, insertions, deletions) needed to match the *whole* read against *some*
substring of the reference window, capped at a threshold `K <= 7`. It returns that distance,
a pass/fail flag (`dist <= K`) and the start position of the best match.

This is the **pre-alignment filter** step of DNA read mapping. Each read has many candidate
locations in the genome and most of them are wrong. Rejecting pairs with more than `K` edits
before the expensive dynamic-programming alignment step saves most of that work. The filter
runs on every (read, candidate location) pair, which means millions to billions of pairs per
sample, so per-pair throughput is what matters.

### Interaction with the PS
The IP follows the course's command-response pattern for kernels that work on shared memory.
The data stays in DDR; only short messages travel on streams.

1. The PS packs the pairs into DDR. Each pair is a 64-byte record of sixteen 32-bit words.
   Words 0-7 hold the read and words 8-15 the reference, 2 bits per base
   (`A=00, C=01, G=10, T=11`). Base `j` of each sequence sits in word `j >> 4` of its
   eight-word half, at bits `[2(j&15)+1 : 2(j&15)]`.
2. The PS sends one **command** per batch on an AXI4-Stream, `s_cmd`:
   `BitapCmd {job_id, pair_addr, res_addr, num_pairs, m, n, K}`. The addresses are word
   offsets in DDR, and `m`, `n`, `K` are fixed for the batch.
3. The IP reads the records itself, filters every pair, and writes one 32-bit result word
   per pair, in pair order, starting at `res_addr`:

   | Bits | Field |
   | --- | --- |
   | `[3:0]` | `dist`: 0..K, or K+1 if there is no match with <= K edits |
   | `[4]` | `pass` = (`dist <= K`) |
   | `[14:8]` | `start`: leftmost reference index of a best match (valid when `pass`) |

4. Once the last result word has been acknowledged by memory, the IP sends a **response**
   on a second AXI4-Stream, `s_done`: `BitapResp {job_id, status}`. `status` is `OK`, or
   `BAD_PARAMS` if `m` or `n` is outside 1..128 or `K > 7`. A refused job reads and writes
   nothing but still gets exactly one response.

The kernel is free-running (`ap_ctrl_none`), and the PS can queue several commands. On the
board, the PS reaches `s_cmd` and `s_done` through memory-mapped command and response queues,
as in the course's histogram demo.

### Mathematical operations
The IP implements the **Bitap (Wu-Manber) bit-parallel approximate string matching**
algorithm. It keeps one `m`-bit state vector `R[d]` per edit count `d = 0..K` and updates
all of them once per reference base, using only shifts, AND and OR. The reference model
below is also the golden model. It matches edlib's infix (`HW`) edit distance and its
leftmost start position exactly on 30,000 random test pairs.

```python
BASES = "ACGT"                                  # 2-bit codes 0..3

def bitap_filter(read: str, ref: str, K: int):
    """Min edits to match all of `read` against any substring of `ref` (capped at K+1)."""
    m, n = len(read), len(ref)                  # 1 <= m, n <= 128, 0 <= K <= 7
    ONES = (1 << m) - 1
    # Pattern masks: read[0] is bit m-1 (MSB); a 0 bit marks a match.
    PM = {c: ONES for c in BASES}
    for j, c in enumerate(read):
        PM[c] &= ~(1 << (m - 1 - j)) & ONES
    # R[d] bit b == 0  <=>  the last b+1 read bases match starting at ref[i] with <= d edits.
    R = [(ONES << d) & ONES for d in range(K + 1)]  # last d read bases may be inserted
    dist, start = K + 1, -1
    for i in range(n - 1, -1, -1):              # scan the reference right to left
        old, pm = R[:], PM[ref[i]]
        R[0] = ((old[0] << 1) | pm) & ONES      # exact match
        for d in range(1, K + 1):
            R[d] = (old[d-1]                    # deletion: ref base unmatched
                    & (old[d-1] << 1)           # substitution
                    & (R[d-1] << 1)             # insertion: read base unmatched
                    & ((old[d] << 1) | pm)      # match, no new edit
                   ) & ONES
        # smallest d whose MSB is 0: whole read matches starting at ref[i]
        d_i = next((d for d in range(K + 1) if not (R[d] >> (m - 1)) & 1), K + 1)
        if d_i <= K and d_i <= dist:            # ties go to smaller i (leftmost start)
            dist, start = d_i, i
    return dist, dist <= K, start
```

Target sizes: `m, n <= 128` (128-bit state vectors), `K <= 7` (8 distances), and `L = 4`
parallel lanes (a compile-time parameter), with a 100 MHz clock on the team's Zynq FPGA board.

### Why it suits hardware acceleration
- **Bit-level parallelism.** Every operation is a shift, AND or OR on an `m`-bit word. For
  `d >= 1`, each bit of `R[d]` depends on exactly five bits (`old[d-1][b]`, `old[d-1][b-1]`,
  `R[d-1][b-1]`, `old[d][b-1]`, `pm[b]`), so it costs one 6-input LUT, and a full 128-bit
  update finishes in one cycle. On the PS's ARM cores, a 128-bit vector spans several
  machine words (four on a 32-bit core) with carries between them on every shift, and the
  loop over `d` runs sequentially.
- **Parallel in two dimensions.** Across distances, the dependency `R[d-1] -> R[d]` can be
  skewed into a wavefront: PE `d` works on base `i` while PE `d-1` works on the next one.
  Across pairs, every pair is independent, so lanes replicate freely.
- **Regular, fixed control.** Each pair takes exactly `n` iterations with no data-dependent
  branches, so the schedule pipelines at one reference base per cycle.
- **Tiny state, local compute.** A few kilobits of registers per lane. The only memory
  traffic is 64 bytes in and 4 bytes out per pair. With 32-bit memory words, the read
  mover delivers one pair every 16 cycles, while a lane needs about `n` (~100) cycles per
  pair, so a handful of lanes balances compute against memory bandwidth.

## IP Architecture

### Sub-modules
The kernel is a line of concurrent `hls::task`s joined by FIFOs: load → compute → store, as
in the Unit 10 reference design. Every task fires once per job, so the next batch's read
starts while the current batch is still computing and writing. Tasks 4-7 are replicated `L`
times (one lane each), and task 6 contains `K_MAX+1 = 8` identical PEs.

1. **Control.** Reads `BitapCmd` and checks the parameters. It builds a job descriptor
   `BitapDesc {job_id, status, num_pairs, m, n, K, res_addr}`, with `num_pairs = 0` for a
   refused job, and sends the read mover a memory command for `16 * num_pairs` words at
   `pair_addr`, with the descriptor framed behind it. This is the only task that sees the
   command.
2. **Read mover** (Waveflow's `mem_r_stream_framed_task`, reused). Turns the memory command
   into AXI4 bursts and streams out the descriptor followed by the record words.
3. **Pair Dispatcher.** Reads the descriptor and assembles each 16-word record into a 256-bit
   read and a 256-bit reference. It deals pairs round-robin to the `L` lanes (pair `p` to
   lane `p mod L`, so lane `l` gets `lane_pairs = (num_pairs + L - 1 - l) / L` pairs). It
   sends each lane a lane header `{m, n, K, lane_pairs}` in front of its pairs, and forwards
   the descriptor to the Result Collector.
4. **Mask Generator** (per lane). Turns each packed read into the four 128-bit pattern masks
   `PM[A], PM[C], PM[G], PM[T]` (bit `m-1-j` cleared in `PM[read[j]]`; bits at or above `m`
   are zero). Runs once per pair.
5. **Step Sequencer** (per lane). Walks the reference from `i = n-1` down to `0` and emits
   one step token per base: the selected mask `PM[ref[i]]`, the index `i`, and `first`/`last`
   flags marking pair boundaries. One token per cycle.
6. **Bitap Array** (per lane; a linear systolic array of 8 PEs). PE `d` holds `R[d]`. Each
   cycle it receives a step token plus `R[d-1]` for the current base and for the previous
   base from PE `d-1`, and computes the recurrence above. On a `first` token it uses the
   initial values `ONES << d` in place of the previous-base vectors. It tests bit `m-1` and,
   if this is the first PE with a hit for this base and `d <= K`, records `d` in the token's
   `found` field. It then forwards the token and its vectors to PE `d+1` one cycle later.
   Because of the skew, all PEs are busy on different bases, so throughput is one base per
   cycle per lane for any `K`, and a pair takes `n` cycles.
7. **Match Tracker** (per lane). Receives one `(i, d_i)` per base from the end of the array.
   It reduces them to `(dist, start)` using the golden model's rule (smallest `d`, ties to
   smaller `i`), and at the pair's `last` token emits the 32-bit result word.
8. **Result Collector.** Reads the descriptor, then the result words from the lanes in the
   order the dispatcher dealt them out (pair `p` from lane `p mod L`), so results stay in
   pair order with no reorder buffer. It frames a memory write command (`num_pairs` words at
   `res_addr`), the descriptor and the result words for the write mover.
9. **Write mover** (Waveflow's `mem_w_stream_framed_done_task`, reused). Writes the results
   in AXI4 bursts and passes the descriptor on only after memory has acknowledged the last
   word.
10. **Response.** Turns the echoed descriptor into `BitapResp {job_id, status}` on `s_done`.

### Interfaces

| From → To | Interface | What it carries |
| --- | --- | --- |
| PS → Control | AXI4-Stream (`s_cmd`) | `BitapCmd {job_id, pair_addr, res_addr, num_pairs, m, n, K}`, one per batch |
| Control → Read mover | Framed FIFO | Memory read command `{pair_addr, 16·num_pairs}`, then `BitapDesc` |
| DDR → Read mover | Shared memory, AXI4 master (`m_axi`, bundle `gmem0`, bursts) | The `num_pairs` 64-byte pair records |
| Read mover → Pair Dispatcher | Framed FIFO | `BitapDesc`, then `16·num_pairs` record words |
| Pair Dispatcher → Mask Generator[l] | FIFO (`hls::stream`) | Lane header, then one 256-bit packed read per pair |
| Pair Dispatcher → Step Sequencer[l] | FIFO (`hls::stream`) | Lane header, then one 256-bit packed reference per pair |
| Pair Dispatcher → Result Collector | FIFO (`hls::stream`) | `BitapDesc`, one per job, so the collector can frame the write |
| Mask Generator → Step Sequencer | FIFO (`hls::stream`) | One mask table per pair: 4 × 128 bits |
| Step Sequencer → Bitap Array | FIFO (`hls::stream`) | `n` step tokens per pair: `{pm[127:0], i[6:0], first, last}` |
| PE `d` → PE `d+1` | Registered point-to-point (systolic) | Step token with `found[3:0]`, `R[d]` for the current base, `R[d]` for the previous base |
| Bitap Array → Match Tracker | FIFO (`hls::stream`) | One `{i[6:0], d_i[3:0], last}` per reference base |
| Match Tracker → Result Collector | FIFO (`hls::stream`) | One 32-bit result word per pair |
| Result Collector → Write mover | Framed FIFO | Memory write command `{res_addr, num_pairs}`, `BitapDesc`, then the result words |
| Write mover → DDR | Shared memory, AXI4 master (`m_axi`, bundle `gmem1`, bursts) | `num_pairs` × 32-bit result words |
| Write mover → Response | FIFO (`hls::stream`) | `BitapDesc`, echoed after the last write acknowledgement |
| Response → PS | AXI4-Stream (`s_done`) | `BitapResp {job_id, status}`, one per command |

All messages are declared once as Waveflow schemas in Python, and the C++ structs and
serializers are generated, so the testbench, golden model and kernel share one layout. The
memory side uses 32-bit words, and addresses are word offsets.

| Message | Fields (bits) |
| --- | --- |
| `BitapCmd` | `job_id` 32, `pair_addr` 32, `res_addr` 32, `num_pairs` 32, `m` 8, `n` 8, `K` 4 |
| `BitapDesc` | `job_id` 32, `status` 1, `num_pairs` 32, `m` 8, `n` 8, `K` 4, `res_addr` 32 |
| `LaneHdr` | `m` 8, `n` 8, `K` 4, `lane_pairs` 32 |
| `BitapResp` | `job_id` 32, `status` 1 (`OK` = 0, `BAD_PARAMS` = 1) |
| Step token | `pm` 128, `i` 7, `first` 1, `last` 1 |
| Tracker input | `i` 7, `d_i` 4, `last` 1 |
| Result word | 32, laid out as in the result table above |

On-chip storage is small: per lane, one mask table (512 bits) and 8 PEs × (two state
vectors plus the forwarded mask) × 128-bit registers, plus FIFO buffers. The descriptor FIFOs
are sized to hold two whole headers so consecutive jobs can overlap.

### Why this decomposition
- **Load, compute and store are separate tasks**, so memory reads, filtering and memory
  writes of consecutive batches overlap, and throughput is set by the slowest stage.
- **The compute is split by the shape of the work:** a once-per-pair setup (mask
  generation), a per-base stream (sequencer and array) and a per-pair reduction (tracker).
  Each stage pipelines at its own natural rate.
- **Each module matches one piece of the golden model**, so it can be verified alone against
  Python-generated vectors before integration:
  - Mask Generator ↔ the `PM` dictionary
  - Step Sequencer ↔ the loop order and `pm` selection
  - one PE, and then the array ↔ per-base `R[d]` values dumped from the model
  - Match Tracker ↔ `dist` / `start`
  - Dispatcher and Collector ↔ the packing format and pair order
- **Build order:** golden model → PE → array → one complete lane → control, movers and
  response with `L = 1` → `L` lanes.
- **Parallelism is exposed directly:** PEs across `d`, lanes across pairs, and tasks across
  jobs.

### Alternatives considered
- **Distance computation.** (a) All `K+1` distances in one cycle per base: simplest, but
  `R[d]` waits on `R[d-1]` in the same cycle, which puts about `K+1` LUT levels on a
  loop-carried path, so II or clock rate degrades as `K` grows. (b) One PE looping over `d`:
  `K+1` cycles per base. (c) **Systolic wavefront (chosen):** one LUT level per PE and one
  base per cycle for any `K`. Option (a) will be kept as a measured comparison point.
- **PS ↔ IP control.** (a) AXI4-Lite job registers with `ap_start`/`ap_done`: simple, but
  one batch at a time with no overlap between batches. (b) Streaming the pairs in through an
  AXI DMA: moves the data well, but the host must manage DMA buffers in both directions.
  (c) **Command-response streams plus memory movers (chosen):** one short command per batch,
  the IP fetches its own data, and consecutive batches overlap.

### Existing IP
- The memory side reuses Waveflow's framed read and write movers (the course's data-mover
  library) in place of hand-written AXI4 master logic.
- No AMD IP performs bit-parallel approximate matching. The closest is the
  [AMD Vitis Genomics Library](https://xilinx.github.io/Vitis_Libraries/genomics/2021.2/index.html)
  Smith-Waterman kernel. It solves a different problem (scored local alignment by dynamic
  programming) and targets Alveo data-center cards at about 172K LUTs. It could serve as a
  reference point, but not as a building block.

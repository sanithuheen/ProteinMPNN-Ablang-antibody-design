"""
ProteinMPNN + AbLang ensemble for de novo CDR design (del Alamo et al. 2025 method).

This reproduces Fig. S1 of del Alamo et al.: for each CDR residue, in a random
order, ProteinMPNN's structural logits and AbLang's antibody-language-model
logits are summed BEFORE softmax, then sampled with Boltzmann temperature 0.1.
Residues in all designed CDRs (heavy AND light chain together) share one
random decoding order, matching "All residues in all CDRs were designed
simultaneously" in the paper's Methods.

Requirements (install in your own environment, not this sandbox):
    pip install ablang torch numpy
    git clone https://github.com/dauparas/ProteinMPNN.git
    # then put the ProteinMPNN/ folder on your PYTHONPATH, e.g.:
    #   export PYTHONPATH=$PYTHONPATH:/path/to/ProteinMPNN

Everything below was checked against the real source of both packages
(ablang/pretrained.py, ablang/tokenizers.py, ablang/model.py, and
ProteinMPNN/protein_mpnn_utils.py / protein_mpnn_run.py) rather than assumed.

Bennett et al.'s two publicly deposited structures are NOT a conventional
two-chain Fab, so a plain "chain H = heavy, chain L = light" mapping does not
work for either of them (verified against the real RCSB/SAbDab records):

  * 9NH7 (VHH_flu_01, influenza HA complex): the antibody is a VHH/nanobody
    -- ONE physical chain, ONE Ig domain (heavy-only, no light chain at all).
    Real author chain IDs: antibody = E (or F for the 2nd copy in the HA
    trimer), antigen = HA1 chain B + HA2 chain H (or A/G for the 2nd copy).
  * 9NFU (scFv6, TcdB complex): the antibody is an scFv -- ONE physical PDB
    chain (author chain C) that internally contains VH + linker + VL fused
    together. Real author chain IDs: antibody = C, antigen (Toxin B) = A.

Because of this, this script separates two different ideas that used to be
conflated:
  - "physical PDB chain"      -> used for ProteinMPNN structure/featurization
  - "Ig domain" (H or L)      -> used to pick which AbLang model to query,
                                  and is NOT always the same thing as a chain

`domain_layout` below tells the script how each physical chain's residues
split into Ig domains (for a VHH: the whole chain is domain "H"; for an scFv:
a "H" segment, then a non-Ig linker segment, then an "L" segment).

Two things you must still supply/confirm before trusting results:
  1. That `ablang.pretrained()` can reach
     https://opig.stats.ox.ac.uk/data/downloads/ablang-<chain>.tar.gz
     to download weights (network-restricted sandboxes will fail here).
  2. The exact CDR residue indices (for masking) and, for the scFv, the exact
     VH/linker/VL split point -- both come from your own AHo/Chothia/ANARCI
     numbering step; this script does not derive either on its own. A rough
     linker-finding heuristic is provided as a *convenience default only*.
"""

import re
import numpy as np
import torch
import torch.nn.functional as F

# --- ProteinMPNN imports (requires ProteinMPNN repo on PYTHONPATH) ---
from protein_mpnn_utils import (
    ProteinMPNN, parse_PDB, tied_featurize,
    gather_nodes, cat_neighbors_nodes,
)

# --- AbLang import (pip install ablang) ---
import ablang

PMPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"   # verified: protein_mpnn_utils.py line 50/193
PMPNN_AA20 = PMPNN_ALPHABET[:20]           # drop the trailing 'X' (unknown) for the ensemble sum


# ---------------------------------------------------------------------------
# 1. AbLang wrapper -- loads REAL weights, no mocking.
# ---------------------------------------------------------------------------
class AbLangEnsembleHelper:
    """
    Loads the real heavy- and/or light-chain AbLang models and returns, for a
    given masked sequence and position, a length-20 logit vector reordered
    into ProteinMPNN's amino-acid order (PMPNN_AA20) so the two logit
    vectors can be summed directly.

    `domains`: which AbLang model(s) to load. A VHH construct (e.g. 9NH7) has
    no light chain at all, so pass domains=("H",) to skip downloading/loading
    the light model entirely.
    """

    _ABLANG_CHAIN_NAME = {"H": "heavy", "L": "light"}

    def __init__(self, device="cpu", domains=("H", "L")):
        self.device = device
        # ablang.pretrained() downloads+caches weights on first use
        # (see ablang/pretrained.py: model_folder="download").
        self.models = {
            d: ablang.pretrained(chain=self._ABLANG_CHAIN_NAME[d], device=device)
            for d in domains
        }
        for m in self.models.values():
            m.freeze()

        # Build, once per chain model, the mapping from AbLang's own vocab
        # order to ProteinMPNN's PMPNN_AA20 order. AbLang's likelihood()
        # output columns are vocab indices 1..20 (verified in
        # ablang/pretrained.py: `predictions = self.AbLang(tokens)[:,:,1:21]`).
        # tokenizer.vocab_to_aa maps a vocab index -> amino-acid letter, so
        # column j of the likelihood output corresponds to
        # tokenizer.vocab_to_aa[j + 1].
        self.reorder_idx = {}
        for chain, m in self.models.items():
            vocab_to_aa = m.tokenizer.vocab_to_aa
            col_aa = [vocab_to_aa[j + 1] for j in range(20)]  # AbLang's own column order
            self.reorder_idx[chain] = [col_aa.index(aa) for aa in PMPNN_AA20]

    def logits_at_position(self, masked_seq: str, domain: str, pos0: int) -> np.ndarray:
        """
        masked_seq: the Ig DOMAIN's own sequence only (e.g. just the VH
                    portion of an scFv, not the whole physical PDB chain),
                    '*' at every not-yet-decided residue (including the
                    target position itself). '*' is AbLang's real mask
                    character (confirmed in ablang/tokenizers.py error msg).
        domain: 'H' or 'L' -- which AbLang model to query.
        pos0: 0-indexed position within masked_seq (i.e. within that domain's
              own sequence) to read logits for.
        Returns: length-20 raw (pre-softmax) logits in PMPNN_AA20 order.
        """
        model = self.models[domain]
        # likelihood() returns raw AbHead logits (no softmax -- confirmed in
        # ablang/model.py: AbHead.forward has no activation on its output),
        # shape (1, len(seq), 20), columns in AbLang's own vocab order.
        raw = model(masked_seq, mode="likelihood")[0, pos0, :]   # (20,)
        return raw[self.reorder_idx[domain]]


# ---------------------------------------------------------------------------
# 2. ProteinMPNN loading -- matches protein_mpnn_run.py exactly.
# ---------------------------------------------------------------------------
def load_proteinmpnn(checkpoint_path: str, device: str = "cpu") -> ProteinMPNN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ProteinMPNN(
        ca_only=False,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        augment_eps=0.0,                      # no backbone noise at inference
        k_neighbors=checkpoint["num_edges"],
    )
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 3. The ensemble decode loop -- adapted line-for-line from
#    ProteinMPNN.sample() (protein_mpnn_utils.py, lines ~1104-1188), with the
#    softmax/sample block replaced to inject AbLang logits first. This is
#    the "bespoke version of ProteinMPNN" del Alamo et al. describe in their
#    Methods; the stock sample() has no hook for external logits.
# ---------------------------------------------------------------------------
@torch.no_grad()
def ensemble_design(
    pmpnn_model: ProteinMPNN,
    ablang_helper: AbLangEnsembleHelper,
    X, S_true, chain_M, chain_M_pos, mask, residue_idx, chain_encoding_all,
    domain_of_position,          # list, len L: 'H'/'L' for designed positions, None otherwise
    domain_local_index,          # list, len L: 0-indexed position within that DOMAIN's own sequence
    chain_seqs,                  # dict: 'H' -> current heavy-domain seq (str, '*' for undecided); 'L' only if present
    temperature: float = 0.1,
    seed: int | None = None,
):
    """
    Runs one design trajectory. Returns the final PMPNN token sequence S
    (LongTensor, shape [1, L]) and the updated per-chain sequence strings.
    Only positions with chain_M*chain_M_pos*mask == 1 are (re)designed; all
    others keep their S_true identity, exactly as in the stock sample().
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = X.device
    B, L = S_true.shape
    assert B == 1, "written for a single structure at a time"

    randn = torch.randn(chain_M.shape, device=device)

    # --- Encoder pass (identical to sample(), lines 1106-1115) ---
    E, E_idx = pmpnn_model.features(X, mask, residue_idx, chain_encoding_all)
    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=device)
    h_E = pmpnn_model.W_e(E)
    mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    for layer in pmpnn_model.encoder_layers:
        h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

    # --- Decoding order across ALL designed positions (both chains together,
    #     one shared random order -- this is what makes heavy+light CDRs get
    #     designed "simultaneously" rather than chain-by-chain) ---
    chain_mask_combined = chain_M * chain_M_pos * mask
    decoding_order = torch.argsort((chain_mask_combined + 0.0001) * torch.abs(randn))

    mask_size = E_idx.shape[1]
    permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=mask_size).float()
    order_mask_backward = torch.einsum(
        "ij, biq, bjp->bqp",
        (1 - torch.triu(torch.ones(mask_size, mask_size, device=device))),
        permutation_matrix_reverse, permutation_matrix_reverse,
    )
    mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
    mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1.0 - mask_attend)

    h_S = torch.zeros_like(h_V)
    S = S_true.clone()
    h_V_stack = [h_V] + [torch.zeros_like(h_V) for _ in pmpnn_model.decoder_layers]

    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
    h_EXV_encoder_fw = mask_fw * h_EXV_encoder

    aa_to_pmpnn_idx = {aa: i for i, aa in enumerate(PMPNN_ALPHABET)}

    for t_ in range(L):
        t = decoding_order[:, t_]           # [B]
        pos = int(t.item())
        is_designed = bool(chain_mask_combined[0, pos].item() > 0)

        if not is_designed:
            S_t = S_true[:, pos:pos + 1]
        else:
            # --- ProteinMPNN structural logits at this position (decoder
            #     step, identical math to sample() lines 1152-1165) ---
            E_idx_t = torch.gather(E_idx, 1, t[:, None, None].repeat(1, 1, E_idx.shape[-1]))
            h_E_t = torch.gather(h_E, 1, t[:, None, None, None].repeat(1, 1, h_E.shape[-2], h_E.shape[-1]))
            h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
            h_EXV_encoder_t = torch.gather(
                h_EXV_encoder_fw, 1, t[:, None, None, None].repeat(1, 1, h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1])
            )
            mask_t = torch.gather(mask, 1, t[:, None])
            for l, layer in enumerate(pmpnn_model.decoder_layers):
                h_ESV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                h_V_t = torch.gather(h_V_stack[l], 1, t[:, None, None].repeat(1, 1, h_V_stack[l].shape[-1]))
                h_ESV_t = torch.gather(
                    mask_bw, 1, t[:, None, None, None].repeat(1, 1, mask_bw.shape[-2], mask_bw.shape[-1])
                ) * h_ESV_decoder_t + h_EXV_encoder_t
                h_V_stack[l + 1].scatter_(1, t[:, None, None].repeat(1, 1, h_V.shape[-1]), layer(h_V_t, h_ESV_t, mask_V=mask_t))
            h_V_t = torch.gather(h_V_stack[-1], 1, t[:, None, None].repeat(1, 1, h_V_stack[-1].shape[-1]))[:, 0]
            pmpnn_logits = pmpnn_model.W_out(h_V_t)[0]          # (21,) raw, PRE-softmax, PMPNN_ALPHABET order

            # --- AbLang logits at this position (real weights, real vocab) ---
            domain = domain_of_position[pos]
            assert domain is not None, (
                f"position {pos} is marked for design (chain_M_pos=1) but has "
                f"no Ig domain assigned -- check domain_layout for this chain "
                f"(this can happen if a CDR index accidentally falls inside a "
                f"non-Ig segment such as an scFv linker)."
            )
            pos0 = domain_local_index[pos]
            ablang_logits20 = ablang_helper.logits_at_position(chain_seqs[domain], domain, pos0)  # (20,) PMPNN_AA20 order

            # --- THE ENSEMBLE STEP (Fig. S1): sum logits, then softmax at T=0.1 ---
            combined = pmpnn_logits.clone()
            combined[:20] = combined[:20] + torch.as_tensor(ablang_logits20, device=device, dtype=combined.dtype)
            probs = F.softmax(combined / temperature, dim=-1)

            sampled_idx = torch.multinomial(probs, 1).item()
            S_t = torch.tensor([[sampled_idx]], device=device, dtype=torch.long)

            # Update the running chain string so the NEXT AbLang call sees
            # this residue as decided (matches "executed one residue at a
            # time in a random order").
            sampled_aa = PMPNN_ALPHABET[sampled_idx] if sampled_idx < 20 else "X"
            seq_list = list(chain_seqs[domain])
            seq_list[pos0] = sampled_aa
            chain_seqs[domain] = "".join(seq_list)

        S.scatter_(1, t[:, None], S_t)
        temp1 = pmpnn_model.W_s(S_t)
        h_S.scatter_(1, t[:, None, None].repeat(1, 1, temp1.shape[-1]), temp1)

    return S, chain_seqs


# ---------------------------------------------------------------------------
# 3b. Bookkeeping: map "physical PDB chain, local position" -> "Ig domain,
#     domain-local position". This is the piece that's genuinely different
#     for a VHH (one domain, no light chain) vs. an scFv (one physical chain
#     containing two domains back-to-back) vs. a conventional two-chain Fab.
# ---------------------------------------------------------------------------
def chain_code_map(masked_chains, visible_chains):
    """
    Reproduces tied_featurize's own chain-code assignment (verified in
    protein_mpnn_utils.py: `all_chains = masked_chains + visible_chains`,
    each sorted alphabetically first, then c=1,2,3,... assigned in that
    order) so we never have to guess/print chain_encoding_all by hand.
    """
    all_chains = sorted(masked_chains) + sorted(visible_chains)
    return {letter: code for code, letter in enumerate(all_chains, start=1)}


def build_position_bookkeeping(S, chain_encoding_all, code_to_pdb_chain, domain_layout, chain_M_pos):
    """
    domain_layout: dict, pdb_chain_letter -> list of (domain, start0, end0)
        segments covering that chain's full length, 0-indexed, end-exclusive.
        `domain` is 'H', 'L', or None (a non-Ig segment, e.g. an scFv linker).
        `end0=None` means "to the end of the chain" (resolved here).
        Examples:
          VHH, whole chain is the heavy domain:      [("H", 0, None)]
          scFv, VH then a 15-res linker then VL:      [("H", 0, 120),
                                                        (None, 120, 135),
                                                        ("L", 135, None)]

    Returns:
      domain_of_position:   list[str|None], len L_total
      domain_local_index:   list[int|None], len L_total (0-indexed WITHIN
                             that domain's own extracted sequence)
      chain_seqs:           dict, domain -> full wild-type sequence string
                             for that domain (not yet masked; caller applies
                             chain_M_pos to turn CDR positions into '*')
    """
    L_total = S.shape[1]
    idx_to_aa = {i: aa for i, aa in enumerate(PMPNN_ALPHABET)}

    domain_of_position = [None] * L_total
    domain_local_index = [None] * L_total

    # First pass: physical chain + local (within-chain) index per position,
    # via the SAME contiguous-per-chain ordering tied_featurize itself used.
    pdb_chain_of_position = [None] * L_total
    local_index_of_position = [None] * L_total
    running_local_idx = {}
    for pos in range(L_total):
        code = int(chain_encoding_all[0, pos].item())
        letter = code_to_pdb_chain.get(code)
        if letter is None:
            continue
        local_idx = running_local_idx.get(letter, 0)
        pdb_chain_of_position[pos] = letter
        local_index_of_position[pos] = local_idx
        running_local_idx[letter] = local_idx + 1
    chain_lengths = dict(running_local_idx)  # letter -> length, now that we've counted them

    # Second pass: resolve each chain's domain_layout (fill in end0=None) and
    # extract each domain's own wild-type sequence + assign domain_local_index.
    domain_seq_chars = {}  # domain -> list[str], built in encounter order
    for pos in range(L_total):
        letter = pdb_chain_of_position[pos]
        if letter is None or letter not in domain_layout:
            continue
        local_idx = local_index_of_position[pos]
        segments = domain_layout[letter]
        for domain, start0, end0 in segments:
            end0 = chain_lengths[letter] if end0 is None else end0
            if start0 <= local_idx < end0:
                if domain is not None:
                    domain_of_position[pos] = domain
                    domain_local_index[pos] = local_idx - start0
                    domain_seq_chars.setdefault(domain, [])
                    aa = idx_to_aa[int(S[0, pos].item())]
                    # 'X' = unresolved/missing density at this position (tied_featurize's
                    # stand-in for the '-' gaps seen earlier). AbLang's tokenizer rejects
                    # 'X' outright (it only accepts the 20 real amino acids or its own
                    # mask token). Since we genuinely don't know this residue's identity,
                    # feed AbLang its real "unknown" token ('*') here instead -- this is a
                    # framework position, never actively designed, so it stays '*' for the
                    # whole run, which AbLang (a masked-LM) handles the same way it already
                    # handles the CDR positions being masked during decoding.
                    domain_seq_chars[domain].append("*" if aa == "X" else aa)
                break
        else:
            raise ValueError(f"local index {local_idx} in chain '{letter}' not covered by domain_layout")

    chain_seqs = {d: "".join(chars) for d, chars in domain_seq_chars.items()}
    return domain_of_position, domain_local_index, chain_seqs


def apply_cdr_mask(chain_M_pos, domain_of_position, domain_local_index, chain_seqs, cdr_global_indices):
    """
    Zeroes chain_M_pos everywhere except `cdr_global_indices`, then punches
    '*' into chain_seqs at those (domain-local) positions so AbLang sees them
    as undecided too. Mutates chain_M_pos and chain_seqs in place.
    """
    chain_M_pos[:] = 0
    chain_seqs = {d: list(s) for d, s in chain_seqs.items()}
    for pos in cdr_global_indices:
        chain_M_pos[0, pos] = 1
        domain, pos0 = domain_of_position[pos], domain_local_index[pos]
        chain_seqs[domain][pos0] = "*"
    return {d: "".join(s) for d, s in chain_seqs.items()}


def guess_scfv_linker_span(seq: str):
    """
    CONVENIENCE HEURISTIC ONLY -- finds the longest Gly/Ser-rich stretch
    (typical (G4S)n scFv linkers) and returns (start0, end0) for it. This is
    NOT a substitute for numbering the sequence with ANARCI; always verify
    the VH/VL split it implies actually lands on the linker, not inside a
    CDR or framework region, before trusting it.
    """
    best = max(re.finditer(r"[GS]{10,30}", seq), key=lambda m: len(m.group()), default=None)
    if best is None:
        raise ValueError("No Gly/Ser-rich linker candidate found -- determine the VH/VL split via ANARCI instead.")
    return best.start(), best.end()


# ---------------------------------------------------------------------------
# 4. Example wiring for BOTH Bennett et al. structures. Fill in cdr_indices
#    (and, for the scFv, verify the linker span) before running for real.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "ProteinMPNN/vanilla_model_weights/v_48_020.pt"  # matches del Alamo's "vanilla" ProteinMPNN

    # Real chain IDs verified against RCSB/SAbDab for both depositions -- see
    # the module docstring. Neither structure is a conventional two-chain Fab.
    structures = [
        {
            "name": "9NH7 (VHH_flu_01, nanobody -- heavy-only, no light chain)",
            "pdb_path": "structures/raw/9NH7.pdb",                       # <- fill in
            "masked_chains": ["E"],                                # the VHH (1 of 2 copies; use "F" for the other)
            "visible_chains": ["B", "H"],                          # matched HA1 + HA2 protomer (use "A","G" for the other copy)
            # whole chain is a single heavy-only Ig domain:
            "domain_layout": {"E": [("H", 0, None)]},
            "ablang_domains": ("H",),                              # no light chain -> don't even load the light AbLang model
            "cdr_global_indices": [24, 25, 26, 27, 28, 29, 30, 50, 51, 52, 53, 54, 55, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],                              # <- fill in from your AHo/Chothia numbering of chain E
        },
        {
            "name": "9NFU (scFv6 -- VH + linker + VL fused into one physical chain)",
            "pdb_path": "structures/raw/9NFU.pdb",                        # <- fill in
            "masked_chains": ["C"],                                # the scFv
            "visible_chains": ["A"],                               # Toxin B
            # domain_layout for chain C is filled in below at runtime, once
            # we can read the actual sequence and locate the linker.
            "domain_layout": {'C': [(None, 0, 9), ('H', 9, 118), (None, 118, 168), ('L', 168, 246), (None, 246, None)]},
            "ablang_domains": ("H", "L"),
            "cdr_global_indices": [28, 29, 30, 31, 32, 33, 34, 54, 55, 56, 57, 58, 59, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 168, 169, 170, 171, 172, 173, 189, 190, 191, 192, 193, 194, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237],
        },
    ]

    pmpnn_model = load_proteinmpnn(checkpoint_path, device=device)

    for struct in structures:
        print(f"\n=== {struct['name']} ===")

        pdb_dict_list = parse_PDB(
            struct["pdb_path"], input_chain_list=struct["masked_chains"] + struct["visible_chains"]
        )
        name = pdb_dict_list[0]["name"]
        chain_id_dict = {name: (struct["masked_chains"], struct["visible_chains"])}

        X, S, mask, lengths, chain_M, chain_encoding_all, letter_list, visible_list, masked_list, \
            masked_chain_length_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
            tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, \
            tied_beta = tied_featurize([pdb_dict_list[0]], device, chain_id_dict)

        code_to_pdb_chain = {c: letter for letter, c in chain_code_map(struct["masked_chains"], struct["visible_chains"]).items()}

        domain_layout = struct["domain_layout"]
        if domain_layout is None:
            # scFv case: locate VH/linker/VL split from the actual sequence.
            # ALWAYS re-verify this against your own ANARCI numbering --
            # this heuristic only finds a plausible Gly/Ser-rich stretch.
            scfv_chain = struct["masked_chains"][0]
            full_seq = pdb_dict_list[0][f"seq_chain_{scfv_chain}"]
            linker_start, linker_end = guess_scfv_linker_span(full_seq)
            print(f"  guessed linker span in chain {scfv_chain}: [{linker_start}, {linker_end}) "
                  f"-- VERIFY with ANARCI before trusting this")
            domain_layout = {scfv_chain: [("H", 0, linker_start), (None, linker_start, linker_end), ("L", linker_end, None)]}

        domain_of_position, domain_local_index, chain_seqs = build_position_bookkeeping(
            S, chain_encoding_all, code_to_pdb_chain, domain_layout, chain_M_pos
        )

        # --- Restrict design to CDR residues only ---
        # GLOBAL positions in tied_featurize's concatenated sequence (masked
        # chains first, alphabetically). Fill in from your own AHo/Chothia
        # numbering -- this script does not derive CDR boundaries itself.
        cdr_global_indices = struct["cdr_global_indices"]
        if not cdr_global_indices:
            print("  (no cdr_global_indices supplied -- skipping design for this structure)")
            continue
        chain_seqs = apply_cdr_mask(chain_M_pos, domain_of_position, domain_local_index, chain_seqs, cdr_global_indices)

        ablang_helper = AbLangEnsembleHelper(device=device, domains=struct["ablang_domains"])

        S_design, final_chain_seqs = ensemble_design(
            pmpnn_model, ablang_helper,
            X, S, chain_M, chain_M_pos, mask, residue_idx, chain_encoding_all,
            domain_of_position, domain_local_index, chain_seqs,
            temperature=0.1, seed=0,
        )

        for domain, seq in final_chain_seqs.items():
            print(f"  designed {domain}-domain sequence: {seq}")
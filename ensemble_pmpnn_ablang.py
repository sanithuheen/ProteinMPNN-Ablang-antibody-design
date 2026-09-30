"""
ProteinMPNN + AbLang ensemble for de novo CDR design (del Alamo et al. 2025 method),
merged with the batch-runner that writes out full designed sequences.

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

This file used to be split across ensemble_pmpnn_ablang.py (the library:
AbLang wrapper, ProteinMPNN loading, the ensemble decode loop, chain/domain
bookkeeping) and get_ensemble_sequences.py (batch-runs the ensemble across
many seeds and writes full-length FASTA + manifest files). They're now one
file -- ensemble_pmpnn_ablang.py is no longer needed.
"""

import json
import os
import re
from pathlib import Path

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


# =============================================================================
# PART 1 -- LIBRARY (formerly ensemble_pmpnn_ablang.py)
# =============================================================================

# ---------------------------------------------------------------------------
# 1. AbLang wrapper -- loads weights
# ---------------------------------------------------------------------------
class AbLangEnsembleHelper:
    """
    Loads the real heavy- and/or light-chain AbLang models and returns, for a
    given masked sequence and position, a length-20 logit vector reordered
    into ProteinMPNN's amino-acid order (PMPNN_AA20) so the two logit
    vectors can be summed directly.

    `domains`: which AbLang model(s) to load. A VHH construct (e.g. 9NH7) has
    no light chain at all, so pass domains=("H",) to skip loading the light
    model entirely.

    `weights_dir`: local directory containing the AbLang weight folders --
    e.g. weights_dir/model-weights-heavy/{amodel.pt,hparams.json,vocab.json}
    and weights_dir/model-weights-light/{...}. This is exactly the layout
    ablang's own pretrained() downloads into (verified in ablang/pretrained.py:
    model_folder = os.path.join(<pkg dir>, "model-weights-{chain}")), so
    pre-downloaded/copied weights in that same layout are used as-is with no
    network access. Pass None to fall back to ablang's default "download"
    behavior (fetches from Oxford's servers if not already cached).
    """

    _ABLANG_CHAIN_NAME = {"H": "heavy", "L": "light"}

    def __init__(self, device="cpu", domains=("H", "L"), weights_dir=None):
        self.device = device

        def _model_folder(chain_name):
            if weights_dir is None:
                return "download"  # ablang's own download-and-cache path
            return os.path.join(weights_dir, f"model-weights-{chain_name}")

        self.models = {
            d: ablang.pretrained(
                chain=self._ABLANG_CHAIN_NAME[d],
                model_folder=_model_folder(self._ABLANG_CHAIN_NAME[d]),
                device=device,
            )
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
        raw = model(masked_seq, mode="likelihood")[0, pos0 + 1, :]   # (20,)
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


# =============================================================================
# PART 2 -- BATCH RUNNER (formerly get_ensemble_sequences.py)
# =============================================================================

# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = "ProteinMPNN/vanilla_model_weights/v_48_020.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Parent folder that directly contains "model-weights-heavy" and
# "model-weights-light" (each holding amodel.pt / hparams.json / vocab.json).
# <-- SET THIS to your local weights folder's path.
ABLANG_WEIGHTS_DIR = "ABLANG_MODEL_WEIGHTS"

TEMPERATURE = 0.1
N_DESIGNS = 100
SEED_START = 0          # seeds used will be SEED_START .. SEED_START + N_DESIGNS - 1

OUTPUT_DIR = Path("ensemble_sequences")   # where the full-sequence .fa files land
MANIFEST_DIR = Path("ensemble_sequences/manifests")  # reproducibility logs (params used)


STRUCTURES = [
    {
        "name": "9NH7",
        "pdb_path": "structures/raw/9NH7.pdb",
        "masked_chains": ["E"],           # the VHH
        "visible_chains": ["B", "H"],     # matched HA1 + HA2 protomer
        "domain_layout": {"E": [("H", 0, None)]},
        "ablang_domains": ("H",),
        "cdr_global_indices": [
            24, 25, 26, 27, 28, 29, 30,
            50, 51, 52, 53, 54, 55,
            97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        ],
    },
        {
            
        "name": "9NFU",
        "pdb_path": "structures/raw/9NFU.pdb",
        "masked_chains": ["C"],
        "visible_chains": ["A"],
        "domain_layout": {
            "C": [("H", 0, 145), ("L", 145, None)]
        },
        "ablang_domains": ("H", "L"),
        "cdr_global_indices": [
       28, 29, 30, 31, 32, 33, 34,
        54, 55, 56, 57, 58, 59,
        101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
        166, 167, 168, 169, 170, 171, 172, 173,
        189, 190, 191, 192, 193, 194, 195,
        228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
        ],
    },
]


# ---------------------------------------------------------------------------
# 2. RECONSTRUCT THE FULL PHYSICAL CHAIN FROM THE DESIGNED DOMAIN(S)
# ---------------------------------------------------------------------------

def build_template_chain_from_S(S_true, chain_encoding_all, code_to_pdb_chain, chain_letter):
    """
    Rebuilds `chain_letter`'s full original sequence directly from S_true --
    the SAME tensor chain_seqs/domain_of_position/domain_local_index are
    already built from -- instead of the raw parse_PDB-parsed sequence
    string. This guarantees the framework/linker pieces we splice in below
    live in the exact same index space as the redesigned CDR pieces, and use
    the same 'X' convention for missing density that ProteinMPNN/AbMPNN
    already use (parse_PDB's own text uses '-' for the same thing, and may
    not even be indexed the same way -- that mismatch was the source of the
    stray '-' characters).
    """
    idx_to_aa = {i: aa for i, aa in enumerate(PMPNN_ALPHABET)}
    chars = []
    for pos in range(S_true.shape[1]):
        code = int(chain_encoding_all[0, pos].item())
        if code_to_pdb_chain.get(code) == chain_letter:
            chars.append(idx_to_aa[int(S_true[0, pos].item())])
    return "".join(chars)


def reconstruct_full_chain(original_chain_seq, domain_layout_for_chain, final_chain_seqs):
    """
    Splices the redesigned per-domain sequences (e.g. {'H': ..., 'L': ...})
    back into the full original physical chain. Any non-Ig segment (linker,
    unresolved flanking regions) is copied verbatim from the template, so
    the result is the same length/format as the ProteinMPNN/AbMPNN .fa
    outputs -- a real, OASis-numberable antibody chain, not a CDR-only
    fragment.
    """
    pieces = []
    for domain, start0, end0 in domain_layout_for_chain:
        end0 = len(original_chain_seq) if end0 is None else end0
        if domain is None:
            pieces.append(original_chain_seq[start0:end0])
        else:
            pieces.append(final_chain_seqs[domain])
    return "".join(pieces)


def finalize_sequence(full_seq, struct, domain_of_position, domain_local_index, final_chain_seqs):
    """
    '*' is only meaningful DURING design (it's AbLang's mask token, standing
    in for 'X' because AbLang's tokenizer rejects X outright). Two different
    things can leave a '*' in the finished sequence:
      1. A framework position with genuinely missing density in the original
         structure -- expected, never redesigned, should just read as 'X'
         like the ProteinMPNN/AbMPNN outputs already do.
      2. A CDR position that somehow never got decided -- NOT expected, and
         worth a loud warning rather than silently masking it as 'X' too.
    This checks which case applies before doing the '*' -> 'X' swap.
    """
    cdr_positions_with_star = []
    for pos in struct["cdr_global_indices"]:
        domain, pos0 = domain_of_position[pos], domain_local_index[pos]
        if final_chain_seqs[domain][pos0] == "*":
            cdr_positions_with_star.append((domain, pos0))
    if cdr_positions_with_star:
        print(f"  WARNING: {len(cdr_positions_with_star)} CDR position(s) were never "
              f"decided and are being masked as 'X': {cdr_positions_with_star}")

    return full_seq.replace("*", "X")


# ---------------------------------------------------------------------------
# 3. RUN ONE STRUCTURE: FEATURIZE ONCE, DESIGN N_DESIGNS TIMES
# ---------------------------------------------------------------------------

def run_structure(struct, pmpnn_model):
    print(f"\n=== {struct['name']} ===")

    pdb_dict_list = parse_PDB(
        struct["pdb_path"], input_chain_list=struct["masked_chains"] + struct["visible_chains"]
    )
    name = pdb_dict_list[0]["name"]
    chain_id_dict = {name: (struct["masked_chains"], struct["visible_chains"])}

    X, S, mask, lengths, chain_M, chain_encoding_all, letter_list, visible_list, masked_list, \
        masked_chain_length_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, \
        tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, \
        tied_beta = tied_featurize([pdb_dict_list[0]], DEVICE, chain_id_dict)

    code_to_pdb_chain = {
        c: letter for letter, c in
        chain_code_map(struct["masked_chains"], struct["visible_chains"]).items()
    }

    domain_of_position, domain_local_index, chain_seqs = build_position_bookkeeping(
        S, chain_encoding_all, code_to_pdb_chain, struct["domain_layout"], chain_M_pos
    )
    


    # Full, un-redesigned original sequence for each masked (antibody) chain --
    # used later to fill in framework/linker/unresolved regions verbatim.
    # Built from S_true (not the raw parse_PDB text) so it's guaranteed to be
    # in the same index space as chain_seqs / domain_local_index.
    original_seq_by_chain = {
        c: build_template_chain_from_S(S, chain_encoding_all, code_to_pdb_chain, c)
        for c in struct["masked_chains"]
    }

    # This applies the CDR mask ONCE, producing a template with '*' at every
    # CDR position. We deep-copy this fresh before each seed below, since
    # ensemble_design() mutates the chain_seqs dict it's given in place.
    masked_template = apply_cdr_mask(
        chain_M_pos, domain_of_position, domain_local_index, chain_seqs, struct["cdr_global_indices"]
    )
    



    ablang_helper = AbLangEnsembleHelper(
        device=DEVICE, domains=struct["ablang_domains"], weights_dir=ABLANG_WEIGHTS_DIR
    )

    the_chain = struct["masked_chains"][0]  # single physical chain for both structures here
    records = []

    for seed in range(SEED_START, SEED_START + N_DESIGNS):
        seq_input = {d: s for d, s in masked_template.items()}  # fresh copy for this seed

        _, final_chain_seqs = ensemble_design(
            pmpnn_model, ablang_helper,
            X, S, chain_M, chain_M_pos, mask, residue_idx, chain_encoding_all,
            domain_of_position, domain_local_index, seq_input,
            temperature=TEMPERATURE, seed=seed,
        )

        full_seq = reconstruct_full_chain(
            original_seq_by_chain[the_chain],
            struct["domain_layout"][the_chain],
            final_chain_seqs,
        )
        full_seq = finalize_sequence(
            full_seq, struct, domain_of_position, domain_local_index, final_chain_seqs
        )
        records.append((seed, full_seq))
        print(f"  seed {seed}: {full_seq}")

    return records


# ---------------------------------------------------------------------------
# 4. WRITE OUTPUT (FASTA, one record per design) + A REPRODUCIBILITY MANIFEST
# ---------------------------------------------------------------------------

def write_fasta(struct_name, records):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{struct_name}_ensemble.fa"
    with open(out_path, "w") as f:
        for seed, seq in records:
            f.write(f">{struct_name}_ensemble seed={seed} model=ProteinMPNN+AbLang T={TEMPERATURE}\n")
            f.write(f"{seq}\n")
    print(f"  -> wrote {len(records)} designs to {out_path}")
    return out_path


def write_manifest(struct):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "structure": struct["name"],
        "pdb_path": struct["pdb_path"],
        "checkpoint_path": CHECKPOINT_PATH,
        "temperature": TEMPERATURE,
        "n_designs": N_DESIGNS,
        "seeds": list(range(SEED_START, SEED_START + N_DESIGNS)),
        "domain_layout": {k: list(v) for k, v in struct["domain_layout"].items()},
        "cdr_global_indices": struct["cdr_global_indices"],
        "ablang_domains": struct["ablang_domains"],
    }
    out_path = MANIFEST_DIR / f"{struct['name']}_ensemble_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  -> wrote manifest to {out_path}")


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def main():
    pmpnn_model = load_proteinmpnn(CHECKPOINT_PATH, device=DEVICE)

    for struct in STRUCTURES:
        records = run_structure(struct, pmpnn_model)
        write_fasta(struct["name"], records)
        write_manifest(struct)


if __name__ == "__main__":
    main()
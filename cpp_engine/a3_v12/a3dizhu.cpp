#include "a3dizhu.h"
#include <sstream>
#include <numeric>
#include <cmath>
#include <set>

namespace a3dizhu {

// ======================== Constants ========================
const int STRAIGHT_RANK_VAL[NUM_RANKS] = {
    2,  // rank 0 = '4'
    3,  // rank 1 = '5'
    4,  // rank 2 = '6'
    5,  // rank 3 = '7'
    6,  // rank 4 = '8'
    7,  // rank 5 = '9'
    8,  // rank 6 = '10'
    9,  // rank 7 = 'J'
    10, // rank 8 = 'Q'
    11, // rank 9 = 'K'
    12, // rank 10 = 'A'
    0,  // rank 11 = '2' (not valid in straight)
    1,  // rank 12 = '3'
};

const char* SUIT_NAMES[NUM_SUITS] = {"diamond", "club", "heart", "spade"};
const char* RANK_NAMES[NUM_RANKS] = {
    "4","5","6","7","8","9","10","J","Q","K","A","2","3"
};

static const int STRAIGHT_SEQS[8][5] = {
    {12, 0, 1, 2, 3},   // 3-4-5-6-7
    { 0, 1, 2, 3, 4},   // 4-5-6-7-8
    { 1, 2, 3, 4, 5},   // 5-6-7-8-9
    { 2, 3, 4, 5, 6},   // 6-7-8-9-10
    { 3, 4, 5, 6, 7},   // 7-8-9-10-J
    { 4, 5, 6, 7, 8},   // 8-9-10-J-Q
    { 5, 6, 7, 8, 9},   // 9-10-J-Q-K
    { 6, 7, 8, 9, 10},  // 10-J-Q-K-A
};

// ======================== Card Utilities ========================

std::string card_to_string(int card) {
    return std::string(SUIT_NAMES[card_suit(card)]) + "_" + RANK_NAMES[card_rank(card)];
}

static std::unordered_map<std::string, int> build_string_to_card_map() {
    std::unordered_map<std::string, int> m;
    for (int i = 0; i < NUM_CARDS; i++)
        m[card_to_string(i)] = i;
    return m;
}
static const auto& s2c_map() {
    static auto m = build_string_to_card_map();
    return m;
}

int string_to_card(const std::string& s) {
    auto it = s2c_map().find(s);
    return (it != s2c_map().end()) ? it->second : -1;
}

// ======================== HandInfo Methods ========================

std::string HandInfo::to_key() const {
    if (type == HAND_PASS) return "pass";
    if (type == HAND_DECLARE) return "declare";
    std::vector<std::string> ids;
    for_each_card(cards, [&](int c){ ids.push_back(card_to_string(c)); });
    std::sort(ids.begin(), ids.end());
    std::string key;
    for (size_t i = 0; i < ids.size(); i++) {
        if (i > 0) key += '|';
        key += ids[i];
    }
    return key;
}

void HandInfo::to_card_bits(int8_t* out) const {
    std::memset(out, 0, NUM_CARDS);
    if (type == HAND_DECLARE) {
        for (int i = 0; i < NUM_CARDS; i++) out[i] = 1;
        return;
    }
    for_each_card(cards, [&](int c){ out[c] = 1; });
}

void Engine::cardset_to_feature(CardSet cs, int8_t* out) {
    std::memset(out, 0, NUM_CARDS);
    for_each_card(cs, [&](int c){ out[c] = 1; });
}

// ======================== Hand Detection ========================

static int find_max_score_card(CardSet cs) {
    int best = -1, best_score = -1;
    for_each_card(cs, [&](int c){
        int sc = card_score(c);
        if (sc > best_score) { best_score = sc; best = c; }
    });
    return best;
}

static int find_straight_primary(CardSet cs) {
    int best = -1, best_srv = -1, best_suit = -1;
    for_each_card(cs, [&](int c){
        int srv = STRAIGHT_RANK_VAL[card_rank(c)];
        int suit = card_suit(c);
        if (srv > best_srv || (srv == best_srv && suit > best_suit)) {
            best = c; best_srv = srv; best_suit = suit;
        }
    });
    return best;
}

static bool is_same_suit(CardSet cs) {
    for (int s = 0; s < NUM_SUITS; s++)
        if ((cs & suit_mask(s)) == cs) return true;
    return false;
}

static bool check_straight_ranks(CardSet cs, int start_val = 1, int end_val = 12) {
    uint16_t rank_bits = 0;
    CardSet tmp_cs = cs;
    while (tmp_cs) {
        int c = ctz64(tmp_cs);
        rank_bits |= (uint16_t)(1 << card_rank(c));
        tmp_cs &= tmp_cs - 1;
    }
    if (rank_bits & (1 << 11)) return false; // '2' not allowed
    int bit_count = 0;
    for (uint16_t tmp = rank_bits; tmp; tmp &= tmp - 1) bit_count++;
    if (bit_count != 5) return false;
    int mn = 13, mx = 0;
    for (int r = 0; r < NUM_RANKS; r++) {
        if (rank_bits & (1 << r)) {
            int v = STRAIGHT_RANK_VAL[r];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
    }
    if (mn < start_val || mx > end_val) return false;
    return (mx - mn == 4);
}

static HandInfo detect_single(CardSet cs) {
    int c = ctz64(cs);
    return {HAND_SINGLE, cs, c, 1};
}

static HandInfo detect_pair(CardSet cs) {
    auto v = cards_vec(cs);
    if (v.size() != 2 || card_rank(v[0]) != card_rank(v[1]))
        return {};
    int primary = (card_score(v[0]) >= card_score(v[1])) ? v[0] : v[1];
    return {HAND_PAIR, cs, primary, 2};
}

static HandInfo detect_triple(CardSet cs) {
    auto v = cards_vec(cs);
    if (v.size() != 3) return {};
    if (card_rank(v[0]) != card_rank(v[1]) || card_rank(v[1]) != card_rank(v[2]))
        return {};
    return {HAND_TRIPLE, cs, find_max_score_card(cs), 3};
}

static HandInfo detect_four_with_one(CardSet cs) {
    for (int r = 0; r < NUM_RANKS; r++) {
        CardSet rc = cs & rank_mask(r);
        if (popcount64(rc) == 4) {
            CardSet kicker = cs & ~rc;
            if (popcount64(kicker) == 1)
                return {HAND_FOUR_WITH_ONE, cs, find_max_score_card(rc), 5};
        }
    }
    return {};
}

static HandInfo detect_full_house(CardSet cs) {
    int triple_rank = -1, pair_rank = -1;
    for (int r = 0; r < NUM_RANKS; r++) {
        int cnt = popcount64(cs & rank_mask(r));
        if (cnt == 3) triple_rank = r;
        else if (cnt == 2) pair_rank = r;
    }
    if (triple_rank >= 0 && pair_rank >= 0)
        return {HAND_FULL_HOUSE, cs, find_max_score_card(cs & rank_mask(triple_rank)), 5};
    return {};
}

HandInfo detect_hand(CardSet cs, int size) {
    if (size == 1) return detect_single(cs);
    if (size == 2) return detect_pair(cs);
    if (size == 3) return detect_triple(cs);
    if (size == 5) {
        bool same_suit = is_same_suit(cs);
        bool is_str = check_straight_ranks(cs);
        if (same_suit && is_str)
            return {HAND_STRAIGHT_FLUSH, cs, find_straight_primary(cs), 5};
        auto fwo = detect_four_with_one(cs);
        if (fwo.is_play()) return fwo;
        auto fh = detect_full_house(cs);
        if (fh.is_play()) return fh;
        if (same_suit)
            return {HAND_FLUSH, cs, find_max_score_card(cs), 5};
        if (is_str)
            return {HAND_STRAIGHT, cs, find_straight_primary(cs), 5};
    }
    return {};
}

// ======================== Hand Comparison ========================

static int compare_primary(const HandInfo& a, const HandInfo& b) {
    if ((a.type == HAND_STRAIGHT || a.type == HAND_STRAIGHT_FLUSH) &&
        (b.type == HAND_STRAIGHT || b.type == HAND_STRAIGHT_FLUSH)) {
        int ra = STRAIGHT_RANK_VAL[card_rank(a.primary_card)];
        int rb = STRAIGHT_RANK_VAL[card_rank(b.primary_card)];
        if (ra != rb) return ra - rb;
        return card_suit(a.primary_card) - card_suit(b.primary_card);
    }
    if (a.type == HAND_FLUSH && b.type == HAND_FLUSH) {
        int sd = card_suit(a.primary_card) - card_suit(b.primary_card);
        if (sd != 0) return sd;
        return card_rank(a.primary_card) - card_rank(b.primary_card);
    }
    return card_score(a.primary_card) - card_score(b.primary_card);
}

int compare_hands(const HandInfo& a, const HandInfo& b) {
    if (a.size == 5 && b.size == 5) {
        int pa = five_card_priority(a.type);
        int pb = five_card_priority(b.type);
        if (pa != pb) return pa - pb;
    }
    return compare_primary(a, b);
}

bool can_beat(const HandInfo& a, const HandInfo& b) {
    if (a.size != b.size) return false;
    return compare_hands(a, b) > 0;
}

// ======================== Subset Enumeration Helpers ========================

using EnumFn = std::function<void(CardSet)>;

static void enum_2subsets(CardSet cs, const EnumFn& fn) {
    // Optimized: iterate bits directly without vector allocation
    uint64_t tmp1 = cs;
    while (tmp1) {
        int c1 = ctz64(tmp1);
        tmp1 &= tmp1 - 1;
        uint64_t tmp2 = tmp1;
        while (tmp2) {
            int c2 = ctz64(tmp2);
            tmp2 &= tmp2 - 1;
            fn(card_bit(c1) | card_bit(c2));
        }
    }
}

static void enum_3subsets(CardSet cs, const EnumFn& fn) {
    uint64_t t1 = cs;
    while (t1) {
        int c1 = ctz64(t1); t1 &= t1 - 1;
        uint64_t t2 = t1;
        while (t2) {
            int c2 = ctz64(t2); t2 &= t2 - 1;
            uint64_t t3 = t2;
            while (t3) {
                int c3 = ctz64(t3); t3 &= t3 - 1;
                fn(card_bit(c1) | card_bit(c2) | card_bit(c3));
            }
        }
    }
}

static void enum_5subsets(CardSet cs, const EnumFn& fn) {
    uint64_t t1 = cs;
    while (t1) {
        int c1 = ctz64(t1); t1 &= t1 - 1;
        uint64_t t2 = t1;
        while (t2) {
            int c2 = ctz64(t2); t2 &= t2 - 1;
            uint64_t t3 = t2;
            while (t3) {
                int c3 = ctz64(t3); t3 &= t3 - 1;
                uint64_t t4 = t3;
                while (t4) {
                    int c4 = ctz64(t4); t4 &= t4 - 1;
                    uint64_t t5 = t4;
                    while (t5) {
                        int c5 = ctz64(t5); t5 &= t5 - 1;
                        fn(card_bit(c1)|card_bit(c2)|card_bit(c3)|card_bit(c4)|card_bit(c5));
                    }
                }
            }
        }
    }
}

static void enum_straight_combos(
    const CardSet rank_cards[5], int depth, CardSet current,
    std::unordered_set<CardSet>& seen,
    std::vector<HandInfo>& results,
    const HandInfo* last)
{
    if (depth == 5) {
        if (seen.count(current)) return;
        HandInfo h = detect_hand(current, 5);
        if (!h.is_play()) return;
        if (last && !can_beat(h, *last)) return;
        seen.insert(current);
        results.push_back(h);
        return;
    }
    CardSet rem = rank_cards[depth];
    while (rem) {
        int c = ctz64(rem);
        rem &= rem - 1;
        enum_straight_combos(rank_cards, depth+1, current | card_bit(c), seen, results, last);
    }
}

// ======================== Legal Move Enumeration ========================

static void find_straights(CardSet hand, const HandInfo* last,
    std::unordered_set<CardSet>& seen, std::vector<HandInfo>& results,
    int start_val, int end_val)
{
    for (int si = 0; si < 8; si++) {
        int seq_min = 99, seq_max = 0;
        for (int j = 0; j < 5; j++) {
            int v = STRAIGHT_RANK_VAL[STRAIGHT_SEQS[si][j]];
            seq_min = std::min(seq_min, v);
            seq_max = std::max(seq_max, v);
        }
        if (seq_min < start_val || seq_max > end_val) continue;

        bool has_all = true;
        CardSet rc[5];
        for (int j = 0; j < 5; j++) {
            rc[j] = hand & rank_mask(STRAIGHT_SEQS[si][j]);
            if (rc[j] == 0) { has_all = false; break; }
        }
        if (!has_all) continue;
        enum_straight_combos(rc, 0, 0, seen, results, last);
    }
}

static void find_flushes(CardSet hand, const HandInfo* last,
    std::unordered_set<CardSet>& seen, std::vector<HandInfo>& results,
    int start_val, int end_val)
{
    for (int s = 0; s < NUM_SUITS; s++) {
        CardSet sc = hand & suit_mask(s);
        if (popcount64(sc) < 5) continue;
        enum_5subsets(sc, [&](CardSet combo) {
            if (seen.count(combo)) return;
            HandInfo h = detect_hand(combo, 5);
            if (!h.is_play()) return;
            if (h.type == HAND_STRAIGHT_FLUSH) {
                if (check_straight_ranks(combo, start_val, end_val))
                    return;
                h.type = HAND_FLUSH;
                h.primary_card = find_max_score_card(combo);
            }
            if (h.type != HAND_FLUSH) return;
            if (last && !can_beat(h, *last)) return;
            seen.insert(combo);
            results.push_back(h);
        });
    }
}

static void find_full_houses(CardSet hand, const HandInfo* last,
    std::unordered_set<CardSet>& seen, std::vector<HandInfo>& results)
{
    for (int tr = 0; tr < NUM_RANKS; tr++) {
        CardSet tc = hand & rank_mask(tr);
        if (popcount64(tc) < 3) continue;
        enum_3subsets(tc, [&](CardSet triple_set) {
            for (int pr = 0; pr < NUM_RANKS; pr++) {
                if (pr == tr) continue;
                CardSet pc = hand & rank_mask(pr);
                if (popcount64(pc) < 2) continue;
                enum_2subsets(pc, [&](CardSet pair_set) {
                    CardSet combo = triple_set | pair_set;
                    if (seen.count(combo)) return;
                    HandInfo h = detect_hand(combo, 5);
                    if (!h.is_play() || h.type != HAND_FULL_HOUSE) return;
                    if (last && !can_beat(h, *last)) return;
                    seen.insert(combo);
                    results.push_back(h);
                });
            }
        });
    }
}

static void find_four_with_ones(CardSet hand, const HandInfo* last,
    std::unordered_set<CardSet>& seen, std::vector<HandInfo>& results)
{
    for (int qr = 0; qr < NUM_RANKS; qr++) {
        CardSet qc = hand & rank_mask(qr);
        if (popcount64(qc) < 4) continue;
        CardSet quad = qc;
        for_each_card(hand & ~quad, [&](int kicker) {
            CardSet combo = quad | card_bit(kicker);
            if (seen.count(combo)) return;
            HandInfo h = detect_hand(combo, 5);
            if (!h.is_play() || h.type != HAND_FOUR_WITH_ONE) return;
            if (last && !can_beat(h, *last)) return;
            seen.insert(combo);
            results.push_back(h);
        });
    }
}

static void find_straight_flushes(CardSet hand, const HandInfo* last,
    std::unordered_set<CardSet>& seen, std::vector<HandInfo>& results,
    int start_val, int end_val)
{
    for (int s = 0; s < NUM_SUITS; s++) {
        CardSet sc = hand & suit_mask(s);
        if (popcount64(sc) < 5) continue;
        for (int si = 0; si < 8; si++) {
            int seq_min = 99, seq_max = 0;
            for (int j = 0; j < 5; j++) {
                int v = STRAIGHT_RANK_VAL[STRAIGHT_SEQS[si][j]];
                seq_min = std::min(seq_min, v);
                seq_max = std::max(seq_max, v);
            }
            if (seq_min < start_val || seq_max > end_val) continue;

            bool has_all = true;
            CardSet rc[5];
            for (int j = 0; j < 5; j++) {
                rc[j] = sc & rank_mask(STRAIGHT_SEQS[si][j]);
                if (rc[j] == 0) { has_all = false; break; }
            }
            if (!has_all) continue;
            CardSet combo = rc[0]|rc[1]|rc[2]|rc[3]|rc[4];
            if (seen.count(combo)) continue;
            HandInfo h = detect_hand(combo, 5);
            if (!h.is_play() || h.type != HAND_STRAIGHT_FLUSH) continue;
            if (last && !can_beat(h, *last)) continue;
            seen.insert(combo);
            results.push_back(h);
        }
    }
}

static int hand_sort_key_priority(const HandInfo& h) {
    if (h.size == 5) return five_card_priority(h.type);
    return 0;
}

static bool hand_sort_cmp(const HandInfo& a, const HandInfo& b) {
    if (a.size != b.size) return a.size < b.size;
    int pa = hand_sort_key_priority(a);
    int pb = hand_sort_key_priority(b);
    if (pa != pb) return pa < pb;
    return card_score(a.primary_card) < card_score(b.primary_card);
}

std::vector<HandInfo> get_all_hands(CardSet hand, int start_val, int end_val) {
    std::vector<HandInfo> results;
    std::unordered_set<CardSet> seen;

    for_each_card(hand, [&](int c) {
        CardSet cs = card_bit(c);
        results.push_back({HAND_SINGLE, cs, c, 1});
        seen.insert(cs);
    });

    for (int r = 0; r < NUM_RANKS; r++) {
        CardSet rc = hand & rank_mask(r);
        int cnt = popcount64(rc);
        if (cnt >= 2) {
            enum_2subsets(rc, [&](CardSet cs) {
                HandInfo h = detect_hand(cs, 2);
                if (h.is_play() && !seen.count(cs)) {
                    seen.insert(cs);
                    results.push_back(h);
                }
            });
        }
        if (cnt >= 3) {
            enum_3subsets(rc, [&](CardSet cs) {
                HandInfo h = detect_hand(cs, 3);
                if (h.is_play() && !seen.count(cs)) {
                    seen.insert(cs);
                    results.push_back(h);
                }
            });
        }
    }

    find_straights(hand, nullptr, seen, results, start_val, end_val);
    find_flushes(hand, nullptr, seen, results, start_val, end_val);
    find_full_houses(hand, nullptr, seen, results);
    find_four_with_ones(hand, nullptr, seen, results);

    std::sort(results.begin(), results.end(), hand_sort_cmp);
    return results;
}

std::vector<HandInfo> get_beating_hands(CardSet hand, const HandInfo& last, int start_val, int end_val) {
    std::vector<HandInfo> results;
    std::unordered_set<CardSet> seen;

    if (last.size == 1) {
        for_each_card(hand, [&](int c) {
            HandInfo h = {HAND_SINGLE, card_bit(c), c, 1};
            if (can_beat(h, last)) results.push_back(h);
        });
    } else if (last.size == 2) {
        for (int r = 0; r < NUM_RANKS; r++) {
            CardSet rc = hand & rank_mask(r);
            if (popcount64(rc) < 2) continue;
            enum_2subsets(rc, [&](CardSet cs) {
                HandInfo h = detect_hand(cs, 2);
                if (h.is_play() && can_beat(h, last))
                    results.push_back(h);
            });
        }
    } else if (last.size == 3) {
        for (int r = 0; r < NUM_RANKS; r++) {
            CardSet rc = hand & rank_mask(r);
            if (popcount64(rc) < 3) continue;
            enum_3subsets(rc, [&](CardSet cs) {
                HandInfo h = detect_hand(cs, 3);
                if (h.is_play() && can_beat(h, last))
                    results.push_back(h);
            });
        }
    } else if (last.size == 5) {
        int lp = five_card_priority(last.type);
        if (lp <= 0) find_straights(hand, &last, seen, results, start_val, end_val);
        if (lp <= 1) find_flushes(hand, &last, seen, results, start_val, end_val);
        if (lp <= 2) find_full_houses(hand, &last, seen, results);
        if (lp <= 3) find_four_with_ones(hand, &last, seen, results);
        if (lp <= 4) find_straight_flushes(hand, &last, seen, results, start_val, end_val);
    }

    std::sort(results.begin(), results.end(), hand_sort_cmp);
    return results;
}

// ======================== Team Assignment ========================

void assign_teams(const CardSet hands[NUM_PLAYERS],
                  Team out[NUM_PLAYERS], bool& is_solo)
{
    int s3_owner = -1, sA_owner = -1;
    int spade3 = make_card(3, 12);
    int spadeA = make_card(3, 10);
    for (int i = 0; i < NUM_PLAYERS; i++) {
        if (hands[i] & card_bit(spade3)) s3_owner = i;
        if (hands[i] & card_bit(spadeA)) sA_owner = i;
    }
    for (int i = 0; i < NUM_PLAYERS; i++) out[i] = TEAM_OPPONENT;
    is_solo = false;
    if (s3_owner == sA_owner && s3_owner >= 0) {
        out[s3_owner] = TEAM_SOLO;
        is_solo = true;
    } else {
        if (s3_owner >= 0) out[s3_owner] = TEAM_SPADE_A3;
        if (sA_owner >= 0) out[sA_owner] = TEAM_SPADE_A3;
    }
}

// ======================== GameState ========================

void GameState::compute_observed_teams() {
    if (is_declared && declarant >= 0) {
        for (int i = 0; i < num_players; i++) observed_teams[i] = TEAM_OPPONENT;
        observed_teams[declarant] = TEAM_SOLO;
        return;
    }
    int s3 = spade3_player, sA = spadeA_player;
    if (s3 >= 0 && sA >= 0) {
        if (s3 == sA) {
            for (int i = 0; i < num_players; i++) observed_teams[i] = TEAM_OPPONENT;
            observed_teams[s3] = TEAM_SOLO;
        } else {
            for (int i = 0; i < num_players; i++) observed_teams[i] = TEAM_OPPONENT;
            observed_teams[s3] = TEAM_SPADE_A3;
            observed_teams[sA] = TEAM_SPADE_A3;
        }
    } else if (s3 >= 0) {
        for (int i = 0; i < num_players; i++) observed_teams[i] = TEAM_UNKNOWN;
        observed_teams[s3] = TEAM_SPADE_A3;
    } else if (sA >= 0) {
        for (int i = 0; i < num_players; i++) observed_teams[i] = TEAM_UNKNOWN;
        observed_teams[sA] = TEAM_SPADE_A3;
    } else {
        for (int i = 0; i < num_players; i++) observed_teams[i] = TEAM_UNKNOWN;
    }
}

bool GameState::is_terminal() const {
    if (is_declaration_phase) return false;
    if (is_declared && declarant >= 0)
        return rankings.size() >= 1;
    if ((int)rankings.size() >= num_players - 1) return true;

    std::set<int> finished(rankings.begin(), rankings.end());

    if (is_solo) {
        int solo_idx = -1;
        for (int i = 0; i < num_players; i++)
            if (actual_teams[i] == TEAM_SOLO) { solo_idx = i; break; }
        if (solo_idx >= 0) {
            if (finished.count(solo_idx)) return true;
            bool all_opp_done = true;
            for (int i = 0; i < num_players; i++) {
                if (actual_teams[i] != TEAM_SOLO && !finished.count(i))
                    all_opp_done = false;
            }
            if (all_opp_done) return true;
        }
    }

    std::unordered_map<int, std::vector<int>> team_members;
    for (int i = 0; i < num_players; i++) {
        if (actual_teams[i] != TEAM_UNKNOWN)
            team_members[(int)actual_teams[i]].push_back(i);
    }
    for (auto& [team, members] : team_members) {
        if (team == (int)TEAM_SOLO) continue;
        if (members.empty()) continue;
        bool all_done = true;
        for (int m : members)
            if (!finished.count(m)) all_done = false;
        if (all_done) return true;
    }
    return false;
}

int GameState::next_active(int from_player) const {
    std::set<int> finished(rankings.begin(), rankings.end());
    for (int off = 1; off <= num_players; off++) {
        int p = (from_player + off) % num_players;
        if (!finished.count(p)) return p;
    }
    return from_player;
}

std::vector<HandInfo> GameState::get_legal_moves() const {
    if (is_declaration_phase) {
        std::vector<HandInfo> moves;
        bool can_declare = true;
        if (declare_require_both_spades) {
            int spade3 = make_card(3, 12);
            int spadeA = make_card(3, 10);
            can_declare = (hands[declaration_turn] & card_bit(spade3)) != 0
                       && (hands[declaration_turn] & card_bit(spadeA)) != 0;
        }
        if (can_declare) {
            HandInfo decl; decl.type = HAND_DECLARE; decl.size = 0;
            moves.push_back(decl);
        }
        HandInfo pass;
        moves.push_back(pass);
        return moves;
    }
    CardSet my = hands[current_player];
    if (my == 0) return {};

    int sv = straight_start_val, ev = straight_end_val;

    if (last_play.is_pass()) {
        auto all = get_all_hands(my, sv, ev);
        if (is_first_turn) {
            int diamond4 = make_card(0, 0);
            std::vector<HandInfo> filtered;
            for (auto& h : all) {
                if (h.cards & card_bit(diamond4))
                    filtered.push_back(h);
            }
            return filtered;
        }
        return all;
    } else {
        auto beating = get_beating_hands(my, last_play, sv, ev);
        if (popcount64(my) == 1 && !beating.empty())
            return beating;
        HandInfo pass;
        beating.push_back(pass);
        return beating;
    }
}

GameState GameState::apply_move(const HandInfo& move) const {
    if (is_declaration_phase) {
        int p = declaration_turn;
        if (move.is_declare()) {
            GameState ns;
            ns.declare_require_both_spades = declare_require_both_spades;
            ns.straight_start_val = straight_start_val;
            ns.straight_end_val = straight_end_val;
            for (int i = 0; i < num_players; i++) ns.hands[i] = hands[i];
            ns.current_player = current_player;
            ns.last_play = {}; ns.last_play_player = -1;
            ns.pass_count = 0; ns.is_first_turn = true;
            ns.num_players = num_players;
            for (int i = 0; i < num_players; i++) ns.actual_teams[i] = TEAM_OPPONENT;
            ns.actual_teams[p] = TEAM_SOLO;
            ns.is_solo = true;
            ns.spade3_player = -1; ns.spadeA_player = -1;
            ns.is_declaration_phase = false;
            ns.declaration_turn = 0; ns.declaration_passes = 0;
            ns.is_declared = true; ns.declarant = p;
            ns.compute_observed_teams();
            return ns;
        } else {
            int new_passes = declaration_passes + 1;
            int next_turn = (p + 1) % num_players;
            GameState ns;
            ns.declare_require_both_spades = declare_require_both_spades;
            ns.straight_start_val = straight_start_val;
            ns.straight_end_val = straight_end_val;
            for (int i = 0; i < num_players; i++) ns.hands[i] = hands[i];
            ns.current_player = current_player;
            ns.last_play = {}; ns.last_play_player = -1;
            ns.pass_count = 0; ns.is_first_turn = true;
            ns.num_players = num_players;
            for (int i = 0; i < num_players; i++) ns.actual_teams[i] = actual_teams[i];
            ns.is_solo = is_solo;
            ns.spade3_player = spade3_player; ns.spadeA_player = spadeA_player;
            ns.is_declared = false; ns.declarant = -1;
            if (new_passes >= num_players) {
                ns.is_declaration_phase = false;
                ns.declaration_turn = 0; ns.declaration_passes = new_passes;
            } else {
                ns.is_declaration_phase = true;
                ns.declaration_turn = next_turn; ns.declaration_passes = new_passes;
            }
            ns.compute_observed_teams();
            return ns;
        }
    }

    if (move.is_pass()) {
        std::set<int> finished(rankings.begin(), rankings.end());
        int active_count = num_players - (int)rankings.size();
        int new_pass_count = pass_count + 1;
        bool last_still_active = (last_play_player >= 0 && !finished.count(last_play_player));
        int passes_needed = last_still_active ? active_count - 1 : active_count;

        GameState ns;
        ns.declare_require_both_spades = declare_require_both_spades;
        ns.straight_start_val = straight_start_val;
        ns.straight_end_val = straight_end_val;
        for (int i = 0; i < num_players; i++) ns.hands[i] = hands[i];
        ns.rankings = rankings;
        for (int i = 0; i < num_players; i++) ns.actual_teams[i] = actual_teams[i];
        ns.is_solo = is_solo; ns.num_players = num_players;
        ns.spade3_player = spade3_player; ns.spadeA_player = spadeA_player;
        ns.is_first_turn = false;
        ns.is_declared = is_declared; ns.declarant = declarant;
        ns.is_declaration_phase = false;

        if (new_pass_count >= passes_needed) {
            int new_leader;
            if (last_still_active) {
                new_leader = last_play_player;
            } else {
                GameState tmp = *this;
                new_leader = tmp.next_active(
                    last_play_player >= 0 ? last_play_player : current_player);
            }
            ns.current_player = new_leader;
            ns.last_play = {}; ns.last_play_player = -1;
            ns.pass_count = 0;
        } else {
            ns.current_player = next_active(current_player);
            ns.last_play = last_play; ns.last_play_player = last_play_player;
            ns.pass_count = new_pass_count;
        }
        ns.compute_observed_teams();
        return ns;
    }

    CardSet played = move.cards;
    GameState ns;
    ns.declare_require_both_spades = declare_require_both_spades;
    ns.straight_start_val = straight_start_val;
    ns.straight_end_val = straight_end_val;
    for (int i = 0; i < num_players; i++)
        ns.hands[i] = (i == current_player) ? (hands[i] & ~played) : hands[i];

    int new_s3 = spade3_player, new_sA = spadeA_player;
    int spade3_card = make_card(3, 12);
    int spadeA_card = make_card(3, 10);
    if (played & card_bit(spade3_card)) new_s3 = current_player;
    if (played & card_bit(spadeA_card)) new_sA = current_player;

    ns.rankings = rankings;
    if (ns.hands[current_player] == 0)
        ns.rankings.push_back(current_player);

    for (int i = 0; i < num_players; i++) ns.actual_teams[i] = actual_teams[i];
    ns.is_solo = is_solo; ns.num_players = num_players;
    ns.spade3_player = new_s3; ns.spadeA_player = new_sA;
    ns.last_play = move; ns.last_play_player = current_player;
    ns.pass_count = 0; ns.is_first_turn = false;
    ns.is_declared = is_declared; ns.declarant = declarant;
    ns.is_declaration_phase = false;
    ns.compute_observed_teams();

    if (ns.is_terminal()) {
        std::set<int> fin(ns.rankings.begin(), ns.rankings.end());
        std::vector<int> remaining;
        for (int i = 0; i < num_players; i++)
            if (!fin.count(i)) remaining.push_back(i);
        std::sort(remaining.begin(), remaining.end(), [&](int a, int b){
            return popcount64(ns.hands[a]) > popcount64(ns.hands[b]);
        });
        for (int i : remaining) ns.rankings.push_back(i);
    }

    std::set<int> finished_final(ns.rankings.begin(), ns.rankings.end());
    int np = current_player;
    for (int off = 1; off <= num_players; off++) {
        int p = (current_player + off) % num_players;
        if (!finished_final.count(p)) { np = p; break; }
    }
    ns.current_player = np;

    return ns;
}

// ======================== Payoff Computation ========================

std::array<float,NUM_PLAYERS> compute_payoffs(const GameState& st) {
    if (st.is_declared && st.declarant >= 0)
        return compute_declared_payoffs(st.rankings, st.declarant);
    std::array<float,NUM_PLAYERS> payoffs = {0,0,0,0};
    int rank_points[] = {3, 2, 1, 0};
    if (st.is_solo) {
        int solo_idx = -1;
        for (int i = 0; i < st.num_players; i++)
            if (st.actual_teams[i] == TEAM_SOLO) { solo_idx = i; break; }
        if (solo_idx < 0) return payoffs;
        int solo_rank = st.num_players - 1;
        for (int i = 0; i < (int)st.rankings.size(); i++)
            if (st.rankings[i] == solo_idx) { solo_rank = i; break; }
        float table[] = {2.0f, 1.0f, -1.0f, -2.0f};
        payoffs[solo_idx] = table[std::min(solo_rank, 3)];
        for (int i = 0; i < st.num_players; i++)
            if (i != solo_idx) payoffs[i] = -payoffs[solo_idx] / 3.0f;
        return payoffs;
    }
    std::vector<int> team_a, team_b;
    for (int i = 0; i < st.num_players; i++) {
        if (st.actual_teams[i] == TEAM_SPADE_A3) team_a.push_back(i);
        else if (st.actual_teams[i] == TEAM_OPPONENT) team_b.push_back(i);
    }
    auto team_avg = [&](const std::vector<int>& members) -> float {
        if (members.empty()) return 0.0f;
        float total = 0;
        for (int m : members) {
            int rank = st.num_players - 1;
            for (int i = 0; i < (int)st.rankings.size(); i++)
                if (st.rankings[i] == m) { rank = i; break; }
            total += rank_points[std::min(rank, 3)];
        }
        return total / members.size();
    };
    float diff = team_avg(team_a) - team_avg(team_b);
    for (int i : team_a) payoffs[i] = diff;
    for (int i : team_b) payoffs[i] = -diff;
    return payoffs;
}

std::array<float,NUM_PLAYERS> compute_declared_payoffs(
    const std::vector<int>& rankings, int declarant)
{
    std::array<float,NUM_PLAYERS> payoffs = {0,0,0,0};
    int declared_rank = NUM_PLAYERS - 1;
    for (int i = 0; i < (int)rankings.size(); i++)
        if (rankings[i] == declarant) { declared_rank = i; break; }
    if (declared_rank == 0) {
        payoffs[declarant] = 4.0f;
        for (int i = 0; i < NUM_PLAYERS; i++)
            if (i != declarant) payoffs[i] = -4.0f / 3.0f;
    } else {
        payoffs[declarant] = -4.0f;
        for (int i = 0; i < NUM_PLAYERS; i++)
            if (i != declarant) payoffs[i] = 4.0f / 3.0f;
    }
    return payoffs;
}

std::array<float,NUM_PLAYERS> compute_training_payoffs(const GameState& st) {
    if (st.is_declared && st.declarant >= 0)
        return compute_declared_payoffs(st.rankings, st.declarant);
    auto payoffs = compute_payoffs(st);
    const float rank_bonus[] = {0.3f, 0.1f, -0.1f, -0.3f};
    for (int i = 0; i < st.num_players; i++) {
        int rank = st.num_players - 1;
        for (int j = 0; j < (int)st.rankings.size(); j++)
            if (st.rankings[j] == i) { rank = j; break; }
        payoffs[i] += rank_bonus[std::min(rank, 3)];
    }
    return payoffs;
}

// ======================== Step Reward ========================

static std::pair<int, float> get_teammate_confidence(
    int pid, const Team actual[], const Team observed[], int np)
{
    Team my_team = actual[pid];
    if (my_team == TEAM_SOLO) return {-1, 0.0f};
    int actual_teammate = -1;
    for (int i = 0; i < np; i++)
        if (i != pid && actual[i] == my_team) { actual_teammate = i; break; }
    if (actual_teammate < 0) return {-1, 0.0f};
    if (my_team == TEAM_SPADE_A3) {
        if (observed[actual_teammate] == TEAM_SPADE_A3) return {actual_teammate, 1.0f};
        return {actual_teammate, 0.0f};
    }
    int revealed = 0;
    for (int i = 0; i < np; i++) {
        if (i != pid && (observed[i] == TEAM_SPADE_A3 || observed[i] == TEAM_SOLO))
            revealed++;
    }
    if (revealed >= 2) return {actual_teammate, 1.0f};
    if (revealed == 1) return {actual_teammate, 0.5f};
    return {actual_teammate, 0.0f};
}

static constexpr float REWARD_LET_TEAMMATE  =  0.015f;
static constexpr float REWARD_BEAT_TEAMMATE = -0.01f;
static constexpr float REWARD_FEED_TEAMMATE =  0.03f;

float compute_step_reward(
    const GameState& prev, const HandInfo& action,
    const GameState& next, int player_id)
{
    if (prev.is_declaration_phase) return 0.0f;
    if (prev.is_declared) {
        if (player_id == prev.declarant) return 0.0f;
        auto is_teammate = [&](int other) {
            return other >= 0 && other < prev.num_players
                && other != player_id && other != prev.declarant;
        };
        float reward = 0.0f;
        if (is_teammate(prev.last_play_player)) {
            if (action.is_pass()) reward += REWARD_LET_TEAMMATE;
            if (action.is_play()) reward += REWARD_BEAT_TEAMMATE;
        }
        if (action.is_play() && next.hands[player_id] == 0
            && !next.is_terminal() && is_teammate(next.current_player))
            reward += REWARD_FEED_TEAMMATE;
        return reward;
    }
    auto [teammate, confidence] = get_teammate_confidence(
        player_id, prev.actual_teams, prev.observed_teams, prev.num_players);
    if (confidence <= 0.0f || teammate < 0) return 0.0f;
    float reward = 0.0f;
    int last_player = prev.last_play_player;
    if (action.is_pass() && last_player == teammate)
        reward += REWARD_LET_TEAMMATE;
    if (action.is_play() && last_player == teammate)
        reward += REWARD_BEAT_TEAMMATE;
    if (action.is_play() && next.hands[player_id] == 0
        && next.current_player == teammate)
        reward += REWARD_FEED_TEAMMATE;
    return reward * confidence;
}

// ======================== Afterstate Computation ========================

AfterstateInfo compute_afterstate(CardSet hand_after, int start_val, int end_val) {
    AfterstateInfo info;
    std::memset(&info, 0, sizeof(info));
    info.remaining_count = popcount64(hand_after);

    int rank_count[NUM_RANKS] = {};
    for_each_card(hand_after, [&](int c) {
        rank_count[card_rank(c)]++;
    });

    for (int r = 0; r < NUM_RANKS; r++) {
        if (rank_count[r] == 1) info.singles_count++;
        else if (rank_count[r] == 2) info.pairs_count++;
        else if (rank_count[r] == 3) info.triples_count++;
    }

    int spade3 = make_card(3, 12);
    int spadeA = make_card(3, 10);
    info.has_s3 = (hand_after & card_bit(spade3)) != 0;
    info.has_sa = (hand_after & card_bit(spadeA)) != 0;

    // rank 12 = '3' (strongest single), rank 11 = '2' (second strongest)
    info.has_rank3_single = (rank_count[12] == 1);
    info.has_rank2_single = (rank_count[11] == 1);

    // Straight potential: at least 5 consecutive ranks present
    for (int si = 0; si < 8; si++) {
        int seq_min = 99, seq_max = 0;
        bool has_all = true;
        for (int j = 0; j < 5; j++) {
            int v = STRAIGHT_RANK_VAL[STRAIGHT_SEQS[si][j]];
            seq_min = std::min(seq_min, v);
            seq_max = std::max(seq_max, v);
            if (rank_count[STRAIGHT_SEQS[si][j]] == 0) { has_all = false; break; }
        }
        if (has_all && seq_min >= start_val && seq_max <= end_val) {
            info.has_straight_potential = true;
            break;
        }
    }

    // Flush potential: 5+ cards of same suit
    for (int s = 0; s < NUM_SUITS; s++) {
        int suit_cnt = popcount64(hand_after & suit_mask(s));
        if (suit_cnt >= 5) {
            info.has_flush_potential = true;
            if (info.has_straight_potential) {
                // Check straight flush potential within this suit
                CardSet sc = hand_after & suit_mask(s);
                for (int si = 0; si < 8; si++) {
                    int seq_min = 99, seq_max = 0;
                    bool ok = true;
                    for (int j = 0; j < 5; j++) {
                        int v = STRAIGHT_RANK_VAL[STRAIGHT_SEQS[si][j]];
                        seq_min = std::min(seq_min, v);
                        seq_max = std::max(seq_max, v);
                        if (!(sc & rank_mask(STRAIGHT_SEQS[si][j]))) { ok = false; break; }
                    }
                    if (ok && seq_min >= start_val && seq_max <= end_val) {
                        info.has_sf_potential = true;
                        break;
                    }
                }
            }
            if (info.has_sf_potential) break;
        }
    }

    // Three+pair or four+one potential
    int quads = 0;
    int pair_ranks = 0, triple_ranks = 0;
    for (int r = 0; r < NUM_RANKS; r++) {
        if (rank_count[r] >= 4) quads++;
        if (rank_count[r] >= 2) pair_ranks++;
        if (rank_count[r] >= 3) triple_ranks++;
    }
    info.has_threepair_or_fourone_potential =
        (triple_ranks > 0 && pair_ranks >= 2) ||
        (quads > 0 && info.remaining_count >= 5);

    // Approximate 5-card action count
    int fc = 0;
    if (info.has_straight_potential) fc++;
    if (info.has_flush_potential) fc++;
    if (info.has_sf_potential) fc++;
    if (info.has_threepair_or_fourone_potential) fc++;
    info.fivecard_potential = fc;

    // Rank-group heuristic only; this is not an optimal hand decomposition.
    int steps = info.singles_count + info.pairs_count + info.triples_count + quads;
    info.min_steps = std::max(steps, info.remaining_count > 0 ? 1 : 0);

    return info;
}

// ======================== Engine ========================

Engine::Engine() : rng_(42) {
    std::memset(played_cards_, 0, sizeof(played_cards_));
    std::memset(seat_agents_, 0, sizeof(seat_agents_));
    last_nonpass_player_ = -1;
}

void Engine::seed(unsigned int s) { rng_.seed(s); }

int Engine::reset() {
    action_history_.clear();
    std::memset(played_cards_, 0, sizeof(played_cards_));
    for (int i = 0; i < NUM_PLAYERS; i++) step_rewards_[i].clear();
    key_to_hand_.clear();
    last_nonpass_action_ = {};
    last_nonpass_player_ = -1;

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int seat = 0; seat < NUM_PLAYERS; seat++) {
        double roll = dist(rng_);
        if (roll < greedy_ratio_)
            seat_agents_[seat] = SEAT_GREEDY;
        else if (roll < greedy_ratio_ + random_ratio_)
            seat_agents_[seat] = SEAT_RANDOM;
        else
            seat_agents_[seat] = SEAT_RL;
    }

    std::vector<int> deck(NUM_CARDS);
    std::iota(deck.begin(), deck.end(), 0);
    std::shuffle(deck.begin(), deck.end(), rng_);

    for (int i = 0; i < NUM_PLAYERS; i++) state_.hands[i] = 0;
    for (int i = 0; i < NUM_CARDS; i++)
        state_.hands[i / 13] |= card_bit(deck[i]);

    int diamond4 = make_card(0, 0);
    int start = 0;
    for (int i = 0; i < NUM_PLAYERS; i++)
        if (state_.hands[i] & card_bit(diamond4)) { start = i; break; }

    assign_teams(state_.hands, state_.actual_teams, state_.is_solo);

    state_.current_player = start;
    state_.last_play = {}; state_.last_play_player = -1;
    state_.pass_count = 0; state_.is_first_turn = true;
    state_.rankings.clear();
    state_.num_players = NUM_PLAYERS;
    state_.spade3_player = -1; state_.spadeA_player = -1;

    std::uniform_int_distribution<int> bin(0, 1);
    state_.declare_require_both_spades = bin(rng_);
    state_.straight_start_val = bin(rng_) ? 2 : 1;
    state_.straight_end_val   = bin(rng_) ? 12 : 11;

    state_.is_declaration_phase = true;
    state_.declaration_turn = 0;
    state_.declaration_passes = 0;
    state_.is_declared = false; state_.declarant = -1;
    state_.compute_observed_teams();

    prev_state_ = state_;
    return state_.declaration_turn;
}

void Engine::set_rules(bool declare_require_both_spades, int straight_start_val, int straight_end_val) {
    state_.declare_require_both_spades = declare_require_both_spades;
    state_.straight_start_val = straight_start_val;
    state_.straight_end_val = straight_end_val;
    key_to_hand_.clear();
}

int Engine::reset_with_hands(
    const std::vector<std::vector<std::string>>& hands_ids,
    int start_player)
{
    action_history_.clear();
    std::memset(played_cards_, 0, sizeof(played_cards_));
    for (int i = 0; i < NUM_PLAYERS; i++) step_rewards_[i].clear();
    key_to_hand_.clear();
    last_nonpass_action_ = {};
    last_nonpass_player_ = -1;

    for (int i = 0; i < NUM_PLAYERS; i++) seat_agents_[i] = SEAT_RL;

    for (int i = 0; i < NUM_PLAYERS; i++) {
        state_.hands[i] = 0;
        for (auto& cid : hands_ids[i]) {
            int c = string_to_card(cid);
            if (c >= 0) state_.hands[i] |= card_bit(c);
        }
    }

    assign_teams(state_.hands, state_.actual_teams, state_.is_solo);

    state_.current_player = start_player;
    state_.last_play = {}; state_.last_play_player = -1;
    state_.pass_count = 0; state_.is_first_turn = true;
    state_.rankings.clear();
    state_.num_players = NUM_PLAYERS;
    state_.spade3_player = -1; state_.spadeA_player = -1;
    state_.declare_require_both_spades = false;
    state_.straight_start_val = 1;
    state_.straight_end_val = 12;
    state_.is_declaration_phase = true;
    state_.declaration_turn = 0;
    state_.declaration_passes = 0;
    state_.is_declared = false; state_.declarant = -1;
    state_.compute_observed_teams();

    prev_state_ = state_;
    return state_.declaration_turn;
}

HandInfo Engine::resolve_action(const std::string& key) const {
    if (key == "pass") return {};
    if (key == "declare") { HandInfo h; h.type = HAND_DECLARE; return h; }
    auto it = key_to_hand_.find(key);
    if (it != key_to_hand_.end()) return it->second;
    CardSet cs = 0;
    std::istringstream iss(key);
    std::string token;
    while (std::getline(iss, token, '|')) {
        int c = string_to_card(token);
        if (c >= 0) cs |= card_bit(c);
    }
    int sz = popcount64(cs);
    if (sz == 0) return {};
    HandInfo hand = detect_hand(cs, sz);
    // Legal move enumeration treats an out-of-range straight flush as a
    // flush. Uncached keys must use the same room-rule interpretation.
    if (hand.type == HAND_STRAIGHT_FLUSH
        && !check_straight_ranks(cs, state_.straight_start_val, state_.straight_end_val)) {
        hand.type = HAND_FLUSH;
        hand.primary_card = find_max_score_card(cs);
    }
    return hand;
}

int Engine::step(const std::string& action_key) {
    prev_state_ = state_;
    int player_id = get_player_id();

    HandInfo action = resolve_action(action_key);

    if (!state_.is_declaration_phase) {
        action_history_.push_back({player_id, action});
        if (action.is_play()) {
            played_cards_[player_id] |= action.cards;
            last_nonpass_action_ = action;
            last_nonpass_player_ = player_id;
        }
    }

    state_ = state_.apply_move(action);

    float sr = compute_step_reward(prev_state_, action, state_, player_id);
    step_rewards_[player_id].push_back(sr);

    return get_player_id();
}

int Engine::get_player_id() const {
    if (state_.is_declaration_phase) return state_.declaration_turn;
    return state_.current_player;
}

bool Engine::is_over() const { return state_.is_terminal(); }
bool Engine::is_declaration_phase() const { return state_.is_declaration_phase; }
bool Engine::is_rule_agent_seat(int pid) const {
    return seat_agents_[pid] != SEAT_RL;
}

std::array<float,NUM_PLAYERS> Engine::get_payoffs() const {
    return compute_payoffs(state_);
}
std::array<float,NUM_PLAYERS> Engine::get_training_payoffs() const {
    return compute_training_payoffs(state_);
}
const std::vector<float>& Engine::get_step_rewards(int p) const {
    return step_rewards_[p];
}

// ======================== V11 Aux Targets ========================

std::array<Engine::AuxTargets,NUM_PLAYERS> Engine::get_aux_targets() const {
    std::array<AuxTargets, NUM_PLAYERS> all;
    const Team* actual = state_.actual_teams;
    const Team* observed = state_.observed_teams;

    for (int p = 0; p < NUM_PLAYERS; p++) {
        int rel_order[NUM_PLAYERS];
        for (int i = 0; i < NUM_PLAYERS; i++)
            rel_order[i] = (p + i) % NUM_PLAYERS;

        Team my_team = actual[p];

        // relation: for each of 3 opponents
        // target values align with obs encoding: 1=same_side, 2=opposite_side, -1=unknown
        for (int j = 0; j < 3; j++) {
            int other = rel_order[j + 1];
            if (observed[other] == TEAM_UNKNOWN) {
                all[p].relation[j] = -1;
            } else if (state_.is_declared) {
                if (p == state_.declarant)
                    all[p].relation[j] = 2; // opposite
                else if (other == state_.declarant)
                    all[p].relation[j] = 2; // opposite
                else
                    all[p].relation[j] = 1; // same (both anti-declarant)
            } else {
                bool same = (my_team == observed[other]);
                all[p].relation[j] = same ? 1 : 2;
            }
        }

        // s3_owner: relative slot of spade3 holder (-1 if not publicly known)
        all[p].s3_owner = -1;
        if (state_.spade3_player >= 0) {
            for (int i = 0; i < NUM_PLAYERS; i++) {
                if (rel_order[i] == state_.spade3_player) {
                    all[p].s3_owner = i;
                    break;
                }
            }
        }

        // sa_owner: relative slot of spadeA holder
        all[p].sa_owner = -1;
        if (state_.spadeA_player >= 0) {
            for (int i = 0; i < NUM_PLAYERS; i++) {
                if (rel_order[i] == state_.spadeA_player) {
                    all[p].sa_owner = i;
                    break;
                }
            }
        }
    }
    return all;
}

// ======================== V11 State Encoding ========================

void Engine::encode_obs(int player_id, int8_t* out) const {
    std::memset(out, 0, STATE_DIM);
    encode_obs_static(player_id, out);
    encode_hist_tokens(player_id, out + OBS_STATIC_DIM);
}

void Engine::encode_obs_static(int player_id, int8_t* out) const {
    int8_t* ptr = out;
    int n = NUM_PLAYERS;

    int rel_order[NUM_PLAYERS];
    for (int i = 0; i < n; i++) rel_order[i] = (player_id + i) % n;

    int spade3 = make_card(3, 12);
    int spadeA = make_card(3, 10);

    // [0:52] own_cards
    for_each_card(state_.hands[player_id], [&](int c){ ptr[c] = 1; });
    ptr += NUM_CARDS;

    // [52:104] current_top_action (last play to beat, 0 if free play)
    if (state_.last_play.is_play())
        for_each_card(state_.last_play.cards, [&](int c){ ptr[c] = 1; });
    ptr += NUM_CARDS;

    // [104:108] current_top_actor
    if (state_.last_play_player >= 0 && state_.last_play.is_play()) {
        for (int i = 0; i < n; i++)
            if (rel_order[i] == state_.last_play_player) { ptr[i] = 1; break; }
    }
    ptr += n;

    // [108:160] last_nonpass_action
    if (last_nonpass_action_.is_play())
        for_each_card(last_nonpass_action_.cards, [&](int c){ ptr[c] = 1; });
    ptr += NUM_CARDS;

    // [160:164] last_nonpass_actor
    if (last_nonpass_player_ >= 0) {
        for (int i = 0; i < n; i++)
            if (rel_order[i] == last_nonpass_player_) { ptr[i] = 1; break; }
    }
    ptr += n;

    // [164:372] public_played_4p (4×52D)
    for (int i = 0; i < n; i++) {
        int abs_i = rel_order[i];
        for_each_card(played_cards_[abs_i], [&](int c){ ptr[c] = 1; });
        ptr += NUM_CARDS;
    }

    // [372:428] remaining_count_4p (4×14D)
    for (int i = 0; i < n; i++) {
        int abs_i = rel_order[i];
        int cnt = std::min(popcount64(state_.hands[abs_i]), 13);
        ptr[cnt] = 1;
        ptr += 14;
    }

    // [428:444] public_status_4p (4×4D: played_s3, played_sa, declared, finished)
    std::set<int> finished_set(state_.rankings.begin(), state_.rankings.end());
    for (int i = 0; i < n; i++) {
        int abs_i = rel_order[i];
        ptr[0] = (played_cards_[abs_i] & card_bit(spade3)) ? 1 : 0;
        ptr[1] = (played_cards_[abs_i] & card_bit(spadeA)) ? 1 : 0;
        ptr[2] = (state_.is_declared && abs_i == state_.declarant) ? 1 : 0;
        ptr[3] = finished_set.count(abs_i) ? 1 : 0;
        ptr += 4;
    }

    // [444:447] game_mode_state (3D one-hot: normal, solo, declarer)
    if (state_.is_declared) {
        ptr[2] = 1;
    } else if (state_.is_solo && state_.actual_teams[player_id] == TEAM_SOLO) {
        ptr[1] = 1; // I know I'm solo
    } else {
        ptr[0] = 1; // normal team mode (from my perspective)
    }
    ptr += 3;

    // [447:452] self_side_state (5D: unknown, a3_side, non_a3, declarer_side, anti_declarer)
    Team my_actual = state_.actual_teams[player_id];
    if (state_.is_declared) {
        if (player_id == state_.declarant)
            ptr[3] = 1; // declarer_side
        else
            ptr[4] = 1; // anti_declarer_side
    } else if (my_actual == TEAM_SPADE_A3 || my_actual == TEAM_SOLO) {
        ptr[1] = 1; // a3_side
    } else if (my_actual == TEAM_OPPONENT) {
        ptr[2] = 1; // non_a3_side
    } else {
        ptr[0] = 1; // unknown (shouldn't happen)
    }
    ptr += 5;

    // [452:461] other_relation_to_self (3×3D: unknown, same, opposite)
    for (int j = 0; j < 3; j++) {
        int other = rel_order[j + 1];
        Team other_obs = state_.observed_teams[other];

        if (other_obs == TEAM_UNKNOWN) {
            ptr[0] = 1; // unknown
        } else if (state_.is_declared) {
            if (player_id == state_.declarant) {
                ptr[2] = 1; // opposite (I'm declarant, everyone else is against me)
            } else if (other == state_.declarant) {
                ptr[2] = 1; // opposite (other is declarant)
            } else {
                ptr[1] = 1; // same (both anti-declarant)
            }
        } else {
            bool same_side = (my_actual == other_obs);
            if (same_side)
                ptr[1] = 1; // same_side
            else
                ptr[2] = 1; // opposite_side
        }
        ptr += 3;
    }

    // [461:513] unseen_cards (52D)
    CardSet my_hand = state_.hands[player_id];
    CardSet all_played = 0;
    for (int i = 0; i < n; i++) all_played |= played_cards_[i];
    CardSet unseen = ~(my_hand | all_played) & ((1ULL << NUM_CARDS) - 1);
    for_each_card(unseen, [&](int c){ ptr[c] = 1; });
    ptr += NUM_CARDS;

    // [513:525] s3_sa_tracker (12D)
    *ptr++ = (my_hand & card_bit(spade3)) ? 1 : 0;
    *ptr++ = (my_hand & card_bit(spadeA)) ? 1 : 0;
    *ptr++ = (all_played & card_bit(spade3)) ? 1 : 0;
    *ptr++ = (all_played & card_bit(spadeA)) ? 1 : 0;
    for (int i = 0; i < n; i++)
        *ptr++ = (played_cards_[rel_order[i]] & card_bit(spade3)) ? 1 : 0;
    for (int i = 0; i < n; i++)
        *ptr++ = (played_cards_[rel_order[i]] & card_bit(spadeA)) ? 1 : 0;

    // [525:537] phase_flags (12D)
    bool has_last = state_.last_play.is_play();
    bool is_free = !has_last && !state_.is_first_turn && !state_.is_declaration_phase;
    CardSet my_hand_pf = state_.hands[player_id];
    int my_cnt_pf = popcount64(my_hand_pf);
    bool must_play = (my_cnt_pf == 1 && has_last);

    *ptr++ = state_.is_declaration_phase ? 1 : 0;  // [0] is_declaration_phase
    *ptr++ = state_.is_first_turn ? 1 : 0;          // [1] is_first_play_of_game
    *ptr++ = is_free ? 1 : 0;                       // [2] is_free_lead
    *ptr++ = (has_last && !must_play) ? 1 : 0;      // [3] can_pass
    *ptr++ = must_play ? 1 : 0;                     // [4] must_play_if_possible
    *ptr++ = has_last ? 1 : 0;                       // [5] has_current_top
    // top_is_single / pair / triple / 5card
    *ptr++ = (has_last && state_.last_play.size == 1) ? 1 : 0;
    *ptr++ = (has_last && state_.last_play.size == 2) ? 1 : 0;
    *ptr++ = (has_last && state_.last_play.size == 3) ? 1 : 0;
    *ptr++ = (has_last && state_.last_play.size == 5) ? 1 : 0;
    // pass_count_eq_1 / eq_2
    *ptr++ = (state_.pass_count == 1) ? 1 : 0;
    *ptr++ = (state_.pass_count == 2) ? 1 : 0;

    // [537:544] room_rules (7D)
    *ptr++ = (state_.straight_start_val <= 1) ? 1 : 0;  // start is 3
    *ptr++ = (state_.straight_start_val >= 2) ? 1 : 0;  // start is 4
    *ptr++ = (state_.straight_end_val <= 11) ? 1 : 0;   // end is K
    *ptr++ = (state_.straight_end_val >= 12) ? 1 : 0;   // end is A
    *ptr++ = 0; // declare_rule_none (reserved)
    *ptr++ = state_.declare_require_both_spades ? 0 : 1; // free declare
    *ptr++ = state_.declare_require_both_spades ? 1 : 0; // only s3+sA

    // [544:556] threat_flags (12D)
    for (int j = 1; j <= 3; j++) {
        int abs_j = rel_order[j];
        int cnt_j = popcount64(state_.hands[abs_j]);
        *ptr++ = (cnt_j == 1) ? 1 : 0;  // eq_1
    }
    for (int j = 1; j <= 3; j++) {
        int abs_j = rel_order[j];
        int cnt_j = popcount64(state_.hands[abs_j]);
        *ptr++ = (cnt_j <= 2) ? 1 : 0;  // le_2
    }
    for (int j = 1; j <= 3; j++) {
        int abs_j = rel_order[j];
        *ptr++ = finished_set.count(abs_j) ? 1 : 0;  // finished
    }
    // any_other_eq_1
    bool any_eq1 = false, any_le2 = false;
    for (int j = 1; j <= 3; j++) {
        int cnt_j = popcount64(state_.hands[rel_order[j]]);
        if (cnt_j == 1) any_eq1 = true;
        if (cnt_j <= 2) any_le2 = true;
    }
    *ptr++ = any_eq1 ? 1 : 0;
    *ptr++ = any_le2 ? 1 : 0;
    *ptr++ = (my_cnt_pf == 1) ? 1 : 0;  // self_eq_1
}

// ======================== V11 History Encoding ========================

void Engine::encode_hist_tokens(int player_id, int8_t* out) const {
    int n = NUM_PLAYERS;
    int rel_order[NUM_PLAYERS];
    for (int i = 0; i < n; i++) rel_order[i] = (player_id + i) % n;

    int hist_start = std::max(0, (int)action_history_.size() - HISTORY_LEN);
    int hist_count = (int)action_history_.size() - hist_start;

    int8_t* ptr = out;
    for (int h = 0; h < hist_count; h++) {
        encode_hist_token(action_history_[hist_start + h], player_id, rel_order, ptr);
        ptr += HIST_TOKEN_DIM;
    }
    // Remaining slots are already zeroed (valid=0)
}

void Engine::encode_hist_token(const RichHistoryEntry& entry, int player_id,
                               const int* rel_order, int8_t* out) const {
    int8_t* ptr = out;

    // [0:4] actor (4D relative one-hot)
    for (int i = 0; i < NUM_PLAYERS; i++)
        if (rel_order[i] == entry.player_id) { ptr[i] = 1; break; }
    ptr += 4;

    // [4] valid
    *ptr++ = 1;

    const HandInfo& h = entry.hand;

    // [5] is_pass
    *ptr++ = h.is_pass() ? 1 : 0;

    // [6:15] action_type (9D one-hot)
    int ati = hand_type_to_action_idx(h.type);
    if (ati >= 0 && ati < NUM_ACTION_TYPES) ptr[ati] = 1;
    ptr += NUM_ACTION_TYPES;

    // [15:28] main_rank (13D one-hot)
    if (h.is_play() && h.primary_card >= 0)
        ptr[card_rank(h.primary_card)] = 1;
    ptr += NUM_RANKS;

    // [28:32] main_suit (4D one-hot)
    if (h.is_play() && h.primary_card >= 0) {
        bool encode_suit = (h.type == HAND_SINGLE || h.type == HAND_PAIR ||
                           h.type == HAND_STRAIGHT || h.type == HAND_FLUSH ||
                           h.type == HAND_STRAIGHT_FLUSH);
        if (encode_suit) ptr[card_suit(h.primary_card)] = 1;
    }
    ptr += NUM_SUITS;

    // [32:36] action_len (4D one-hot)
    if (h.is_play()) {
        int li = hand_size_to_len_idx(h.size);
        if (li >= 0) ptr[li] = 1;
    }
    ptr += 4;

    // [36:88] card_bits (52D)
    if (h.is_play())
        for_each_card(h.cards, [&](int c){ ptr[c] = 1; });
    // ptr += NUM_CARDS; // not needed, caller advances by HIST_TOKEN_DIM
}

// ======================== V11 Action Feature Encoding ========================

void Engine::encode_action_feature(const HandInfo& hand, int player_id,
                                   int8_t* out) const {
    std::memset(out, 0, ACTION_FEAT_DIM);
    int8_t* ptr = out;

    int spade3 = make_card(3, 12);
    int spadeA = make_card(3, 10);
    int diamond4 = make_card(0, 0);

    // [0:52] card_bits
    if (hand.is_declare()) {
        for (int i = 0; i < NUM_CARDS; i++) ptr[i] = 1;
    } else if (hand.is_play()) {
        for_each_card(hand.cards, [&](int c){ ptr[c] = 1; });
    }
    ptr += NUM_CARDS;

    // [52:61] action_type (9D one-hot)
    int ati = hand_type_to_action_idx(hand.type);
    if (ati >= 0 && ati < NUM_ACTION_TYPES) ptr[ati] = 1;
    ptr += NUM_ACTION_TYPES;

    // [61:74] main_rank (13D)
    if (hand.is_play() && hand.primary_card >= 0)
        ptr[card_rank(hand.primary_card)] = 1;
    ptr += NUM_RANKS;

    // [74:78] main_suit (4D)
    if (hand.is_play() && hand.primary_card >= 0) {
        bool encode_suit = (hand.type == HAND_SINGLE || hand.type == HAND_PAIR ||
                           hand.type == HAND_STRAIGHT || hand.type == HAND_FLUSH ||
                           hand.type == HAND_STRAIGHT_FLUSH);
        if (encode_suit) ptr[card_suit(hand.primary_card)] = 1;
    }
    ptr += NUM_SUITS;

    // [78:82] action_len (4D)
    if (hand.is_play()) {
        int li = hand_size_to_len_idx(hand.size);
        if (li >= 0) ptr[li] = 1;
    }
    ptr += 4;

    // [82:90] special_flags (8D)
    ptr[0] = hand.is_pass() ? 1 : 0;
    if (hand.is_play()) {
        ptr[1] = (hand.cards & card_bit(diamond4)) ? 1 : 0;
        bool has_s3 = (hand.cards & card_bit(spade3)) != 0;
        bool has_sa = (hand.cards & card_bit(spadeA)) != 0;
        ptr[2] = has_s3 ? 1 : 0;
        ptr[3] = has_sa ? 1 : 0;

        // reveals_self_identity: only if identity NOT already public
        bool s3_already_public = (state_.spade3_player >= 0);
        bool sa_already_public = (state_.spadeA_player >= 0);
        bool reveals = (has_s3 && !s3_already_public) || (has_sa && !sa_already_public);
        ptr[4] = reveals ? 1 : 0;

        CardSet hand_after = state_.hands[player_id] & ~hand.cards;
        int remaining = popcount64(hand_after);
        ptr[5] = (remaining == 0) ? 1 : 0;
        ptr[6] = (remaining == 1) ? 1 : 0;

        // forced_nonpass_context: must play (1 card left and can beat, or free lead)
        CardSet my = state_.hands[player_id];
        int my_cnt = popcount64(my);
        bool cant_pass = state_.last_play.is_pass()
                         || (my_cnt == 1 && !state_.last_play.is_pass());
        ptr[7] = cant_pass ? 1 : 0;
    }
    ptr += 8;

    // [90:106] afterstate_summary (16D)
    if (hand.is_play()) {
        CardSet hand_after = state_.hands[player_id] & ~hand.cards;
        AfterstateInfo as = compute_afterstate(hand_after,
            state_.straight_start_val, state_.straight_end_val);
        ptr[0] = (int8_t)std::min(as.remaining_count, 13);
        ptr[1] = (as.remaining_count == 1) ? 1 : 0;
        ptr[2] = (as.remaining_count == 2) ? 1 : 0;
        ptr[3] = (int8_t)std::min(as.singles_count, 13);
        ptr[4] = (int8_t)std::min(as.pairs_count, 6);
        ptr[5] = (int8_t)std::min(as.triples_count, 4);
        ptr[6] = (int8_t)std::min(as.fivecard_potential, 10);
        ptr[7] = (int8_t)std::min(as.min_steps, 13);
        ptr[8] = as.has_s3 ? 1 : 0;
        ptr[9] = as.has_sa ? 1 : 0;
        ptr[10] = as.has_rank3_single ? 1 : 0;
        ptr[11] = as.has_rank2_single ? 1 : 0;
        ptr[12] = as.has_straight_potential ? 1 : 0;
        ptr[13] = as.has_flush_potential ? 1 : 0;
        ptr[14] = as.has_sf_potential ? 1 : 0;
        ptr[15] = as.has_threepair_or_fourone_potential ? 1 : 0;
    }
    ptr += 16;

    // [106:111] rule_aware_straight_features (5D)
    bool is_straight_fam = (hand.type == HAND_STRAIGHT || hand.type == HAND_STRAIGHT_FLUSH);
    ptr[0] = is_straight_fam ? 1 : 0;
    if (is_straight_fam && hand.is_play()) {
        int min_srv = 99, max_srv = 0;
        for_each_card(hand.cards, [&](int c) {
            int v = STRAIGHT_RANK_VAL[card_rank(c)];
            if (v < min_srv) min_srv = v;
            if (v > max_srv) max_srv = v;
        });
        int room_min = state_.straight_start_val;
        int room_max = state_.straight_end_val;

        ptr[1] = (min_srv == room_min) ? 1 : 0; // touches min boundary
        ptr[2] = (max_srv == room_max) ? 1 : 0; // touches max boundary
        ptr[3] = (max_srv == room_max && min_srv == room_max - 4) ? 1 : 0; // room max straight
        ptr[4] = (min_srv == room_min && max_srv == room_min + 4) ? 1 : 0; // room min straight
    }
}

// ======================== Legal Actions (V11) ========================

std::vector<Engine::ActionEntry> Engine::get_legal_actions() const {
    std::vector<ActionEntry> result;
    int pid = get_player_id();

    auto moves = state_.get_legal_moves();
    for (auto& h : moves) {
        ActionEntry ae;
        ae.key = h.to_key();
        encode_action_feature(h, pid, ae.feature);
        key_to_hand_[ae.key] = h;
        result.push_back(ae);
    }
    return result;
}

void Engine::get_action_feature(const std::string& key, int8_t* out) const {
    int pid = get_player_id();
    if (key == "declare") {
        HandInfo decl; decl.type = HAND_DECLARE;
        encode_action_feature(decl, pid, out);
        return;
    }
    if (key == "pass" || key.empty()) {
        HandInfo pass;
        encode_action_feature(pass, pid, out);
        return;
    }
    auto it = key_to_hand_.find(key);
    if (it != key_to_hand_.end()) {
        encode_action_feature(it->second, pid, out);
        return;
    }
    HandInfo h = resolve_action(key);
    encode_action_feature(h, pid, out);
}

// ======================== Rule Agent ========================

std::string Engine::greedy_action() const {
    if (state_.is_declaration_phase) return "pass";

    auto moves = state_.get_legal_moves();
    std::vector<HandInfo> plays;
    for (auto& h : moves)
        if (h.is_play()) plays.push_back(h);
    if (plays.empty()) return "pass";

    int pid = state_.current_player;
    CardSet my = state_.hands[pid];
    int my_count = popcount64(my);

    if (state_.last_play.is_pass()) {
        if (my_count == 1) return plays[0].to_key();
        if (my_count == 2) {
            for (auto& h : plays)
                if (h.size == 2) return h.to_key();
        }
        std::sort(plays.begin(), plays.end(), [](const HandInfo& a, const HandInfo& b){
            return card_score(a.primary_card) < card_score(b.primary_card);
        });
        for (auto& h : plays)
            if (h.size == 1) return h.to_key();
        return plays[0].to_key();
    }

    int lpp = state_.last_play_player;
    Team my_team = state_.observed_teams[pid];
    Team opp_team = (lpp >= 0) ? state_.observed_teams[lpp] : TEAM_UNKNOWN;
    bool is_teammate = (my_team != TEAM_UNKNOWN && opp_team != TEAM_UNKNOWN
                        && my_team == opp_team);
    if (is_teammate && std::any_of(moves.begin(), moves.end(),
                                  [](const HandInfo& hand) { return hand.is_pass(); }))
        return "pass";

    std::sort(plays.begin(), plays.end(), [](const HandInfo& a, const HandInfo& b){
        return card_score(a.primary_card) < card_score(b.primary_card);
    });
    return plays[0].to_key();
}

std::string Engine::random_action() const {
    auto moves = state_.get_legal_moves();
    if (moves.empty()) return "pass";
    std::uniform_int_distribution<int> dist(0, (int)moves.size() - 1);
    return moves[dist(rng_)].to_key();
}

std::string Engine::get_rule_agent_action() const {
    int pid = get_player_id();
    if (seat_agents_[pid] == SEAT_GREEDY) return greedy_action();
    if (seat_agents_[pid] == SEAT_RANDOM) return random_action();
    return "";
}

// ======================== VectorizedEngine ========================

VectorizedEngine::VectorizedEngine(int n) : engines_(n), n_(n) {}

void VectorizedEngine::set_greedy_ratio(double r) {
    for (auto& e : engines_) e.set_greedy_ratio(r);
}
void VectorizedEngine::set_random_ratio(double r) {
    for (auto& e : engines_) e.set_random_ratio(r);
}
void VectorizedEngine::seed(unsigned int base) {
    for (int i = 0; i < n_; i++) engines_[i].seed(base + i);
}

int  VectorizedEngine::reset(int i)                          { return engines_[i].reset(); }
void VectorizedEngine::set_rules(int i, bool both, int start, int end) {
    engines_[i].set_rules(both, start, end);
}
int  VectorizedEngine::step(int i, const std::string& k)     { return engines_[i].step(k); }
int  VectorizedEngine::get_player_id(int i)           const  { return engines_[i].get_player_id(); }
bool VectorizedEngine::is_over(int i)                 const  { return engines_[i].is_over(); }
bool VectorizedEngine::is_rule_agent_seat(int i, int p) const { return engines_[i].is_rule_agent_seat(p); }
bool VectorizedEngine::is_declaration_phase(int i)    const  { return engines_[i].is_declaration_phase(); }
std::string VectorizedEngine::get_rule_agent_action(int i) const { return engines_[i].get_rule_agent_action(); }

void VectorizedEngine::encode_obs(int i, int pid, int8_t* out) const {
    engines_[i].encode_obs(pid, out);
}
void VectorizedEngine::get_action_feature(int i, const std::string& k, int8_t* out) const {
    engines_[i].get_action_feature(k, out);
}

std::array<float, NUM_PLAYERS> VectorizedEngine::get_payoffs(int i) const {
    return engines_[i].get_payoffs();
}
std::array<float, NUM_PLAYERS> VectorizedEngine::get_training_payoffs(int i) const {
    return engines_[i].get_training_payoffs();
}
const std::vector<float>& VectorizedEngine::get_step_rewards(int i, int p) const {
    return engines_[i].get_step_rewards(p);
}
Engine::AuxTargets VectorizedEngine::get_aux_targets_for_player(int idx, int pid) const {
    return engines_[idx].get_aux_targets()[pid];
}
std::array<Engine::AuxTargets,NUM_PLAYERS> VectorizedEngine::get_aux_targets(int i) const {
    return engines_[i].get_aux_targets();
}

std::pair<bool, std::vector<VectorizedEngine::RuleStepData>>
VectorizedEngine::advance_to_decision(int idx) {
    std::vector<RuleStepData> steps;
    while (!engines_[idx].is_over()) {
        int pid = engines_[idx].get_player_id();
        if (!engines_[idx].is_rule_agent_seat(pid)) break;

        RuleStepData sd;
        sd.player_id = pid;
        engines_[idx].encode_obs(pid, sd.obs);

        std::string key = engines_[idx].get_rule_agent_action();
        engines_[idx].get_action_feature(key, sd.action);
        sd.auxiliary = engines_[idx].get_aux_targets()[pid];
        engines_[idx].step(key);
        steps.push_back(sd);
    }
    return {engines_[idx].is_over(), std::move(steps)};
}

int VectorizedEngine::step_random(int idx) {
    if (engines_[idx].is_over()) return engines_[idx].get_player_id();
    auto actions = engines_[idx].get_legal_actions();
    if (actions.empty()) return engines_[idx].get_player_id();
    int choice = std::rand() % (int)actions.size();
    return engines_[idx].step(actions[choice].key);
}

VectorizedEngine::BatchData
VectorizedEngine::prepare_batch(const std::vector<int>& pending) const {
    BatchData bd;
    int K = (int)pending.size();
    bd.offsets.reserve(K + 1);
    bd.offsets.push_back(0);
    bd.obs_raw.resize(K * STATE_DIM);
    bd.action_keys.reserve(K);

    int est = K * 20;
    bd.action_flat.reserve(est * ACTION_DIM);

    for (int idx = 0; idx < K; idx++) {
        int e = pending[idx];
        int pid = engines_[e].get_player_id();

        int8_t* obs_ptr = bd.obs_raw.data() + idx * STATE_DIM;
        engines_[e].encode_obs(pid, obs_ptr);

        auto actions = engines_[e].get_legal_actions();
        std::vector<std::string> keys;
        keys.reserve(actions.size());

        for (auto& a : actions) {
            bd.action_flat.insert(bd.action_flat.end(),
                                  a.feature, a.feature + ACTION_DIM);
            keys.push_back(std::move(a.key));
        }

        bd.total_actions += (int)actions.size();
        bd.offsets.push_back(bd.total_actions);
        bd.action_keys.push_back(std::move(keys));
    }
    return bd;
}

} // namespace a3dizhu

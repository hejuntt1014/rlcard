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

// 8 possible straight sequences (ranks that form consecutive straight values)
// straight_rank_value 1-5:  3(12),4(0),5(1),6(2),7(3)
// straight_rank_value 2-6:  4(0),5(1),6(2),7(3),8(4)
// ...
// straight_rank_value 8-12: 10(6),J(7),Q(8),K(9),A(10)
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

void HandInfo::to_feature(int8_t* out) const {
    std::memset(out, 0, ACTION_DIM);
    if (type == HAND_DECLARE) {
        std::memset(out, 1, ACTION_DIM); // all-ones
        return;
    }
    for_each_card(cards, [&](int c){ out[c] = 1; });
}

void Engine::cardset_to_feature(CardSet cs, int8_t* out) {
    std::memset(out, 0, ACTION_DIM);
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

static bool check_straight_ranks(CardSet cs) {
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
    auto v = cards_vec(cs);
    for (size_t i = 0; i < v.size(); i++)
        for (size_t j = i+1; j < v.size(); j++)
            fn(card_bit(v[i]) | card_bit(v[j]));
}

static void enum_3subsets(CardSet cs, const EnumFn& fn) {
    auto v = cards_vec(cs);
    for (size_t i = 0; i < v.size(); i++)
        for (size_t j = i+1; j < v.size(); j++)
            for (size_t k = j+1; k < v.size(); k++)
                fn(card_bit(v[i]) | card_bit(v[j]) | card_bit(v[k]));
}

static void enum_5subsets(CardSet cs, const EnumFn& fn) {
    auto v = cards_vec(cs);
    int n = (int)v.size();
    for (int a=0; a<n; a++)
     for (int b=a+1; b<n; b++)
      for (int c=b+1; c<n; c++)
       for (int d=c+1; d<n; d++)
        for (int e=d+1; e<n; e++)
         fn(card_bit(v[a])|card_bit(v[b])|card_bit(v[c])|card_bit(v[d])|card_bit(v[e]));
}

// Cartesian product for straight enumeration
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
    std::unordered_set<CardSet>& seen, std::vector<HandInfo>& results)
{
    for (int si = 0; si < 8; si++) {
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
    std::unordered_set<CardSet>& seen, std::vector<HandInfo>& results)
{
    for (int s = 0; s < NUM_SUITS; s++) {
        CardSet sc = hand & suit_mask(s);
        if (popcount64(sc) < 5) continue;
        enum_5subsets(sc, [&](CardSet combo) {
            if (seen.count(combo)) return;
            HandInfo h = detect_hand(combo, 5);
            if (!h.is_play()) return;
            if (h.type != HAND_FLUSH) return; // straight_flush handled by find_straights
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
        // quad is all 4 cards of this rank
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
    std::unordered_set<CardSet>& seen, std::vector<HandInfo>& results)
{
    for (int s = 0; s < NUM_SUITS; s++) {
        CardSet sc = hand & suit_mask(s);
        if (popcount64(sc) < 5) continue;
        // Find straights within this suit
        for (int si = 0; si < 8; si++) {
            bool has_all = true;
            CardSet rc[5];
            for (int j = 0; j < 5; j++) {
                rc[j] = sc & rank_mask(STRAIGHT_SEQS[si][j]);
                if (rc[j] == 0) { has_all = false; break; }
            }
            if (!has_all) continue;
            // Each rank in this suit has exactly 1 card, so only 1 combo
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

std::vector<HandInfo> get_all_hands(CardSet hand) {
    std::vector<HandInfo> results;
    std::unordered_set<CardSet> seen;

    // Singles
    for_each_card(hand, [&](int c) {
        CardSet cs = card_bit(c);
        results.push_back({HAND_SINGLE, cs, c, 1});
        seen.insert(cs);
    });

    // Pairs and Triples
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

    // 5-card hands
    find_straights(hand, nullptr, seen, results);
    find_flushes(hand, nullptr, seen, results);
    find_full_houses(hand, nullptr, seen, results);
    find_four_with_ones(hand, nullptr, seen, results);
    // straight_flushes already found by find_straights (detect_hand returns SF)

    std::sort(results.begin(), results.end(), hand_sort_cmp);
    return results;
}

std::vector<HandInfo> get_beating_hands(CardSet hand, const HandInfo& last) {
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
        if (lp <= 0) find_straights(hand, &last, seen, results);
        if (lp <= 1) find_flushes(hand, &last, seen, results);
        if (lp <= 2) find_full_houses(hand, &last, seen, results);
        if (lp <= 3) find_four_with_ones(hand, &last, seen, results);
        if (lp <= 4) find_straight_flushes(hand, &last, seen, results);
    }

    std::sort(results.begin(), results.end(), hand_sort_cmp);
    return results;
}

// ======================== Team Assignment ========================

void assign_teams(const CardSet hands[NUM_PLAYERS],
                  Team out[NUM_PLAYERS], bool& is_solo)
{
    int s3_owner = -1, sA_owner = -1;
    int spade3 = make_card(3, 12); // spade_3
    int spadeA = make_card(3, 10); // spade_A
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

    // Check if any team's members are all finished
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
        HandInfo decl; decl.type = HAND_DECLARE; decl.size = 0;
        HandInfo pass; // default HAND_PASS
        return {decl, pass};
    }
    CardSet my = hands[current_player];
    if (my == 0) return {};

    if (last_play.is_pass()) {
        // Free play
        auto all = get_all_hands(my);
        if (is_first_turn) {
            int diamond4 = make_card(0, 0); // diamond_4
            std::vector<HandInfo> filtered;
            for (auto& h : all) {
                if (h.cards & card_bit(diamond4))
                    filtered.push_back(h);
            }
            return filtered;
        }
        return all;
    } else {
        auto beating = get_beating_hands(my, last_play);
        // If only 1 card left and can beat, must play (no pass)
        if (popcount64(my) == 1 && !beating.empty())
            return beating;
        // Add pass option
        HandInfo pass;
        beating.push_back(pass);
        return beating;
    }
}

GameState GameState::apply_move(const HandInfo& move) const {
    if (is_declaration_phase) {
        // Declaration phase
        int p = declaration_turn;
        if (move.is_declare()) {
            GameState ns;
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
        // Pass
        std::set<int> finished(rankings.begin(), rankings.end());
        int active_count = num_players - (int)rankings.size();
        int new_pass_count = pass_count + 1;
        bool last_still_active = (last_play_player >= 0 && !finished.count(last_play_player));
        int passes_needed = last_still_active ? active_count - 1 : active_count;

        GameState ns;
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

    // Play cards
    CardSet played = move.cards;
    GameState ns;
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

    // Check terminal → fill remaining players into rankings
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
    // Find next active player
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

    // OPPONENT team
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

// ======================== Engine ========================

Engine::Engine() : rng_(42) {
    std::memset(played_cards_, 0, sizeof(played_cards_));
    std::memset(seat_agents_, 0, sizeof(seat_agents_));
}

void Engine::seed(unsigned int s) { rng_.seed(s); }

int Engine::reset() {
    // Clear tracking
    action_history_.clear();
    std::memset(played_cards_, 0, sizeof(played_cards_));
    for (int i = 0; i < NUM_PLAYERS; i++) step_rewards_[i].clear();
    key_to_hand_.clear();

    // Assign rule agent seats
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

    // Shuffle and deal
    std::vector<int> deck(NUM_CARDS);
    std::iota(deck.begin(), deck.end(), 0);
    std::shuffle(deck.begin(), deck.end(), rng_);

    for (int i = 0; i < NUM_PLAYERS; i++) state_.hands[i] = 0;
    for (int i = 0; i < NUM_CARDS; i++)
        state_.hands[i / 13] |= card_bit(deck[i]);

    // Find diamond_4 holder
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
    state_.is_declaration_phase = true;
    state_.declaration_turn = 0;
    state_.declaration_passes = 0;
    state_.is_declared = false; state_.declarant = -1;
    state_.compute_observed_teams();

    prev_state_ = state_;

    // Declaration phase starts from seat 0
    return state_.declaration_turn;
}

int Engine::reset_with_hands(
    const std::vector<std::vector<std::string>>& hands_ids,
    int start_player)
{
    action_history_.clear();
    std::memset(played_cards_, 0, sizeof(played_cards_));
    for (int i = 0; i < NUM_PLAYERS; i++) step_rewards_[i].clear();
    key_to_hand_.clear();

    // No rule agents for parity testing
    for (int i = 0; i < NUM_PLAYERS; i++) seat_agents_[i] = SEAT_RL;

    // Build hands from card id strings
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
    state_.is_declaration_phase = true;
    state_.declaration_turn = 0;
    state_.declaration_passes = 0;
    state_.is_declared = false; state_.declarant = -1;
    state_.compute_observed_teams();

    prev_state_ = state_;
    return state_.declaration_turn; // always 0
}

HandInfo Engine::resolve_action(const std::string& key) const {
    if (key == "pass") return {}; // HAND_PASS
    if (key == "declare") { HandInfo h; h.type = HAND_DECLARE; return h; }
    auto it = key_to_hand_.find(key);
    if (it != key_to_hand_.end()) return it->second;
    // Parse key: "suit_rank|suit_rank|..."
    CardSet cs = 0;
    std::istringstream iss(key);
    std::string token;
    while (std::getline(iss, token, '|')) {
        int c = string_to_card(token);
        if (c >= 0) cs |= card_bit(c);
    }
    int sz = popcount64(cs);
    if (sz == 0) return {};
    return detect_hand(cs, sz);
}

int Engine::step(const std::string& action_key) {
    prev_state_ = state_;
    int player_id = get_player_id();

    HandInfo action = resolve_action(action_key);

    // Track history and played cards (only in play phase)
    if (!state_.is_declaration_phase) {
        action_history_.push_back({player_id, action.is_play() ? action.cards : 0});
        if (action.is_play()) {
            played_cards_[player_id] |= action.cards;
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

std::array<std::array<int64_t,3>,NUM_PLAYERS> Engine::get_aux_targets() const {
    std::array<std::array<int64_t,3>,NUM_PLAYERS> targets;
    const Team* actual = state_.actual_teams;
    const Team* observed = state_.observed_teams;
    const int team_cls[] = {0, 1, 2, 1}; // SPADE_A3=0, OPPONENT=1, SOLO=2, UNKNOWN→1

    for (int p = 0; p < NUM_PLAYERS; p++) {
        for (int j = 0; j < 3; j++) {
            int other = (p + j + 1) % NUM_PLAYERS;
            if (observed[other] == TEAM_UNKNOWN)
                targets[p][j] = -1;
            else
                targets[p][j] = team_cls[(int)actual[other]];
        }
    }
    return targets;
}

// ======================== State Encoding (850-dim) ========================

void Engine::encode_obs(int player_id, int8_t* out) const {
    std::memset(out, 0, STATE_DIM);
    int8_t* ptr = out;
    int n = NUM_PLAYERS;

    // Relative seat order: [me, next, across, previous]
    int rel_order[NUM_PLAYERS];
    for (int i = 0; i < n; i++) rel_order[i] = (player_id + i) % n;

    // 1. My hand (52d)
    for_each_card(state_.hands[player_id], [&](int c){ ptr[c] = 1; });
    ptr += NUM_CARDS;

    // 2. Last play cards (52d)
    if (state_.last_play.is_play())
        for_each_card(state_.last_play.cards, [&](int c){ ptr[c] = 1; });
    ptr += NUM_CARDS;

    // 3. Last play player (4d relative one-hot)
    if (state_.last_play_player >= 0) {
        for (int i = 0; i < n; i++)
            if (rel_order[i] == state_.last_play_player) { ptr[i] = 1; break; }
    }
    ptr += n;

    // 4. History (HISTORY_LEN × 57d)
    int hist_start = std::max(0, (int)action_history_.size() - HISTORY_LEN);
    int hist_count = (int)action_history_.size() - hist_start;
    for (int h = 0; h < hist_count; h++) {
        auto& entry = action_history_[hist_start + h];
        // Player (4d relative one-hot)
        for (int i = 0; i < n; i++)
            if (rel_order[i] == entry.player_id) { ptr[i] = 1; break; }
        ptr += n;
        // Valid flag (1d)
        *ptr++ = 1;
        // Cards (52d)
        for_each_card(entry.cards, [&](int c){ ptr[c] = 1; });
        ptr += NUM_CARDS;
    }
    // Pad empty history slots (valid=0, rest=0)
    ptr += (HISTORY_LEN - hist_count) * HISTORY_STEP_DIM;

    // 5. Played cards per player (4×52d, relative order)
    for (int i = 0; i < n; i++) {
        int abs_i = rel_order[i];
        for_each_card(played_cards_[abs_i], [&](int c){ ptr[c] = 1; });
        ptr += NUM_CARDS;
    }

    // 6. Remaining hand count one-hot (4×14d, relative order)
    for (int i = 0; i < n; i++) {
        int abs_i = rel_order[i];
        int cnt = std::min(popcount64(state_.hands[abs_i]), 13);
        ptr[cnt] = 1;
        ptr += 14;
    }

    // 7. Team one-hot (4×4d, relative order)
    // Teams order: SPADE_A3=0, OPPONENT=1, SOLO=2, UNKNOWN=3
    for (int i = 0; i < n; i++) {
        int abs_i = rel_order[i];
        Team team;
        if (abs_i == player_id)
            team = state_.actual_teams[abs_i]; // I know my own team
        else
            team = state_.observed_teams[abs_i]; // Others: observed only
        ptr[(int)team] = 1;
        ptr += 4;
    }

    // 8. Misc (6d)
    ptr[0] = state_.last_play.is_pass() ? 1 : 0;  // free play
    ptr[1] = state_.is_first_turn ? 1 : 0;
    ptr[2] = (int8_t)std::min(state_.pass_count, 3);
    ptr[3] = (int8_t)state_.rankings.size();
    ptr[4] = state_.is_solo ? 1 : 0;
    ptr[5] = state_.is_declaration_phase ? 1 : 0;
}

// ======================== Legal Actions ========================

std::vector<Engine::ActionEntry> Engine::get_legal_actions() const {
    std::vector<ActionEntry> result;

    if (state_.is_declaration_phase) {
        ActionEntry decl;
        decl.key = "declare";
        std::memset(decl.feature, 1, ACTION_DIM); // all-ones
        result.push_back(decl);

        ActionEntry pass;
        pass.key = "pass";
        std::memset(pass.feature, 0, ACTION_DIM);
        result.push_back(pass);
        return result;
    }

    auto moves = state_.get_legal_moves();
    for (auto& h : moves) {
        ActionEntry ae;
        ae.key = h.to_key();
        h.to_feature(ae.feature);
        key_to_hand_[ae.key] = h;
        result.push_back(ae);
    }
    return result;
}

void Engine::get_action_feature(const std::string& key, int8_t* out) const {
    std::memset(out, 0, ACTION_DIM);
    if (key == "declare") {
        std::memset(out, 1, ACTION_DIM);
        return;
    }
    if (key == "pass" || key.empty()) return;
    auto it = key_to_hand_.find(key);
    if (it != key_to_hand_.end()) {
        it->second.to_feature(out);
        return;
    }
    // Parse key
    HandInfo h = resolve_action(key);
    if (h.is_play()) h.to_feature(out);
}

// ======================== Rule Agent ========================

std::string Engine::greedy_action() const {
    if (state_.is_declaration_phase) return "pass";

    auto moves = state_.get_legal_moves();
    // Separate plays and pass
    std::vector<HandInfo> plays;
    for (auto& h : moves)
        if (h.is_play()) plays.push_back(h);
    if (plays.empty()) return "pass";

    int pid = state_.current_player;
    CardSet my = state_.hands[pid];
    int my_count = popcount64(my);

    if (state_.last_play.is_pass()) {
        // Free play
        if (my_count == 1) return plays[0].to_key();
        if (my_count == 2) {
            for (auto& h : plays)
                if (h.size == 2) return h.to_key();
        }
        // Smallest single
        std::sort(plays.begin(), plays.end(), [](const HandInfo& a, const HandInfo& b){
            return card_score(a.primary_card) < card_score(b.primary_card);
        });
        for (auto& h : plays)
            if (h.size == 1) return h.to_key();
        return plays[0].to_key();
    }

    // Follow play - check if teammate
    int lpp = state_.last_play_player;
    Team my_team = state_.observed_teams[pid];
    Team opp_team = (lpp >= 0) ? state_.observed_teams[lpp] : TEAM_UNKNOWN;
    bool is_teammate = (my_team != TEAM_UNKNOWN && opp_team != TEAM_UNKNOWN
                        && my_team == opp_team);
    if (is_teammate && std::any_of(moves.begin(), moves.end(),
                                  [](const HandInfo& hand) { return hand.is_pass(); }))
        return "pass";

    // Play smallest beater
    std::sort(plays.begin(), plays.end(), [](const HandInfo& a, const HandInfo& b){
        return card_score(a.primary_card) < card_score(b.primary_card);
    });
    return plays[0].to_key();
}

std::string Engine::random_action() const {
    if (state_.is_declaration_phase) {
        std::uniform_int_distribution<int> dist(0, 1);
        return dist(rng_) == 0 ? "declare" : "pass";
    }
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

std::array<float, NUM_PLAYERS> VectorizedEngine::get_training_payoffs(int i) const {
    return engines_[i].get_training_payoffs();
}
const std::vector<float>& VectorizedEngine::get_step_rewards(int i, int p) const {
    return engines_[i].get_step_rewards(p);
}
std::array<std::array<int64_t,3>,NUM_PLAYERS> VectorizedEngine::get_aux_targets(int i) const {
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
    bd.obs_expanded.reserve(est * STATE_DIM);
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
            bd.obs_expanded.insert(bd.obs_expanded.end(),
                                   obs_ptr, obs_ptr + STATE_DIM);
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

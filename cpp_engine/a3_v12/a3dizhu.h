#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <string>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <cassert>
#include <functional>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace a3dizhu {

// ======================== Platform Compat ========================
#ifdef _MSC_VER
inline int popcount64(uint64_t x) { return (int)__popcnt64(x); }
inline int ctz64(uint64_t x) {
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return (int)idx;
}
#else
inline int popcount64(uint64_t x) { return __builtin_popcountll(x); }
inline int ctz64(uint64_t x) { return __builtin_ctzll(x); }
#endif

// ======================== Card Constants ========================
constexpr int NUM_PLAYERS   = 4;
constexpr int NUM_OPPONENTS = 3;
constexpr int NUM_SUITS     = 4;
constexpr int NUM_RANKS     = 13;
constexpr int NUM_CARDS     = 52;

// ======================== V11 Encoding Dimensions ========================
//
// obs_static (556D) + hist_tokens (24 × 88D = 2112D) = 2668D
// action_feat = 111D
// total flat input = 2668 + 111 = 2779D
//
constexpr int HISTORY_LEN      = 24;
constexpr int HIST_TOKEN_DIM   = 88;    // 4+1+1+9+13+4+4+52
constexpr int OBS_STATIC_DIM   = 556;
constexpr int ACTION_FEAT_DIM  = 111;

constexpr int STATE_DIM   = OBS_STATIC_DIM + HISTORY_LEN * HIST_TOKEN_DIM;  // 2668
constexpr int ACTION_DIM  = ACTION_FEAT_DIM;                                // 111

// Aux: relation(9) + s3_owner(4) + sa_owner(4) = 17
constexpr int AUX_DIM = NUM_OPPONENTS * 3 + 4 + 4;  // 17

// Action type one-hot (9 categories)
constexpr int NUM_ACTION_TYPES = 9;
// [single=0, pair=1, triple=2, straight=3, flush=4,
//  full_house=5, four_with_one=6, straight_flush=7, pass=8]

// Suit: 0=diamond, 1=club, 2=heart, 3=spade
// Rank: 0='4', 1='5', ..., 9='K', 10='A', 11='2', 12='3'
using CardSet = uint64_t;

inline int card_suit(int c)  { return c / NUM_RANKS; }
inline int card_rank(int c)  { return c % NUM_RANKS; }
inline int make_card(int s, int r) { return s * NUM_RANKS + r; }
inline CardSet card_bit(int c) { return 1ULL << c; }

// rank * 10 + suit  (matches Python Card.score())
inline int card_score(int c) { return card_rank(c) * 10 + card_suit(c); }

// Straight rank values: '3'=1, '4'=2, '5'=3, ..., 'A'=12, '2'=0(invalid)
extern const int STRAIGHT_RANK_VAL[NUM_RANKS];

inline CardSet rank_mask(int rank) {
    return card_bit(rank) | card_bit(NUM_RANKS + rank)
         | card_bit(2 * NUM_RANKS + rank) | card_bit(3 * NUM_RANKS + rank);
}

inline CardSet suit_mask(int suit) {
    return ((1ULL << NUM_RANKS) - 1) << (suit * NUM_RANKS);
}

extern const char* SUIT_NAMES[NUM_SUITS];
extern const char* RANK_NAMES[NUM_RANKS];

std::string card_to_string(int card);
int         string_to_card(const std::string& s);

template<typename Fn>
inline void for_each_card(CardSet cs, Fn fn) {
    while (cs) {
        int c = ctz64(cs);
        fn(c);
        cs &= cs - 1;
    }
}

inline std::vector<int> cards_vec(CardSet cs) {
    std::vector<int> v;
    for_each_card(cs, [&](int c){ v.push_back(c); });
    return v;
}

// ======================== Action Type Helpers ========================

inline int hand_type_to_action_idx(int type) {
    // HAND_SINGLE=0..HAND_STRAIGHT_FLUSH=7 map directly
    // HAND_PASS=-2 maps to 8
    if (type >= 0 && type <= 7) return type;
    if (type == -2) return 8; // pass
    return -1;
}

inline int hand_size_to_len_idx(int size) {
    switch (size) {
        case 1: return 0;
        case 2: return 1;
        case 3: return 2;
        case 5: return 3;
        default: return -1;
    }
}

// ======================== Hand Types ========================
enum HandType : int8_t {
    HAND_PASS           = -2,
    HAND_DECLARE        = -1,
    HAND_SINGLE         = 0,
    HAND_PAIR           = 1,
    HAND_TRIPLE         = 2,
    HAND_STRAIGHT       = 3,
    HAND_FLUSH          = 4,
    HAND_FULL_HOUSE     = 5,
    HAND_FOUR_WITH_ONE  = 6,
    HAND_STRAIGHT_FLUSH = 7,
};

inline int five_card_priority(HandType t) {
    switch (t) {
        case HAND_STRAIGHT:       return 0;
        case HAND_FLUSH:          return 1;
        case HAND_FULL_HOUSE:     return 2;
        case HAND_FOUR_WITH_ONE:  return 3;
        case HAND_STRAIGHT_FLUSH: return 4;
        default: return -1;
    }
}

struct HandInfo {
    HandType type = HAND_PASS;
    CardSet  cards = 0;
    int      primary_card = -1;
    int      size = 0;

    bool is_pass()    const { return type == HAND_PASS; }
    bool is_declare() const { return type == HAND_DECLARE; }
    bool is_play()    const { return type >= HAND_SINGLE; }

    std::string to_key() const;
    void to_card_bits(int8_t* out52) const;
};

// ======================== Afterstate Summary ========================

struct AfterstateInfo {
    int remaining_count;
    int singles_count;
    int pairs_count;
    int triples_count;
    int fivecard_potential;
    int min_steps;
    bool has_s3;
    bool has_sa;
    bool has_rank3_single;
    bool has_rank2_single;
    bool has_straight_potential;
    bool has_flush_potential;
    bool has_sf_potential;
    bool has_threepair_or_fourone_potential;
};

AfterstateInfo compute_afterstate(CardSet hand_after, int start_val, int end_val);

// ======================== Hand Detection & Comparison ========================
HandInfo detect_hand(CardSet cards, int size);
int  compare_hands(const HandInfo& a, const HandInfo& b);
bool can_beat(const HandInfo& a, const HandInfo& b);

// ======================== Legal Move Enumeration ========================
std::vector<HandInfo> get_all_hands(CardSet my_cards, int start_val = 1, int end_val = 12);
std::vector<HandInfo> get_beating_hands(CardSet my_cards, const HandInfo& last, int start_val = 1, int end_val = 12);

// ======================== Team ========================
enum Team : int8_t {
    TEAM_SPADE_A3 = 0,
    TEAM_OPPONENT = 1,
    TEAM_SOLO     = 2,
    TEAM_UNKNOWN  = 3,
};

// ======================== Game State ========================
struct GameState {
    CardSet  hands[NUM_PLAYERS];
    int      current_player;
    HandInfo last_play;
    int      last_play_player;
    int      pass_count;
    bool     is_first_turn;
    std::vector<int> rankings;
    Team     actual_teams[NUM_PLAYERS];
    bool     is_solo;
    int      num_players = NUM_PLAYERS;
    int      spade3_player = -1;
    int      spadeA_player = -1;

    // Declaration phase
    bool is_declaration_phase = false;
    int  declaration_turn     = 0;
    int  declaration_passes   = 0;
    bool is_declared          = false;
    int  declarant            = -1;

    // Rule variants (randomized each game)
    bool declare_require_both_spades = false;
    int  straight_start_val = 1;   // STRAIGHT_RANK_VAL range: 1=from 3, 2=from 4
    int  straight_end_val   = 12;  // 11=up to K, 12=up to A

    // Observed teams (computed from spade tracking)
    Team observed_teams[NUM_PLAYERS];

    void compute_observed_teams();
    bool is_terminal() const;
    std::vector<HandInfo> get_legal_moves() const;
    GameState apply_move(const HandInfo& move) const;
    int  next_active(int from_player) const;
};

void assign_teams(const CardSet hands[NUM_PLAYERS],
                  Team out_teams[NUM_PLAYERS], bool& is_solo);

// ======================== Payoff / Reward ========================
std::array<float,NUM_PLAYERS> compute_payoffs(const GameState& st);
std::array<float,NUM_PLAYERS> compute_training_payoffs(const GameState& st);
std::array<float,NUM_PLAYERS> compute_declared_payoffs(
    const std::vector<int>& rankings, int declarant);
float compute_step_reward(
    const GameState& prev, const HandInfo& action,
    const GameState& next, int player_id);

// ======================== Engine (full env) ========================

struct RichHistoryEntry {
    int      player_id;
    HandInfo hand;
};

class Engine {
public:
    Engine();

    void set_greedy_ratio(double r) { greedy_ratio_ = r; }
    void set_random_ratio(double r) { random_ratio_ = r; }
    void seed(unsigned int s);

    int  reset();
    int  reset_with_hands(
        const std::vector<std::vector<std::string>>& hands,
        int start_player);
    void set_rules(bool declare_require_both_spades, int straight_start_val, int straight_end_val);
    int  step(const std::string& action_key);

    int  get_player_id()        const;
    bool is_over()              const;
    bool is_declaration_phase()  const;
    // Evaluation metadata only; not part of a player's observation.
    std::string get_mode() const {
        return state_.is_declared ? "declared" : (state_.is_solo ? "solo" : "normal");
    }
    bool is_rule_agent_seat(int pid) const;

    // V11: encode obs_static(556D) + hist_tokens(24×88D) = 2668D flat
    void encode_obs(int player_id, int8_t* out) const;

    // V11: ActionEntry now has 111D feature (card_bits + type + rank + suit + len + flags + afterstate + rule_straight)
    struct ActionEntry {
        std::string key;
        int8_t feature[ACTION_FEAT_DIM];
    };
    std::vector<ActionEntry> get_legal_actions() const;

    void get_action_feature(const std::string& key, int8_t* out) const;

    std::string get_rule_agent_action() const;
    std::string get_greedy_action() const { return greedy_action(); }

    std::array<float,NUM_PLAYERS> get_payoffs()          const;
    std::array<float,NUM_PLAYERS> get_training_payoffs()  const;
    const std::vector<float>&     get_step_rewards(int p) const;

    // V11 aux targets: relation(3×int64) + s3_owner(int64) + sa_owner(int64)
    struct AuxTargets {
        std::array<int64_t, 3> relation;  // 1=same_side, 2=opposite, -1=unknown
        int64_t s3_owner;  // relative slot of ♠3 holder (-1 if unknown)
        int64_t sa_owner;  // relative slot of ♠A holder (-1 if unknown)
    };
    std::array<AuxTargets, NUM_PLAYERS> get_aux_targets() const;

    const GameState& game_state() const { return state_; }

private:
    GameState state_;
    GameState prev_state_;

    std::vector<RichHistoryEntry> action_history_;
    CardSet played_cards_[NUM_PLAYERS];
    std::vector<float> step_rewards_[NUM_PLAYERS];

    // V11: last non-pass action tracking (persists across round resets)
    HandInfo last_nonpass_action_;
    int      last_nonpass_player_ = -1;

    double greedy_ratio_ = 0.0;
    double random_ratio_ = 0.0;

    enum SeatAgent : int8_t { SEAT_RL=0, SEAT_GREEDY=1, SEAT_RANDOM=2 };
    SeatAgent seat_agents_[NUM_PLAYERS];

    mutable std::unordered_map<std::string, HandInfo> key_to_hand_;

    mutable std::mt19937 rng_;

    HandInfo resolve_action(const std::string& key) const;
    std::string greedy_action() const;
    std::string random_action() const;

    // V11 encoding helpers
    void encode_obs_static(int player_id, int8_t* out) const;
    void encode_hist_tokens(int player_id, int8_t* out) const;
    void encode_hist_token(const RichHistoryEntry& entry, int player_id,
                           const int* rel_order, int8_t* out) const;
    void encode_action_feature(const HandInfo& hand, int player_id,
                               int8_t* out) const;

    static void cardset_to_feature(CardSet cs, int8_t* out);
};

// ======================== Vectorized Engine ========================

class VectorizedEngine {
public:
    struct RuleStepData {
        int player_id;
        int8_t obs[STATE_DIM];
        int8_t action[ACTION_DIM];
        Engine::AuxTargets auxiliary;
    };

    struct BatchData {
        std::vector<int8_t> action_flat;    // [total_actions * ACTION_DIM]
        std::vector<int8_t> obs_raw;        // [K * STATE_DIM]
        std::vector<int>    offsets;         // [K + 1]
        std::vector<std::vector<std::string>> action_keys;
        int total_actions = 0;
    };

    explicit VectorizedEngine(int num_envs);

    int  num_envs() const { return n_; }

    void set_greedy_ratio(double r);
    void set_random_ratio(double r);
    void seed(unsigned int base_seed);

    int  reset(int idx);
    void set_rules(int idx, bool require_both, int start_val, int end_val);
    int  step(int idx, const std::string& action_key);
    int  get_player_id(int idx) const;
    bool is_over(int idx) const;
    bool is_rule_agent_seat(int idx, int pid) const;
    bool is_declaration_phase(int idx) const;
    std::string get_rule_agent_action(int idx) const;

    void encode_obs(int idx, int player_id, int8_t* out) const;
    void get_action_feature(int idx, const std::string& key, int8_t* out) const;

    std::array<float, NUM_PLAYERS> get_payoffs(int idx) const;
    std::array<float, NUM_PLAYERS> get_training_payoffs(int idx) const;
    const std::vector<float>& get_step_rewards(int idx, int p) const;
    Engine::AuxTargets get_aux_targets_for_player(int idx, int pid) const;
    std::array<Engine::AuxTargets, NUM_PLAYERS> get_aux_targets(int idx) const;

    std::pair<bool, std::vector<RuleStepData>> advance_to_decision(int idx);
    int step_random(int idx);
    BatchData prepare_batch(const std::vector<int>& pending) const;

private:
    std::vector<Engine> engines_;
    int n_;
};

} // namespace a3dizhu

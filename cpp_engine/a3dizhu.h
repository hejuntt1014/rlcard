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

// ======================== Constants ========================
constexpr int NUM_PLAYERS   = 4;
constexpr int NUM_SUITS     = 4;
constexpr int NUM_RANKS     = 13;
constexpr int NUM_CARDS     = 52;
constexpr int STATE_DIM     = 850;
constexpr int ACTION_DIM    = 52;
constexpr int HISTORY_LEN   = 8;
constexpr int HISTORY_STEP_DIM = 4 + 1 + NUM_CARDS; // 57

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

// All 4 cards of a given rank
inline CardSet rank_mask(int rank) {
    return card_bit(rank) | card_bit(NUM_RANKS + rank)
         | card_bit(2 * NUM_RANKS + rank) | card_bit(3 * NUM_RANKS + rank);
}

// All 13 cards of a given suit
inline CardSet suit_mask(int suit) {
    return ((1ULL << NUM_RANKS) - 1) << (suit * NUM_RANKS);
}

extern const char* SUIT_NAMES[NUM_SUITS];
extern const char* RANK_NAMES[NUM_RANKS];

std::string card_to_string(int card);
int         string_to_card(const std::string& s);

// Iterate set bits
template<typename Fn>
inline void for_each_card(CardSet cs, Fn fn) {
    while (cs) {
        int c = ctz64(cs);
        fn(c);
        cs &= cs - 1;
    }
}

// Collect set bits into vector
inline std::vector<int> cards_vec(CardSet cs) {
    std::vector<int> v;
    for_each_card(cs, [&](int c){ v.push_back(c); });
    return v;
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
    void to_feature(int8_t* out) const; // 52-dim
};

// ======================== Hand Detection & Comparison ========================
HandInfo detect_hand(CardSet cards, int size);
int  compare_hands(const HandInfo& a, const HandInfo& b);
bool can_beat(const HandInfo& a, const HandInfo& b);

// ======================== Legal Move Enumeration ========================
std::vector<HandInfo> get_all_hands(CardSet my_cards);
std::vector<HandInfo> get_beating_hands(CardSet my_cards, const HandInfo& last);

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
class Engine {
public:
    Engine();

    void set_greedy_ratio(double r) { greedy_ratio_ = r; }
    void set_random_ratio(double r) { random_ratio_ = r; }
    void seed(unsigned int s);

    int  reset();                                // returns first player_id
    // Reset with specific hands (for parity testing against Python engine)
    // hands: 4 × 13 card id strings, e.g. "spade_A"
    // start_player: the player holding diamond_4
    int  reset_with_hands(
        const std::vector<std::vector<std::string>>& hands,
        int start_player);
    int  step(const std::string& action_key);    // returns next player_id

    int  get_player_id()        const;
    bool is_over()              const;
    bool is_declaration_phase()  const;
    // Evaluation metadata only; not part of a player's observation.
    std::string get_mode() const {
        return state_.is_declared ? "declared" : (state_.is_solo ? "solo" : "normal");
    }
    bool is_rule_agent_seat(int pid) const;

    void encode_obs(int player_id, int8_t* out) const;

    struct ActionEntry {
        std::string key;
        int8_t feature[ACTION_DIM];
    };
    std::vector<ActionEntry> get_legal_actions() const;

    void get_action_feature(const std::string& key, int8_t* out) const;

    std::string get_rule_agent_action() const;
    std::string get_greedy_action() const { return greedy_action(); }

    std::array<float,NUM_PLAYERS> get_payoffs()          const;
    std::array<float,NUM_PLAYERS> get_training_payoffs()  const;
    const std::vector<float>&     get_step_rewards(int p) const;
    std::array<std::array<int64_t,3>,NUM_PLAYERS> get_aux_targets() const;

    const GameState& game_state() const { return state_; }

private:
    GameState state_;
    GameState prev_state_;

    struct HistoryEntry { int player_id; CardSet cards; };
    std::vector<HistoryEntry> action_history_;
    CardSet played_cards_[NUM_PLAYERS];
    std::vector<float> step_rewards_[NUM_PLAYERS];

    double greedy_ratio_ = 0.0;
    double random_ratio_ = 0.0;

    enum SeatAgent : int8_t { SEAT_RL=0, SEAT_GREEDY=1, SEAT_RANDOM=2 };
    SeatAgent seat_agents_[NUM_PLAYERS];

    mutable std::unordered_map<std::string, HandInfo> key_to_hand_;

    mutable std::mt19937 rng_;

    HandInfo resolve_action(const std::string& key) const;
    std::string greedy_action() const;
    std::string random_action() const;

    static void cardset_to_feature(CardSet cs, int8_t* out);
};

// ======================== Vectorized Engine (batch RL training) ========================

class VectorizedEngine {
public:
    struct RuleStepData {
        int player_id;
        int8_t obs[STATE_DIM];
        int8_t action[ACTION_DIM];
    };

    struct BatchData {
        std::vector<int8_t> obs_expanded;   // [total_actions * STATE_DIM]
        std::vector<int8_t> action_flat;    // [total_actions * ACTION_DIM]
        std::vector<int8_t> obs_raw;        // [K * STATE_DIM]
        std::vector<int>    offsets;        // [K + 1]
        std::vector<std::vector<std::string>> action_keys;
        int total_actions = 0;
    };

    explicit VectorizedEngine(int num_envs);

    int  num_envs() const { return n_; }

    void set_greedy_ratio(double r);
    void set_random_ratio(double r);
    void seed(unsigned int base_seed);

    int  reset(int idx);
    int  step(int idx, const std::string& action_key);
    int  get_player_id(int idx) const;
    bool is_over(int idx) const;
    bool is_rule_agent_seat(int idx, int pid) const;
    bool is_declaration_phase(int idx) const;
    std::string get_rule_agent_action(int idx) const;

    void encode_obs(int idx, int player_id, int8_t* out) const;
    void get_action_feature(int idx, const std::string& key, int8_t* out) const;

    std::array<float, NUM_PLAYERS> get_training_payoffs(int idx) const;
    const std::vector<float>& get_step_rewards(int idx, int p) const;
    std::array<std::array<int64_t, 3>, NUM_PLAYERS> get_aux_targets(int idx) const;

    // Advance one env through rule-agent turns until it needs an RL
    // decision or the game ends.  Returns (is_game_over, rule_steps).
    std::pair<bool, std::vector<RuleStepData>> advance_to_decision(int idx);

    // Pick a random legal action and step (for staggering init only).
    int step_random(int idx);

    // Prepare a single contiguous batch for GPU inference over several envs.
    BatchData prepare_batch(const std::vector<int>& pending) const;

private:
    std::vector<Engine> engines_;
    int n_;
};

} // namespace a3dizhu

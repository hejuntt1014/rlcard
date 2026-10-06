#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "a3dizhu.h"
#include "rollout_buffer.h"

namespace py = pybind11;
using namespace a3dizhu_v12;

// Arrays own a shared BatchData capsule. No copy into a second NumPy allocation,
// and outstanding trajectories remain valid after later engine calls.
static py::tuple prepare_numpy_batch(const VectorizedEngine& v,
                                    const std::vector<int>& pending, bool metadata,
                                    bool indexed = false) {
    for (const int idx : pending)
        if (idx < 0 || idx >= v.num_envs())
            throw py::index_error("Environment index out of range");
    VectorizedEngine::BatchData* batch;
    {
        py::gil_scoped_release release;
        batch = new VectorizedEngine::BatchData(v.prepare_batch(pending, indexed));
    }
    py::capsule owner(batch, [](void* ptr) {
        delete static_cast<VectorizedEngine::BatchData*>(ptr);
    });
    const py::ssize_t count = pending.size(), total = batch->total_actions;
    py::array_t<int8_t> observations({count, py::ssize_t(STATE_DIM)},
        {py::ssize_t(STATE_DIM), py::ssize_t(1)}, batch->obs_raw.data(), owner);
    py::array_t<int8_t> actions({total, py::ssize_t(ACTION_DIM)},
        {py::ssize_t(ACTION_DIM), py::ssize_t(1)}, batch->action_flat.data(), owner);
    if (!metadata)
        return py::make_tuple(observations, actions, batch->offsets, batch->action_keys);
    std::vector<int> players;
    players.reserve(count);
    py::array_t<int64_t> auxiliary({count, py::ssize_t(5)});
    for (py::ssize_t row = 0; row < count; ++row) {
        const int pid = v.get_player_id(pending[row]);
        players.push_back(pid);
        const auto labels = v.get_aux_targets_for_player(pending[row], pid);
        auto* out = auxiliary.mutable_data() + row * 5;
        for (int j = 0; j < 3; ++j) out[j] = labels.relation[j];
        out[3] = labels.s3_owner;
        out[4] = labels.sa_owner;
    }
    if (indexed)
        return py::make_tuple(observations, actions, batch->offsets, players, auxiliary);
    return py::make_tuple(observations, actions, batch->offsets, batch->action_keys,
                          players, auxiliary);
}

PYBIND11_MODULE(a3dizhu_v12_cpp, m) {
    m.doc() = "A3 Dizhu engine with compact batches and 24-step history features";
    m.attr("FEATURE_VERSION") = "a3-v12-lite-v1";
    m.attr("STATE_DIM") = STATE_DIM;
    m.attr("ACTION_DIM") = ACTION_DIM;
    m.attr("OBS_STATIC_DIM") = OBS_STATIC_DIM;
    m.attr("HISTORY_LEN") = HISTORY_LEN;
    m.attr("HIST_TOKEN_DIM") = HIST_TOKEN_DIM;
    m.attr("AUX_DIM") = AUX_DIM;

    py::class_<RolloutBuffer>(m, "RolloutBuffer", py::module_local())
        .def(py::init<int, int>(), py::arg("num_envs"), py::arg("max_episode_steps") = 10000)
        .def("size", &RolloutBuffer::size, py::arg("idx"))
        .def("clear", &RolloutBuffer::clear, py::arg("idx"))
        .def("record_choices", &RolloutBuffer::record_choices, py::arg("indices"), py::arg("choices"),
             py::arg("obs"), py::arg("actions"), py::arg("offsets"), py::arg("players"), py::arg("auxiliary"))
        .def("record_block", &RolloutBuffer::record_block, py::arg("idx"), py::arg("obs"),
             py::arg("players"), py::arg("actions"), py::arg("auxiliary"))
        .def("finish", &RolloutBuffer::finish, py::arg("idx"), py::arg("payoffs"), py::arg("rewards") = py::none());

    m.def("compute_afterstate", [](const std::vector<std::string>& cards,
                                  int start_val, int end_val) {
        CardSet hand = 0;
        for (const auto& key : cards) {
            int card = string_to_card(key);
            if (card < 0) throw py::value_error("Invalid card: " + key);
            if (hand & card_bit(card)) throw py::value_error("Duplicate card: " + key);
            hand |= card_bit(card);
        }
        if (start_val < 1 || start_val > 2 || end_val < 11 || end_val > 12)
            throw py::value_error("Straight limits must be 1/2 and 11/12");
        auto info = compute_afterstate(hand, start_val, end_val);
        py::dict result;
        result["remaining_count"] = info.remaining_count;
        result["singles_count"] = info.singles_count;
        result["pairs_count"] = info.pairs_count;
        result["triples_count"] = info.triples_count;
        result["fivecard_potential"] = info.fivecard_potential;
        result["min_steps"] = info.min_steps;
        result["has_s3"] = info.has_s3;
        result["has_sa"] = info.has_sa;
        result["has_rank3_single"] = info.has_rank3_single;
        result["has_rank2_single"] = info.has_rank2_single;
        result["has_straight_potential"] = info.has_straight_potential;
        result["has_flush_potential"] = info.has_flush_potential;
        result["has_sf_potential"] = info.has_sf_potential;
        result["has_threepair_or_fourone_potential"] = info.has_threepair_or_fourone_potential;
        return result;
    }, py::arg("cards"), py::arg("straight_start_val") = 1,
       py::arg("straight_end_val") = 12);

    py::class_<Engine>(m, "CppEngine", py::module_local())
        .def(py::init<>())

        .def("set_greedy_ratio", &Engine::set_greedy_ratio)
        .def("set_random_ratio", &Engine::set_random_ratio)
        .def("set_reward_shaping", &Engine::set_reward_shaping)
        .def("seed", &Engine::seed)

        .def("reset", &Engine::reset,
             "Reset game, returns first player_id")

        .def("reset_with_hands", &Engine::reset_with_hands,
             py::arg("hands"), py::arg("start_player"),
             "Reset with specific hands for parity testing")

        .def("set_rules", &Engine::set_rules,
             py::arg("declare_require_both_spades"),
             py::arg("straight_start_val"),
             py::arg("straight_end_val"))

        .def("step", &Engine::step, py::arg("action_key"),
             "Step game with action key, returns next player_id")

        .def("get_player_id", &Engine::get_player_id)
        .def("is_over", &Engine::is_over)
        .def("is_declaration_phase", &Engine::is_declaration_phase)
        .def("get_mode", &Engine::get_mode)
        .def("get_greedy_action", &Engine::get_greedy_action)
        .def("is_rule_agent_seat", &Engine::is_rule_agent_seat, py::arg("pid"))

        .def("encode_obs", [](const Engine& e, int player_id) {
            py::array_t<int8_t> result(STATE_DIM);
            auto buf = result.mutable_data();
            e.encode_obs(player_id, buf);
            return result;
        }, py::arg("player_id"),
           "Encode V11 observation (obs_static 556D + hist_tokens 24x88D = 2668D)")

        .def("get_legal_actions", [](const Engine& e) {
            auto actions = e.get_legal_actions();
            py::dict result;
            for (auto& a : actions) {
                py::array_t<int8_t> feat(ACTION_DIM);
                std::memcpy(feat.mutable_data(), a.feature, ACTION_DIM);
                result[py::str(a.key)] = feat;
            }
            return result;
        }, "Get legal actions as {key: 111d numpy array}")

        .def("get_action_feature", [](const Engine& e, const std::string& key) {
            py::array_t<int8_t> result(ACTION_DIM);
            e.get_action_feature(key, result.mutable_data());
            return result;
        }, py::arg("key"),
           "Get 111-dim V11 action feature for a key")

        .def("get_rule_agent_action", &Engine::get_rule_agent_action)

        .def("get_payoffs", [](const Engine& e) {
            auto p = e.get_payoffs();
            return py::cast(std::vector<float>(p.begin(), p.end()));
        })

        .def("get_training_payoffs", [](const Engine& e) {
            auto p = e.get_training_payoffs();
            return py::cast(std::vector<float>(p.begin(), p.end()));
        })

        .def("get_step_rewards", [](const Engine& e, int p) {
            return py::cast(e.get_step_rewards(p));
        }, py::arg("player"))

        .def("get_aux_targets", [](const Engine& e) {
            auto targets = e.get_aux_targets();
            py::list result;
            for (int p = 0; p < NUM_PLAYERS; p++) {
                // Pack V11 aux: relation(3) + s3_owner(1) + sa_owner(1) = 5 int64
                py::array_t<int64_t> arr(5);
                auto buf = arr.mutable_data();
                for (int j = 0; j < 3; j++) buf[j] = targets[p].relation[j];
                buf[3] = targets[p].s3_owner;
                buf[4] = targets[p].sa_owner;
                result.append(arr);
            }
            return result;
        }, "Get V11 aux targets (4 players x 5 values: relation[3] + s3_owner + sa_owner)")
    ;

    // ─── VectorizedEngine ──────────────────────────────────────────────
    py::class_<VectorizedEngine>(m, "VectorizedEngine", py::module_local())
        .def(py::init<int>(), py::arg("num_envs"))
        .def("num_envs", &VectorizedEngine::num_envs)
        .def("set_greedy_ratio", &VectorizedEngine::set_greedy_ratio)
        .def("set_random_ratio", &VectorizedEngine::set_random_ratio)
        .def("set_reward_shaping", &VectorizedEngine::set_reward_shaping)
        .def("seed", &VectorizedEngine::seed, py::arg("base_seed"))

        .def("reset", &VectorizedEngine::reset, py::arg("idx"))
        .def("set_rules", &VectorizedEngine::set_rules, py::arg("idx"),
             py::arg("declare_require_both_spades"), py::arg("straight_start_val"),
             py::arg("straight_end_val"))
        .def("step",  &VectorizedEngine::step,  py::arg("idx"), py::arg("action_key"))
        .def("get_player_id",       &VectorizedEngine::get_player_id,       py::arg("idx"))
        .def("is_over",             &VectorizedEngine::is_over,             py::arg("idx"))
        .def("is_rule_agent_seat",  &VectorizedEngine::is_rule_agent_seat,  py::arg("idx"), py::arg("pid"))
        .def("is_declaration_phase", &VectorizedEngine::is_declaration_phase, py::arg("idx"))
        .def("get_rule_agent_action", &VectorizedEngine::get_rule_agent_action, py::arg("idx"))

        .def("encode_obs", [](const VectorizedEngine& v, int idx, int pid) {
            py::array_t<int8_t> r(STATE_DIM);
            v.encode_obs(idx, pid, r.mutable_data());
            return r;
        }, py::arg("idx"), py::arg("player_id"))

        .def("get_action_feature", [](const VectorizedEngine& v, int idx, const std::string& k) {
            py::array_t<int8_t> r(ACTION_DIM);
            v.get_action_feature(idx, k, r.mutable_data());
            return r;
        }, py::arg("idx"), py::arg("key"))

        .def("get_payoffs", [](const VectorizedEngine& v, int idx) {
            auto p = v.get_payoffs(idx);
            return py::cast(std::vector<float>(p.begin(), p.end()));
        }, py::arg("idx"))

        .def("get_training_payoffs", [](const VectorizedEngine& v, int idx) {
            auto p = v.get_training_payoffs(idx);
            return py::cast(std::vector<float>(p.begin(), p.end()));
        }, py::arg("idx"))

        .def("get_step_rewards", [](const VectorizedEngine& v, int idx, int p) {
            return py::cast(v.get_step_rewards(idx, p));
        }, py::arg("idx"), py::arg("player"))

        .def("get_aux_targets", [](const VectorizedEngine& v, int idx) {
            auto t = v.get_aux_targets(idx);
            py::list result;
            for (int p = 0; p < NUM_PLAYERS; p++) {
                py::array_t<int64_t> arr(5);
                auto buf = arr.mutable_data();
                for (int j = 0; j < 3; j++) buf[j] = t[p].relation[j];
                buf[3] = t[p].s3_owner;
                buf[4] = t[p].sa_owner;
                result.append(arr);
            }
            return result;
        }, py::arg("idx"))

        .def("get_aux_targets_for_player", [](const VectorizedEngine& v, int idx, int pid) {
            auto t = v.get_aux_targets_for_player(idx, pid);
            py::array_t<int64_t> result(5);
            auto out = result.mutable_data();
            for (int j = 0; j < 3; j++) out[j] = t.relation[j];
            out[3] = t.s3_owner;
            out[4] = t.sa_owner;
            return result;
        }, py::arg("idx"), py::arg("player_id"))

        .def("step_random", &VectorizedEngine::step_random, py::arg("idx"))
        .def("step_choices", &VectorizedEngine::step_choices,
             py::arg("indices"), py::arg("choices"), py::call_guard<py::gil_scoped_release>())

        .def("step_batch", [](VectorizedEngine& v, const std::vector<int>& indices,
                              const std::vector<std::string>& keys) {
            if (indices.size() != keys.size())
                throw py::value_error("One action key is required per environment");
            for (const int idx : indices)
                if (idx < 0 || idx >= v.num_envs())
                    throw py::index_error("Environment index out of range");
            for (size_t row = 0; row < indices.size(); ++row)
                v.step(indices[row], keys[row]);
        }, py::arg("indices"), py::arg("keys"))

        .def("advance_batch", [](VectorizedEngine& v) {
            std::vector<int> done, pending, players;
            pending.reserve(v.num_envs());
            players.reserve(v.num_envs());
            py::list records;
            for (int idx = 0; idx < v.num_envs(); ++idx) {
                auto result = v.advance_to_decision(idx);
                const auto& steps = result.second;
                if (!steps.empty()) {
                    const int count = (int)steps.size();
                    py::array_t<int8_t> obs({count, STATE_DIM});
                    py::array_t<int8_t> actions({count, ACTION_DIM});
                    py::array_t<int64_t> auxiliary({count, 5});
                    std::vector<int> pids;
                    pids.reserve(count);
                    for (int row = 0; row < count; ++row) {
                        std::memcpy(obs.mutable_data() + row * STATE_DIM, steps[row].obs, STATE_DIM);
                        std::memcpy(actions.mutable_data() + row * ACTION_DIM, steps[row].action, ACTION_DIM);
                        auto* out = auxiliary.mutable_data() + row * 5;
                        for (int j = 0; j < 3; ++j) out[j] = steps[row].auxiliary.relation[j];
                        out[3] = steps[row].auxiliary.s3_owner;
                        out[4] = steps[row].auxiliary.sa_owner;
                        pids.push_back(steps[row].player_id);
                    }
                    records.append(py::make_tuple(idx, obs, pids, actions, auxiliary));
                }
                if (result.first) done.push_back(idx);
                else {
                    pending.push_back(idx);
                    players.push_back(v.get_player_id(idx));
                }
            }
            return py::make_tuple(done, pending, players, records);
        })

        .def("advance_to_decision", [](VectorizedEngine& v, int idx) {
            auto [done, steps] = v.advance_to_decision(idx);
            int n = (int)steps.size();
            py::array_t<int8_t> obs({n, (int)STATE_DIM});
            auto* ptr = obs.mutable_data();
            std::vector<int> pids(n);
            for (int i = 0; i < n; i++) {
                std::memcpy(ptr + i * STATE_DIM, steps[i].obs, STATE_DIM);
                pids[i] = steps[i].player_id;
            }
            return py::make_tuple(done, obs, py::cast(pids));
        }, py::arg("idx"))

        .def("advance_to_decision_with_data", [](VectorizedEngine& v, int idx) {
            auto [done, steps] = v.advance_to_decision(idx);
            int n = (int)steps.size();
            py::array_t<int8_t> obs({n, (int)STATE_DIM});
            py::array_t<int8_t> actions({n, (int)ACTION_DIM});
            py::array_t<int64_t> auxiliary({n, 5});
            std::vector<int> pids(n);
            for (int i = 0; i < n; i++) {
                std::memcpy(obs.mutable_data() + i * STATE_DIM, steps[i].obs, STATE_DIM);
                std::memcpy(actions.mutable_data() + i * ACTION_DIM, steps[i].action, ACTION_DIM);
                auto out = auxiliary.mutable_data() + i * 5;
                for (int j = 0; j < 3; j++) out[j] = steps[i].auxiliary.relation[j];
                out[3] = steps[i].auxiliary.s3_owner;
                out[4] = steps[i].auxiliary.sa_owner;
                pids[i] = steps[i].player_id;
            }
            return py::make_tuple(done, obs, py::cast(pids), actions, auxiliary);
        }, py::arg("idx"))

        .def("prepare_batch", [](const VectorizedEngine& v,
                                 const std::vector<int>& pending) {
            return prepare_numpy_batch(v, pending, false);
        }, py::arg("pending_indices"))
        .def("prepare_batch_with_metadata", [](const VectorizedEngine& v,
                                               const std::vector<int>& pending) {
            return prepare_numpy_batch(v, pending, true);
        }, py::arg("pending_indices"))
        .def("prepare_indexed_batch", [](const VectorizedEngine& v,
                                         const std::vector<int>& pending) {
            return prepare_numpy_batch(v, pending, true, true);
        }, py::arg("pending_indices"))
    ;
}
